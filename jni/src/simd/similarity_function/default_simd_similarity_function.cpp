#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <stdint.h>
#include <cmath>

#include "simd_similarity_function_common.cpp"
#include "faiss_score_to_lucene_transform.cpp"
#include "parameter_utils.h"

//
// FP16
//


template <BulkScoreTransform BulkScoreTransformFunc, ScoreTransform ScoreTransformFunc>
struct DefaultFP16SimilarityFunction final : BaseSimilarityFunction<BulkScoreTransformFunc, ScoreTransformFunc> {
    void calculateSimilarityInBulk(SimdVectorSearchContext* srchContext,
                                   int32_t* internalVectorIds,
                                   float* scores,
                                   const int32_t numVectors) {

        // Prepare similarity calculation
        auto func = dynamic_cast<faiss::ScalarQuantizer::SQDistanceComputer*>(srchContext->faissFunction.get());
        knn_jni::util::ParameterCheck::require_non_null(
            func, "Unexpected distance function acquired. Expected SQDistanceComputer, but it was something else");

        for (int32_t i = 0 ; i < numVectors ; ++i) {
            // Calculate distance
            auto vector = reinterpret_cast<uint8_t*>(srchContext->getVectorPointer(internalVectorIds[i]));
            scores[i] = func->query_to_code(vector);
        }

        // Transform score values if it needs to
        BulkScoreTransformFunc(scores, numVectors);
    }
};

//
// FP16
//
// 1. Max IP
DefaultFP16SimilarityFunction<FaissScoreToLuceneScoreTransform::ipToMaxIpTransformBulk, FaissScoreToLuceneScoreTransform::ipToMaxIpTransform> DEFAULT_FP16_MAX_INNER_PRODUCT_SIMIL_FUNC;
// 2. L2
DefaultFP16SimilarityFunction<FaissScoreToLuceneScoreTransform::l2TransformBulk, FaissScoreToLuceneScoreTransform::l2Transform> DEFAULT_FP16_L2_SIMIL_FUNC;


//
// SQ (ADC: 4-bit query x 1-bit data) - Default (non-SIMD) implementation
//
// Pure C++ implementation written for compiler auto-vectorization with -O3.
// No SIMD intrinsics. Uses __builtin_popcount[ll] and std::memcpy.
//
// The query is 4-bit quantized and transposed into 4 bit planes (via transposeHalfByte).
// Each bit plane has `binaryCodeBytes` bytes. The int4BitDotProduct computes:
//   Result = popcount(plane0 AND data) * 1
//          + popcount(plane1 AND data) * 2
//          + popcount(plane2 AND data) * 4
//          + popcount(plane3 AND data) * 8
//

static constexpr float FOUR_BIT_SCALE = 1.0f / 15.0f;

// Reads the per-vector correction factors from a potentially unaligned address.
// On-disk layout after binaryCode: [lowerInterval(f32)][upperInterval(f32)][additionalCorrection(f32)][quantizedComponentSum(i32)]
static FORCE_INLINE void readDataCorrections(const uint8_t* ptr, float& ax, float& lx, float& additional, float& x1) {
    float lower, upper;
    std::memcpy(&lower,      ptr,      sizeof(float));
    std::memcpy(&upper,      ptr + 4,  sizeof(float));
    std::memcpy(&additional, ptr + 8,  sizeof(float));
    int32_t componentSum;
    std::memcpy(&componentSum, ptr + 12, sizeof(int32_t));
    ax = lower;
    lx = upper - lower;
    x1 = static_cast<float>(componentSum);
}

// Scalar int4BitDotProduct
// q has 4 * binaryCodeBytes bytes (4 bit planes), d has binaryCodeBytes bytes.
// Uses std::memcpy for uint64_t loads to avoid undefined behavior from unaligned
// reinterpret_cast when binaryCodeBytes is not a multiple of 8.
static FORCE_INLINE int64_t int4BitDotProduct(const uint8_t* q, const uint8_t* d, const int32_t binaryCodeBytes) {
    int64_t result = 0;
    for (int32_t bitPlane = 0 ; bitPlane < 4 ; ++bitPlane) {
        const int32_t words = binaryCodeBytes >> 3;

        int64_t subResult = 0;
        for (int32_t w = 0 ; w < words ; ++w) {
            uint64_t qWord, dWord;
            std::memcpy(&qWord, q + bitPlane * binaryCodeBytes + w * 8, sizeof(uint64_t));
            std::memcpy(&dWord, d + w * 8, sizeof(uint64_t));
            subResult += __builtin_popcountll(qWord & dWord);
        }

        const int32_t remainStart = words * 8;
        for (int32_t r = remainStart ; r < binaryCodeBytes ; ++r) {
            subResult += __builtin_popcount((q[bitPlane * binaryCodeBytes + r] & d[r]) & 0xFF);
        }

        result += subResult << bitPlane;
    }
    return result;
}

// Σ_i popcount(qplane_i AND docPlane) << i over `planeBytes` bytes. Reusable primitive
// for both B=1 (single call per doc) and B=2 (two calls, one per doc plane).
static FORCE_INLINE int64_t fourPlanesVsOneDocScalar(
    const uint8_t* qplane0, const uint8_t* qplane1,
    const uint8_t* qplane2, const uint8_t* qplane3,
    const uint8_t* docPlane, const int32_t planeBytes) {
    const int32_t words = planeBytes >> 3;
    const int32_t remainStart = words * 8;
    int64_t acc = 0;

    for (int32_t w = 0 ; w < words ; ++w) {
        const int32_t offset = w * 8;
        uint64_t dWord;
        std::memcpy(&dWord, docPlane + offset, sizeof(uint64_t));

        uint64_t q0, q1, q2, q3;
        std::memcpy(&q0, qplane0 + offset, sizeof(uint64_t));
        std::memcpy(&q1, qplane1 + offset, sizeof(uint64_t));
        std::memcpy(&q2, qplane2 + offset, sizeof(uint64_t));
        std::memcpy(&q3, qplane3 + offset, sizeof(uint64_t));

        acc += __builtin_popcountll(q0 & dWord) * 1
             + __builtin_popcountll(q1 & dWord) * 2
             + __builtin_popcountll(q2 & dWord) * 4
             + __builtin_popcountll(q3 & dWord) * 8;
    }

    for (int32_t r = remainStart ; r < planeBytes ; ++r) {
        uint8_t db = docPlane[r];
        acc += __builtin_popcount((qplane0[r] & db) & 0xFF) * 1
             + __builtin_popcount((qplane1[r] & db) & 0xFF) * 2
             + __builtin_popcount((qplane2[r] & db) & 0xFF) * 4
             + __builtin_popcount((qplane3[r] & db) & 0xFF) * 8;
    }

    return acc;
}

// Scalar reference — mirrors Lucene's int4DibitDotProductImpl
// (DefaultVectorUtilSupport.java:303): dp = sum_over_plane0 + (sum_over_plane1 << 1).
static FORCE_INLINE int64_t int4DibitDotProduct(const uint8_t* q, const uint8_t* d, const int32_t planeBytes) {
    const uint8_t* qplane0 = q;
    const uint8_t* qplane1 = q + planeBytes;
    const uint8_t* qplane2 = q + 2 * planeBytes;
    const uint8_t* qplane3 = q + 3 * planeBytes;
    const int64_t ret0 = fourPlanesVsOneDocScalar(qplane0, qplane1, qplane2, qplane3, d,              planeBytes);
    const int64_t ret1 = fourPlanesVsOneDocScalar(qplane0, qplane1, qplane2, qplane3, d + planeBytes, planeBytes);
    return ret0 + (ret1 << 1);
}

// Default (non-SIMD) batched int4BitDotProduct.
// Processes one batch element at a time with the byte loop as the inner loop,
// giving the compiler the best auto-vectorization opportunity across the byte dimension.
template <int BATCH_SIZE>
static FORCE_INLINE void default4bitDotProductBatch(
    const uint8_t* queryPtr,
    uint8_t** dataVecs,
    const int32_t planeBytes,
    float* results) {

    const uint8_t* plane0 = queryPtr;
    const uint8_t* plane1 = queryPtr + planeBytes;
    const uint8_t* plane2 = queryPtr + 2 * planeBytes;
    const uint8_t* plane3 = queryPtr + 3 * planeBytes;

    for (int32_t b = 0 ; b < BATCH_SIZE ; ++b) {
        results[b] = static_cast<float>(
            fourPlanesVsOneDocScalar(plane0, plane1, plane2, plane3, dataVecs[b], planeBytes));
    }
}

template <int BATCH_SIZE>
static FORCE_INLINE void default4bitDibitDotProductBatch(
    const uint8_t* queryPtr,
    uint8_t** dataVecs,
    const int32_t planeBytes,
    float* results) {

    const uint8_t* plane0 = queryPtr;
    const uint8_t* plane1 = queryPtr + planeBytes;
    const uint8_t* plane2 = queryPtr + 2 * planeBytes;
    const uint8_t* plane3 = queryPtr + 3 * planeBytes;

    for (int32_t b = 0 ; b < BATCH_SIZE ; ++b) {
        const int64_t sum0 = fourPlanesVsOneDocScalar(plane0, plane1, plane2, plane3,
                                                     dataVecs[b], planeBytes);
        const int64_t sum1 = fourPlanesVsOneDocScalar(plane0, plane1, plane2, plane3,
                                                     dataVecs[b] + planeBytes, planeBytes);
        results[b] = static_cast<float>(sum0 + (sum1 << 1));
    }
}

// Doc-side reconstruction scale: 1 / (2^docBits - 1). Scales `lx = upper - lower`
// so the delta matches the doc's quantization level range.
static FORCE_INLINE float docBitsScale(int32_t docBits) {
    if (docBits == 1) return 1.0f;
    if (docBits == 2) return 1.0f / 3.0f;
    if (docBits == 4) return FOUR_BIT_SCALE;
    return 1.0f;
}

static FORCE_INLINE int64_t sqDotProductScalar(const uint8_t* query, const uint8_t* doc,
                                               const int32_t planeBytes, const int32_t docBits) {
    if (docBits == 1) return int4BitDotProduct(query, doc, planeBytes);
    if (docBits == 2) return int4DibitDotProduct(query, doc, planeBytes);
    // B=4 (PACKED_NIBBLE) added in a follow-up commit.
    return 0;
}

template <bool IsMaxIP>
struct DefaultSQSimilarityFunction final : SimilarityFunction {
    HOT_SPOT void calculateSimilarityInBulk(SimdVectorSearchContext* srchContext,
                                            int32_t* internalVectorIds,
                                            float* scores,
                                            const int32_t numVectors) {
        const auto* queryPtr = reinterpret_cast<const uint8_t*>(srchContext->queryVectorSimdAligned);
        const int32_t dim = srchContext->dimension;
        const int32_t planeBytes = (dim + 7) / 8;
        const int32_t docBits = srchContext->docBits;
        const int32_t docPackedLength = docBits * planeBytes;
        const float docScale = docBitsScale(docBits);

        // Read query correction factors from tmpBuffer
        const auto* queryCorrectionPtr = reinterpret_cast<const float*>(srchContext->tmpBuffer.data());
        const float ay = queryCorrectionPtr[0];
        const float ly = (queryCorrectionPtr[1] - queryCorrectionPtr[0]) * FOUR_BIT_SCALE;
        const float queryAdditional = queryCorrectionPtr[2];
        int32_t y1Raw; std::memcpy(&y1Raw, &queryCorrectionPtr[3], sizeof(int32_t));
        const float y1 = static_cast<float>(y1Raw);
        const float centroidDp = queryCorrectionPtr[4];

        int32_t processedCount = 0;
        constexpr int32_t vecBlock = 8;
        constexpr int32_t vecHalfBlock = 4;
        uint8_t* vectors[vecBlock];

        // Batch size 8
        for ( ; (processedCount + vecBlock) <= numVectors ; processedCount += vecBlock) {
            srchContext->getVectorPointersInBulk(vectors, &internalVectorIds[processedCount], vecBlock);
            if (docBits == 2) {
                default4bitDibitDotProductBatch<vecBlock>(queryPtr, vectors, planeBytes, &scores[processedCount]);
            } else {
                default4bitDotProductBatch<vecBlock>(queryPtr, vectors, planeBytes, &scores[processedCount]);
            }

            for (int32_t i = 0 ; i < vecBlock ; ++i) {
                float ax, lxRaw, additional, x1;
                readDataCorrections(vectors[i] + docPackedLength, ax, lxRaw, additional, x1);
                const float lx = lxRaw * docScale;

                scores[processedCount + i] = ax * ay * dim
                                           + ay * lx * x1
                                           + ax * ly * y1
                                           + lx * ly * scores[processedCount + i];

                if constexpr (IsMaxIP) {
                    scores[processedCount + i] += queryAdditional + additional - centroidDp;
                } else {
                    scores[processedCount + i] = std::max(0.0F, queryAdditional + additional - 2 * scores[processedCount + i]);
                }
            }
        }

        // Batch size 4
        for ( ; (processedCount + vecHalfBlock) <= numVectors ; processedCount += vecHalfBlock) {
            srchContext->getVectorPointersInBulk(vectors, &internalVectorIds[processedCount], vecHalfBlock);
            if (docBits == 2) {
                default4bitDibitDotProductBatch<vecHalfBlock>(queryPtr, vectors, planeBytes, &scores[processedCount]);
            } else {
                default4bitDotProductBatch<vecHalfBlock>(queryPtr, vectors, planeBytes, &scores[processedCount]);
            }

            for (int32_t i = 0 ; i < vecHalfBlock ; ++i) {
                float ax, lxRaw, additional, x1;
                readDataCorrections(vectors[i] + docPackedLength, ax, lxRaw, additional, x1);
                const float lx = lxRaw * docScale;

                scores[processedCount + i] = ax * ay * dim
                                           + ay * lx * x1
                                           + ax * ly * y1
                                           + lx * ly * scores[processedCount + i];

                if constexpr (IsMaxIP) {
                    scores[processedCount + i] += queryAdditional + additional - centroidDp;
                } else {
                    scores[processedCount + i] =
                        std::max(0.0F, queryAdditional + additional - 2 * scores[processedCount + i]);
                }
            }
        }

        // Tail: remaining vectors (scalar)
        for ( ; processedCount < numVectors ; ++processedCount) {
            const auto* dataVec = srchContext->getVectorPointer(internalVectorIds[processedCount]);
            const float qcDist = static_cast<float>(
                sqDotProductScalar(queryPtr, dataVec, planeBytes, docBits));

            float ax, lxRaw, additional, x1;
            readDataCorrections(dataVec + docPackedLength, ax, lxRaw, additional, x1);
            const float lx = lxRaw * docScale;

            scores[processedCount] = ax * ay * dim
                                   + ay * lx * x1
                                   + ax * ly * y1
                                   + lx * ly * qcDist;

            if constexpr (IsMaxIP) {
                scores[processedCount] += queryAdditional + additional - centroidDp;
            } else {
                scores[processedCount] =
                    std::max(0.0F, queryAdditional + additional - 2 * scores[processedCount]);
            }
        }

        if constexpr (IsMaxIP) {
            FaissScoreToLuceneScoreTransform::ipToMaxIpTransformBulk(scores, numVectors);
        } else {
            FaissScoreToLuceneScoreTransform::l2TransformBulk(scores, numVectors);
        }
    }

    float calculateSimilarity(SimdVectorSearchContext* srchContext, const int32_t internalVectorId) {
        const auto* queryPtr = reinterpret_cast<const uint8_t*>(srchContext->queryVectorSimdAligned);
        const int32_t dim = srchContext->dimension;
        const int32_t planeBytes = (dim + 7) / 8;
        const int32_t docBits = srchContext->docBits;
        const int32_t docPackedLength = docBits * planeBytes;
        const float docScale = docBitsScale(docBits);

        const auto* queryCorrectionPtr = reinterpret_cast<const float*>(srchContext->tmpBuffer.data());
        const float ay = queryCorrectionPtr[0];
        const float ly = (queryCorrectionPtr[1] - queryCorrectionPtr[0]) * FOUR_BIT_SCALE;
        const float queryAdditional = queryCorrectionPtr[2];
        int32_t y1Raw; std::memcpy(&y1Raw, &queryCorrectionPtr[3], sizeof(int32_t));
        const float y1 = static_cast<float>(y1Raw);
        const float centroidDp = queryCorrectionPtr[4];

        const auto* dataVec = srchContext->getVectorPointer(internalVectorId);
        const float qcDist = static_cast<float>(
            sqDotProductScalar(queryPtr, dataVec, planeBytes, docBits));

        float ax, lxRaw, additional, x1;
        readDataCorrections(dataVec + docPackedLength, ax, lxRaw, additional, x1);
        const float lx = lxRaw * docScale;

        float score = ax * ay * dim
                    + ay * lx * x1
                    + ax * ly * y1
                    + lx * ly * qcDist;

        if constexpr (IsMaxIP) {
            score += queryAdditional + additional - centroidDp;
            return FaissScoreToLuceneScoreTransform::ipToMaxIpTransform(score);
        } else {
            score = std::max(0.0F, queryAdditional + additional - 2 * score);
            return FaissScoreToLuceneScoreTransform::l2Transform(score);
        }
    }
};

//
// SQ
//
// 1. Max IP
DefaultSQSimilarityFunction<true> SQ_IP_SIMIL_FUNC;
// 2. L2
DefaultSQSimilarityFunction<false> SQ_L2_SIMIL_FUNC;

#ifndef __NO_SELECT_FUNCTION
SimilarityFunction* SimilarityFunction::selectSimilarityFunction(const NativeSimilarityFunctionType nativeFunctionType) {
    if (nativeFunctionType == NativeSimilarityFunctionType::FP16_MAXIMUM_INNER_PRODUCT) {
        return &DEFAULT_FP16_MAX_INNER_PRODUCT_SIMIL_FUNC;
    } else if (nativeFunctionType == NativeSimilarityFunctionType::FP16_L2) {
        return &DEFAULT_FP16_L2_SIMIL_FUNC;
    } else if (nativeFunctionType == NativeSimilarityFunctionType::SQ_IP) {
        return &SQ_IP_SIMIL_FUNC;
    } else if (nativeFunctionType == NativeSimilarityFunctionType::SQ_L2) {
        return &SQ_L2_SIMIL_FUNC;
    }

    throw std::runtime_error("Invalid native similarity function type was given, nativeFunctionType="
                             + std::to_string(static_cast<int32_t>(nativeFunctionType)));
}
#endif
