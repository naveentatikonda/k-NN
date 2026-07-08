#ifndef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#define __ARM_FEATURE_FP16_VECTOR_ARITHMETIC 1
#endif
#include <arm_neon.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <stdint.h>
#include <cmath>
#include <iostream>

#include "simd_similarity_function_common.cpp"
#include "faiss_score_to_lucene_transform.cpp"



//
// FP16
//

template <BulkScoreTransform BulkScoreTransformFunc, ScoreTransform ScoreTransformFunc>
struct ArmNeonFP16MaxIP final : BaseSimilarityFunction<BulkScoreTransformFunc, ScoreTransformFunc> {
    void calculateSimilarityInBulk(SimdVectorSearchContext* srchContext,
                                   int32_t* internalVectorIds,
                                   float* scores,
                                   const int32_t numVectors) {
        // Bulk inner product with 4 batch
        int32_t processedCount = 0;
        constexpr int32_t vecBlock = 4;
        const uint8_t* vectors[vecBlock];
        const auto* queryPtr = (const float*) srchContext->queryVectorSimdAligned;
        const int32_t dim = srchContext->dimension;
        constexpr int32_t dimensionBatch = 8;

        for ( ; (processedCount + vecBlock) <= numVectors ; processedCount += vecBlock) {
            srchContext->getVectorPointersInBulk((uint8_t**)vectors, &internalVectorIds[processedCount], vecBlock);

            // Score accumulator per each vector
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f);
            float32x4_t acc3 = vdupq_n_f32(0.0f);

            // Batch inner product for 8 values
            int32_t i = 0;
            for (; i + dimensionBatch <= dim; i += dimensionBatch) {
                // Load 8 FP32 query elements
                float32x4_t q0 = vld1q_f32(queryPtr + i);
                float32x4_t q1 = vld1q_f32(queryPtr + i + 4);

                // Load 8 FP16 elements from each target and convert to FP32
                float16x8_t h0 = vld1q_f16((const __fp16 *)(vectors[0] + i * 2));
                float16x8_t h1 = vld1q_f16((const __fp16 *)(vectors[1] + i * 2));
                float16x8_t h2 = vld1q_f16((const __fp16 *)(vectors[2] + i * 2));
                float16x8_t h3 = vld1q_f16((const __fp16 *)(vectors[3] + i * 2));
                float32x4_t d0_lo = vcvt_f32_f16(vget_low_f16(h0));
                float32x4_t d0_hi = vcvt_f32_f16(vget_high_f16(h0));
                float32x4_t d1_lo = vcvt_f32_f16(vget_low_f16(h1));
                float32x4_t d1_hi = vcvt_f32_f16(vget_high_f16(h1));
                float32x4_t d2_lo = vcvt_f32_f16(vget_low_f16(h2));
                float32x4_t d2_hi = vcvt_f32_f16(vget_high_f16(h2));
                float32x4_t d3_lo = vcvt_f32_f16(vget_low_f16(h3));
                float32x4_t d3_hi = vcvt_f32_f16(vget_high_f16(h3));

                // Post-load prefetch: next 8 elements
                // By the time in the next loop,
                if (i + dimensionBatch < dim) {
                    __builtin_prefetch(queryPtr + i + 8);
                    __builtin_prefetch(vectors[0] + (i + 8) * 2);
                    __builtin_prefetch(vectors[1] + (i + 8) * 2);
                    __builtin_prefetch(vectors[2] + (i + 8) * 2);
                    __builtin_prefetch(vectors[3] + (i + 8) * 2);
                }

                // Accumulate FMA
                acc0 = vfmaq_f32(acc0, q0, d0_lo);
                acc0 = vfmaq_f32(acc0, q1, d0_hi);

                acc1 = vfmaq_f32(acc1, q0, d1_lo);
                acc1 = vfmaq_f32(acc1, q1, d1_hi);

                acc2 = vfmaq_f32(acc2, q0, d2_lo);
                acc2 = vfmaq_f32(acc2, q1, d2_hi);

                acc3 = vfmaq_f32(acc3, q0, d3_lo);
                acc3 = vfmaq_f32(acc3, q1, d3_hi);
            }

            // Horizontal sum
            scores[processedCount] = vaddvq_f32(acc0);
            scores[processedCount + 1] = vaddvq_f32(acc1);
            scores[processedCount + 2] = vaddvq_f32(acc2);
            scores[processedCount + 3] = vaddvq_f32(acc3);

            // Scalar tail.
            // For example,
            // if dimension was 66 then this loop will take care of remaining 2 values.
            for (; i < dim; i++) {
                __fp16 h0 = *((const __fp16 *)(vectors[0] + i * 2));
                __fp16 h1 = *((const __fp16 *)(vectors[1] + i * 2));
                __fp16 h2 = *((const __fp16 *)(vectors[2] + i * 2));
                __fp16 h3 = *((const __fp16 *)(vectors[3] + i * 2));
                const float qv = queryPtr[i];
                scores[processedCount] += qv * (float)h0;
                scores[processedCount + 1] += qv * (float)h1;
                scores[processedCount + 2] += qv * (float)h2;
                scores[processedCount + 3] += qv * (float)h3;
            }
        }

        // Tail loop for remaining vectors
        for (; processedCount < numVectors; ++processedCount) {
            const auto* vecPtr = (const __fp16*) srchContext->getVectorPointer(internalVectorIds[processedCount]);
            float32x4_t acc = vdupq_n_f32(0.0f);
            int32_t i = 0;
            for (; i <= dim - dimensionBatch; i += dimensionBatch) {
                float32x4_t q0 = vld1q_f32(queryPtr + i);
                float32x4_t q1 = vld1q_f32(queryPtr + i + 4);
                float16x8_t h0 = vld1q_f16((const __fp16 *)(vecPtr + i));
                float32x4_t d0_lo = vcvt_f32_f16(vget_low_f16(h0));
                float32x4_t d0_hi = vcvt_f32_f16(vget_high_f16(h0));
                acc = vfmaq_f32(acc, q0, d0_lo);
                acc = vfmaq_f32(acc, q1, d0_hi);
            }

            float finalSum = vaddvq_f32(acc);
            // Scalar tail for dimensions not divisible by 4
            for (; i < dim; ++i) {
                finalSum += queryPtr[i] * (float)vecPtr[i];
            }
            scores[processedCount] = finalSum;
        }

        BulkScoreTransformFunc(scores, numVectors);
    }
};

template <BulkScoreTransform BulkScoreTransformFunc, ScoreTransform ScoreTransformFunc>
struct ArmNeonFP16L2 final : BaseSimilarityFunction<BulkScoreTransformFunc, ScoreTransformFunc> {
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
// SQ (ADC: 4-bit query x 1-bit data) - NEON SIMD implementation
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
// Because oneVectorByteSize may not be a multiple of 4, subsequent vectors can start at
// non-4-byte-aligned offsets, making reinterpret_cast<float*> undefined behaviour.
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

// Scalar fallback for int4BitDotProduct
// q has 4 * binaryCodeBytes bytes (4 bit planes), d has binaryCodeBytes bytes
// Uses std::memcpy for uint64_t loads to avoid undefined behavior from unaligned
// reinterpret_cast when binaryCodeBytes is not a multiple of 8. Compilers optimize
// the 8-byte memcpy into a single mov instruction — zero runtime cost.
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

// Per-byte weighted popcount of 4 query planes against a single doc register.
// For each of the 16 bytes:
//   result[byte] = popcount(q0[byte] & d[byte]) * 1
//                + popcount(q1[byte] & d[byte]) * 2
//                + popcount(q2[byte] & d[byte]) * 4
//                + popcount(q3[byte] & d[byte]) * 8
// Max value per byte: 8*(1 + 2 + 4 + 8) = 120, fits in uint8_t.
//
// This is the reusable primitive for both B=1 (called once per doc chunk) and
// B=2 (called twice — once per doc plane — with the second result weighted by <<1).
static FORCE_INLINE uint8x16_t weightedPopcount4PlanesVs1Doc(
    const uint8x16_t q0, const uint8x16_t q1, const uint8x16_t q2, const uint8x16_t q3,
    const uint8x16_t d) {
    uint8x16_t p0 = vcntq_u8(vandq_u8(q0, d));
    uint8x16_t p1 = vcntq_u8(vandq_u8(q1, d));
    uint8x16_t p2 = vcntq_u8(vandq_u8(q2, d));
    uint8x16_t p3 = vcntq_u8(vandq_u8(q3, d));
    uint8x16_t weighted = vaddq_u8(p0, vshlq_n_u8(p1, 1));
    weighted = vaddq_u8(weighted, vshlq_n_u8(p2, 2));
    weighted = vaddq_u8(weighted, vshlq_n_u8(p3, 3));
    return weighted;
}

// NEON SIMD batched int4BitDotProduct (B=1 doc, 4-bit query).
template <int BATCH_SIZE>
static FORCE_INLINE void simd4bitDotProductBatch(
    const uint8_t* queryPtr,
    uint8_t** dataVecs,
    const int32_t binaryCodeBytes,
    float* results) {

    const uint8_t* plane0 = queryPtr;
    const uint8_t* plane1 = queryPtr + binaryCodeBytes;
    const uint8_t* plane2 = queryPtr + 2 * binaryCodeBytes;
    const uint8_t* plane3 = queryPtr + 3 * binaryCodeBytes;

    uint32x4_t acc[BATCH_SIZE];
    for (int32_t b = 0 ; b < BATCH_SIZE ; ++b) {
        acc[b] = vdupq_n_u32(0);
    }

    int32_t i = 0;
    for ( ; i + 16 <= binaryCodeBytes ; i += 16) {
        // Load 16 bytes from each query plane (shared across all data vectors)
        uint8x16_t q0 = vld1q_u8(plane0 + i);
        uint8x16_t q1 = vld1q_u8(plane1 + i);
        uint8x16_t q2 = vld1q_u8(plane2 + i);
        uint8x16_t q3 = vld1q_u8(plane3 + i);

        // Prefetch next chunk — issued once before the batch loop so we don't
        // flood the prefetch queue (ARM cores typically have 4-8 slots).
        if (i + 16 < binaryCodeBytes) {
            __builtin_prefetch(plane0 + i + 16);
            for (int32_t b = 0 ; b < BATCH_SIZE ; ++b) {
                __builtin_prefetch(dataVecs[b] + i + 16);
            }
        }

        for (int32_t b = 0 ; b < BATCH_SIZE ; ++b) {
            uint8x16_t d = vld1q_u8(dataVecs[b] + i);
            uint8x16_t weighted = weightedPopcount4PlanesVs1Doc(q0, q1, q2, q3, d);
            // Widen and accumulate: u8 -> u16 -> u32
            acc[b] = vaddq_u32(acc[b], vpaddlq_u16(vpaddlq_u8(weighted)));
        }
    }

    // Horizontal sum into results
    for (int32_t b = 0 ; b < BATCH_SIZE ; ++b) {
        results[b] = static_cast<float>(vaddvq_u32(acc[b]));
    }

    // Scalar tail for remaining bytes (< 16)
    for ( ; i < binaryCodeBytes ; ++i) {
        uint8_t q0b = plane0[i], q1b = plane1[i], q2b = plane2[i], q3b = plane3[i];
        for (int32_t b = 0 ; b < BATCH_SIZE ; ++b) {
            uint8_t db = dataVecs[b][i];
            results[b] += static_cast<float>(
                __builtin_popcount((q0b & db) & 0xFF) * 1
              + __builtin_popcount((q1b & db) & 0xFF) * 2
              + __builtin_popcount((q2b & db) & 0xFF) * 4
              + __builtin_popcount((q3b & db) & 0xFF) * 8);
        }
    }
}

// Scalar reference — mirrors Lucene's int4DotProductSinglePacked
// (DefaultVectorUtilSupport.java:170). Used for the batch tail so the SIMD kernel
// stays bit-identical to Lucene on partial-chunk vectors.
//
// `unpacked` holds one 4-bit element per byte (query side, length = discretized_dim).
// `packed`   holds two 4-bit elements per byte (doc side, length = discretized_dim / 2).
// The high nibble of packed[i] pairs with unpacked[i]; the low nibble of packed[i]
// pairs with unpacked[i + packedLen] — the same pairing packNibbles produced.
static FORCE_INLINE int64_t int4DotProductSinglePacked(const uint8_t* unpacked, const uint8_t* packed, const int32_t packedLen) {
    int64_t total = 0;
    for (int32_t i = 0 ; i < packedLen ; ++i) {
        const uint8_t p = packed[i];
        const uint8_t u1 = unpacked[i];                // pairs with high nibble
        const uint8_t u2 = unpacked[i + packedLen];    // pairs with low nibble
        total += static_cast<int64_t>((p & 0x0Fu)) * u2;
        total += static_cast<int64_t>((p >> 4) & 0x0Fu) * u1;
    }
    return total;
}

// Scalar reference — mirrors Lucene's int4DibitDotProductImpl
// (DefaultVectorUtilSupport.java:303). Used for the batch tail so the SIMD kernel
// stays bit-identical to Lucene on partial-chunk vectors.
static FORCE_INLINE int64_t int4DibitDotProduct(const uint8_t* q, const uint8_t* d, const int32_t planeBytes) {
    const int64_t ret0 = int4BitDotProduct(q, d,              planeBytes);
    const int64_t ret1 = int4BitDotProduct(q, d + planeBytes, planeBytes);
    return ret0 + (ret1 << 1);
}

// NEON SIMD batched int4DibitDotProduct (B=2 doc, 4-bit query).
// Two doc planes are accumulated into separate u32 accumulators and combined as
// dp = sum0 + (sum1 << 1) at horizontal-sum time. Merging them per byte instead
// would overflow uint8_t: max = 120 + 2*120 = 360.
template <int BATCH_SIZE>
static FORCE_INLINE void simd4bitDibitDotProductBatch(
    const uint8_t* queryPtr,
    uint8_t** dataVecs,
    const int32_t planeBytes,
    float* results) {

    const uint8_t* qplane0 = queryPtr;
    const uint8_t* qplane1 = queryPtr + planeBytes;
    const uint8_t* qplane2 = queryPtr + 2 * planeBytes;
    const uint8_t* qplane3 = queryPtr + 3 * planeBytes;

    uint32x4_t accPlane0[BATCH_SIZE];
    uint32x4_t accPlane1[BATCH_SIZE];
    for (int32_t b = 0 ; b < BATCH_SIZE ; ++b) {
        accPlane0[b] = vdupq_n_u32(0);
        accPlane1[b] = vdupq_n_u32(0);
    }

    int32_t i = 0;
    for ( ; i + 16 <= planeBytes ; i += 16) {
        // Query planes are loaded once and reused across every batch element.
        uint8x16_t q0 = vld1q_u8(qplane0 + i);
        uint8x16_t q1 = vld1q_u8(qplane1 + i);
        uint8x16_t q2 = vld1q_u8(qplane2 + i);
        uint8x16_t q3 = vld1q_u8(qplane3 + i);

        if (i + 16 < planeBytes) {
            __builtin_prefetch(qplane0 + i + 16);
            for (int32_t b = 0 ; b < BATCH_SIZE ; ++b) {
                __builtin_prefetch(dataVecs[b] + i + 16);
                __builtin_prefetch(dataVecs[b] + planeBytes + i + 16);
            }
        }

        for (int32_t b = 0 ; b < BATCH_SIZE ; ++b) {
            uint8x16_t d0 = vld1q_u8(dataVecs[b] + i);
            uint8x16_t d1 = vld1q_u8(dataVecs[b] + planeBytes + i);
            uint8x16_t w0 = weightedPopcount4PlanesVs1Doc(q0, q1, q2, q3, d0);
            uint8x16_t w1 = weightedPopcount4PlanesVs1Doc(q0, q1, q2, q3, d1);
            accPlane0[b] = vaddq_u32(accPlane0[b], vpaddlq_u16(vpaddlq_u8(w0)));
            accPlane1[b] = vaddq_u32(accPlane1[b], vpaddlq_u16(vpaddlq_u8(w1)));
        }
    }

    for (int32_t b = 0 ; b < BATCH_SIZE ; ++b) {
        const uint32_t sum0 = vaddvq_u32(accPlane0[b]);
        const uint32_t sum1 = vaddvq_u32(accPlane1[b]);
        results[b] = static_cast<float>(sum0 + (static_cast<uint64_t>(sum1) << 1));
    }

    if (i < planeBytes) {
        for (int32_t b = 0 ; b < BATCH_SIZE ; ++b) {
            const uint8_t* d0Ptr = dataVecs[b];
            const uint8_t* d1Ptr = dataVecs[b] + planeBytes;
            int64_t sub0 = 0, sub1 = 0;
            for (int32_t r = i ; r < planeBytes ; ++r) {
                const uint8_t db0 = d0Ptr[r];
                const uint8_t db1 = d1Ptr[r];
                const int32_t w0 = __builtin_popcount((qplane0[r] & db0) & 0xFF) * 1
                                 + __builtin_popcount((qplane1[r] & db0) & 0xFF) * 2
                                 + __builtin_popcount((qplane2[r] & db0) & 0xFF) * 4
                                 + __builtin_popcount((qplane3[r] & db0) & 0xFF) * 8;
                const int32_t w1 = __builtin_popcount((qplane0[r] & db1) & 0xFF) * 1
                                 + __builtin_popcount((qplane1[r] & db1) & 0xFF) * 2
                                 + __builtin_popcount((qplane2[r] & db1) & 0xFF) * 4
                                 + __builtin_popcount((qplane3[r] & db1) & 0xFF) * 8;
                sub0 += w0;
                sub1 += w1;
            }
            results[b] += static_cast<float>(sub0 + (sub1 << 1));
        }
    }
}

// NEON SIMD batched int4DotProductSinglePacked (B=4 doc, 4-bit query).
// Query is one-byte-per-element (values in [0,15]); doc is packed-nibble (2 elts/byte).
// Per 16 packed bytes we cover 32 elements: 16 from the high-nibble half (elements
// `i..i+15`) and 16 from the low-nibble half (elements `packedLen+i..packedLen+i+15`).
// Two widening multiply-add lanes hit both halves in the same iteration.
// Max per byte-product: 15*15 = 225 (u8-safe pre-widen); u32 accumulator holds well
// past realistic dims (dim=65535 → ~7.4e6 max, u32 max ≈ 4.3e9).
template <int BATCH_SIZE>
static FORCE_INLINE void simd4bitPackedNibbleDotProductBatch(
    const uint8_t* queryPtr,
    uint8_t** dataVecs,
    const int32_t packedLen,
    float* results) {

    const uint8x16_t lowMask = vdupq_n_u8(0x0F);

    uint32x4_t acc[BATCH_SIZE];
    for (int32_t b = 0 ; b < BATCH_SIZE ; ++b) {
        acc[b] = vdupq_n_u32(0);
    }

    int32_t i = 0;
    for ( ; i + 16 <= packedLen ; i += 16) {
        // Query is shared across all batch elements — load once per outer iteration.
        // u1 pairs with the doc's HIGH nibble, u2 pairs with the LOW nibble.
        uint8x16_t u1 = vld1q_u8(queryPtr + i);
        uint8x16_t u2 = vld1q_u8(queryPtr + packedLen + i);

        if (i + 16 < packedLen) {
            __builtin_prefetch(queryPtr + i + 16);
            __builtin_prefetch(queryPtr + packedLen + i + 16);
            for (int32_t b = 0 ; b < BATCH_SIZE ; ++b) {
                __builtin_prefetch(dataVecs[b] + i + 16);
            }
        }

        for (int32_t b = 0 ; b < BATCH_SIZE ; ++b) {
            uint8x16_t p = vld1q_u8(dataVecs[b] + i);
            uint8x16_t hi = vshrq_n_u8(p, 4);
            uint8x16_t lo = vandq_u8(p, lowMask);

            // 8-bit × 8-bit → 16-bit widening multiply, half at a time.
            // vmull_u8 on the low half of two uint8x16_t gives 8 uint16 products.
            uint16x8_t hiProd0 = vmull_u8(vget_low_u8(hi), vget_low_u8(u1));
            uint16x8_t hiProd1 = vmull_u8(vget_high_u8(hi), vget_high_u8(u1));
            uint16x8_t loProd0 = vmull_u8(vget_low_u8(lo), vget_low_u8(u2));
            uint16x8_t loProd1 = vmull_u8(vget_high_u8(lo), vget_high_u8(u2));

            // Pairwise-add-and-accumulate u16 → u32 into the running acc.
            acc[b] = vpadalq_u16(acc[b], hiProd0);
            acc[b] = vpadalq_u16(acc[b], hiProd1);
            acc[b] = vpadalq_u16(acc[b], loProd0);
            acc[b] = vpadalq_u16(acc[b], loProd1);
        }
    }

    for (int32_t b = 0 ; b < BATCH_SIZE ; ++b) {
        results[b] = static_cast<float>(vaddvq_u32(acc[b]));
    }

    // Scalar tail — same pairing as the SIMD body so we stay bit-identical to Lucene
    // (int4DotProductSinglePacked).
    if (i < packedLen) {
        for (int32_t b = 0 ; b < BATCH_SIZE ; ++b) {
            int64_t sum = 0;
            for (int32_t r = i ; r < packedLen ; ++r) {
                const uint8_t p  = dataVecs[b][r];
                const uint8_t u1 = queryPtr[r];
                const uint8_t u2 = queryPtr[packedLen + r];
                sum += static_cast<int64_t>((p >> 4) & 0x0Fu) * u1;
                sum += static_cast<int64_t>(p & 0x0Fu) * u2;
            }
            results[b] += static_cast<float>(sum);
        }
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

// Dispatches to the width-specific scalar dot product. For bit-plane layouts (B=1, B=2)
// `sizeBytes` is planeBytes = (dim + 7) / 8. For B=4 it is packedLen = discretized_dim / 2.
static FORCE_INLINE int64_t sqDotProductScalar(const uint8_t* query, const uint8_t* doc,
                                               const int32_t sizeBytes, const int32_t docBits) {
    if (docBits == 1) return int4BitDotProduct(query, doc, sizeBytes);
    if (docBits == 2) return int4DibitDotProduct(query, doc, sizeBytes);
    if (docBits == 4) return int4DotProductSinglePacked(query, doc, sizeBytes);
    return 0;
}

// Kernel-size for the batched dot product loop. For bit-plane layouts (B=1, B=2) this
// is planeBytes (each of B planes has this many bytes). For packed-nibble (B=4) it is
// packedLen = discretized_dim / 2 (2 elements per byte).
static FORCE_INLINE int32_t computeKernelSize(int32_t dim, int32_t docBits) {
    if (docBits == 4) {
        // Match Lucene's PACKED_NIBBLE getDocPackedLength; see saveSearchContext body.
        const int32_t totalBits = ((dim * 4 + 7) / 8) * 8;
        return (totalBits + 7) / 8;
    }
    return (dim + 7) / 8; // planeBytes for B=1, B=2
}

// Where the correction factors start on-disk (offset from start of vector). For
// bit-plane layouts this is docBits * planeBytes; for packed-nibble it is packedLen.
static FORCE_INLINE int32_t computeDocPackedLength(int32_t kernelSize, int32_t docBits) {
    return (docBits == 4) ? kernelSize : docBits * kernelSize;
}

template <bool IsMaxIP>
struct ArmNeonSQSimilarityFunction final : SimilarityFunction {
    HOT_SPOT void calculateSimilarityInBulk(SimdVectorSearchContext* srchContext,
                                            int32_t* internalVectorIds,
                                            float* scores,
                                            const int32_t numVectors) {
        const auto* queryPtr = reinterpret_cast<const uint8_t*>(srchContext->queryVectorSimdAligned);
        const int32_t dim = srchContext->dimension;
        const int32_t docBits = srchContext->docBits;
        // kernelSize is what the batch kernel loops over (planeBytes for B=1/B=2,
        // packedLen for B=4). docPackedLength is the on-disk offset to correction
        // factors. They differ only for bit-plane B=2, where docPackedLength = 2 *
        // kernelSize.
        const int32_t kernelSize = computeKernelSize(dim, docBits);
        const int32_t docPackedLength = computeDocPackedLength(kernelSize, docBits);
        const float docScale = docBitsScale(docBits);

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

        for ( ; (processedCount + vecBlock) <= numVectors ; processedCount += vecBlock) {
            srchContext->getVectorPointersInBulk(vectors, &internalVectorIds[processedCount], vecBlock);
            if (docBits == 4) {
                simd4bitPackedNibbleDotProductBatch<vecBlock>(queryPtr, vectors, kernelSize, &scores[processedCount]);
            } else if (docBits == 2) {
                simd4bitDibitDotProductBatch<vecBlock>(queryPtr, vectors, kernelSize, &scores[processedCount]);
            } else {
                simd4bitDotProductBatch<vecBlock>(queryPtr, vectors, kernelSize, &scores[processedCount]);
            }

            for (int32_t i = 0 ; i < vecBlock ; ++i) {
                if ((i + 1) < vecBlock) {
                    __builtin_prefetch(vectors[i + 1] + docPackedLength);
                }
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
            if (docBits == 4) {
                simd4bitPackedNibbleDotProductBatch<vecHalfBlock>(queryPtr, vectors, kernelSize, &scores[processedCount]);
            } else if (docBits == 2) {
                simd4bitDibitDotProductBatch<vecHalfBlock>(queryPtr, vectors, kernelSize, &scores[processedCount]);
            } else {
                simd4bitDotProductBatch<vecHalfBlock>(queryPtr, vectors, kernelSize, &scores[processedCount]);
            }

            for (int32_t i = 0 ; i < vecHalfBlock ; ++i) {
                if ((i + 1) < vecHalfBlock) {
                    __builtin_prefetch(vectors[i + 1] + docPackedLength);
                }
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
                sqDotProductScalar(queryPtr, dataVec, kernelSize, docBits));

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
        const int32_t docBits = srchContext->docBits;
        const int32_t kernelSize = computeKernelSize(dim, docBits);
        const int32_t docPackedLength = computeDocPackedLength(kernelSize, docBits);
        const float docScale = docBitsScale(docBits);

        const auto* queryCorrectionPtr = reinterpret_cast<const float*>(srchContext->tmpBuffer.data());
        const float ay = queryCorrectionPtr[0];
        const float ly = (queryCorrectionPtr[1] - queryCorrectionPtr[0]) * FOUR_BIT_SCALE;
        const float queryAdditional = queryCorrectionPtr[2];
        int32_t y1Raw2; std::memcpy(&y1Raw2, &queryCorrectionPtr[3], sizeof(int32_t));
        const float y1 = static_cast<float>(y1Raw2);
        const float centroidDp = queryCorrectionPtr[4];

        const auto* dataVec = srchContext->getVectorPointer(internalVectorId);
        const float qcDist = static_cast<float>(
            sqDotProductScalar(queryPtr, dataVec, kernelSize, docBits));

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
// FP16
//
// 1. Max IP
ArmNeonFP16MaxIP<FaissScoreToLuceneScoreTransform::ipToMaxIpTransformBulk, FaissScoreToLuceneScoreTransform::ipToMaxIpTransform> FP16_MAX_INNER_PRODUCT_SIMIL_FUNC;
// 2. L2
ArmNeonFP16L2<FaissScoreToLuceneScoreTransform::l2TransformBulk, FaissScoreToLuceneScoreTransform::l2Transform> FP16_L2_SIMIL_FUNC;

//
// SQ
//
// 1. Max IP
ArmNeonSQSimilarityFunction<true> SQ_IP_SIMIL_FUNC;
// 2. L2
ArmNeonSQSimilarityFunction<false> SQ_L2_SIMIL_FUNC;

#ifndef __NO_SELECT_FUNCTION
SimilarityFunction* SimilarityFunction::selectSimilarityFunction(const NativeSimilarityFunctionType nativeFunctionType) {
    if (nativeFunctionType == NativeSimilarityFunctionType::FP16_MAXIMUM_INNER_PRODUCT) {
        return &FP16_MAX_INNER_PRODUCT_SIMIL_FUNC;
    } else if (nativeFunctionType == NativeSimilarityFunctionType::FP16_L2) {
        return &FP16_L2_SIMIL_FUNC;
    } else if (nativeFunctionType == NativeSimilarityFunctionType::SQ_IP) {
        return &SQ_IP_SIMIL_FUNC;
    } else if (nativeFunctionType == NativeSimilarityFunctionType::SQ_L2) {
        return &SQ_L2_SIMIL_FUNC;
    }

    throw std::runtime_error("Invalid native similarity function type was given, nativeFunctionType="
                             + std::to_string(static_cast<int32_t>(nativeFunctionType)));
}
#endif
