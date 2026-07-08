#ifndef KNNPLUGIN_JNI_FAISS_SQ_FLAT_H
#define KNNPLUGIN_JNI_FAISS_SQ_FLAT_H

#include "faiss/Index.h"
#include "faiss/IndexBinary.h"
#include "faiss/MetricType.h"
#include "faiss/impl/DistanceComputer.h"
#include "memory_util.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <iostream>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace knn_jni {

    // Reads correction factors from a potentially unaligned address using std::memcpy.
    // Layout at ptr: [lowerInterval(f32)][upperInterval(f32)][additionalCorrection(f32)][quantizedComponentSum(i32)]
    static inline void readCorrectionFactorsSafe(const uint8_t* ptr, float& lowerInterval,
        float& intervalLength, float& additionalCorrection, float& quantizedComponentSum) {
        float lower, upper;
        std::memcpy(&lower, ptr, sizeof(float));
        std::memcpy(&upper, ptr + 4, sizeof(float));
        std::memcpy(&additionalCorrection, ptr + 8, sizeof(float));
        int32_t componentSum;
        std::memcpy(&componentSum, ptr + 12, sizeof(int32_t));
        lowerInterval = lower;
        intervalLength = upper - lower;
        quantizedComponentSum = static_cast<float>(componentSum);
    }

    // Reads correction factors via direct pointer cast (only safe when ptr is 4-byte aligned).
    static inline void readCorrectionFactorsAligned(const uint8_t* ptr, float& lowerInterval,
        float& intervalLength, float& additionalCorrection, float& quantizedComponentSum) {
        const auto* correctionFactors = reinterpret_cast<const float*>(ptr);
        lowerInterval = correctionFactors[0];
        intervalLength = correctionFactors[1] - correctionFactors[0];
        additionalCorrection = correctionFactors[2];
        int32_t componentSum;
        std::memcpy(&componentSum, ptr + 12, sizeof(int32_t));
        quantizedComponentSum = static_cast<float>(componentSum);
    }

    template <bool IsMaxIP, bool IsBytesMultipleOf8>
    struct FaissSQDistanceComputer final : faiss::DistanceComputer {
        const int64_t oneElementByteSize;
        const uint64_t quantizedVectorBytes;
        // Number of bits used to quantize each document dimension (1, 2, or 4).
        const int32_t docBits;
        // Byte length of a single bit plane. The quantized code is docBits contiguous planes,
        // each planeBytes long: quantizedVectorBytes == docBits * planeBytes.
        const uint64_t planeBytes;
        // Reconstruction scale for the quantization interval: 1 / (2^docBits - 1). For docBits == 1
        // this is 1, preserving the legacy single-bit behavior.
        const float dataScale;
        const uint8_t* data;
        const uint8_t* query;
        const float centroidDp;
        float ay;
        float ly;
        float queryAdditional;
        float y1;
        int32_t dimension;
        int32_t numVectors;

        FaissSQDistanceComputer(int32_t _oneElementByteSize, const void* _data, float _centroidDp, int32_t _dimension, int32_t _numVectors, int32_t _docBits)
          : faiss::DistanceComputer(),
            oneElementByteSize(_oneElementByteSize),
            // Memory layout : [Quantized Vector | lowerInterval (float) | upperInterval (float) | additionalCorrection (float) | quantizedComponentSum (int)]
            quantizedVectorBytes(_oneElementByteSize - (sizeof(float) * 3 + sizeof(int32_t))),
            docBits(_docBits),
            planeBytes((_oneElementByteSize - (sizeof(float) * 3 + sizeof(int32_t))) / _docBits),
            dataScale(1.0f / static_cast<float>((1 << _docBits) - 1)),
            data((const uint8_t*) _data),
            query(),
            centroidDp(_centroidDp),
            ay(),
            ly(),
            queryAdditional(),
            y1(),
            dimension(_dimension),
            numVectors(_numVectors) {
        }

        // Popcount of (a AND b) over `planeBytes` bytes.
        //
        // NEON path: 16 bytes per iteration via vcntq_u8, widened to a u32 accumulator.
        // Scalar path: 8-byte-word popcount via __builtin_popcountll, using std::memcpy for
        // the loads so it works when planeBytes is not a multiple of 8 (B>1 doc planes may
        // start at unaligned offsets within the on-disk vector).
        static inline uint64_t popcountAndPlane(const uint8_t* a, const uint8_t* b, const uint64_t planeBytes) {
#ifdef __ARM_NEON
            uint32x4_t acc = vdupq_n_u32(0);
            uint64_t i = 0;
            for ( ; i + 16 <= planeBytes ; i += 16) {
                uint8x16_t va = vld1q_u8(a + i);
                uint8x16_t vb = vld1q_u8(b + i);
                uint8x16_t cnt = vcntq_u8(vandq_u8(va, vb));
                acc = vaddq_u32(acc, vpaddlq_u16(vpaddlq_u8(cnt)));
            }
            uint64_t pc = vaddvq_u32(acc);
            for ( ; i < planeBytes ; ++i) {
                pc += __builtin_popcount((a[i] & b[i]) & 0xFF);
            }
            return pc;
#else
            const uint64_t words = planeBytes >> 3;
            uint64_t pc = 0;
            for (uint64_t w = 0; w < words; ++w) {
                uint64_t wa, wb;
                std::memcpy(&wa, a + w * 8, sizeof(uint64_t));
                std::memcpy(&wb, b + w * 8, sizeof(uint64_t));
                pc += __builtin_popcountll(wa & wb);
            }
            for (uint64_t r = words * 8; r < planeBytes; ++r) {
                pc += __builtin_popcount((a[r] & b[r]) & 0xFF);
            }
            return pc;
#endif
        }

        // Symmetric B=2 dot product used by the HNSW build path (both operands are stored
        // 2-plane docs in `transposeDibit` layout). The 2x2 loop expands to
        //   dp = pc(a0,b0) + 2·pc(a0,b1) + 2·pc(a1,b0) + 4·pc(a1,b1)
        //
        // On aarch64 the four traversals are fused into a single sweep: each 16-byte
        // chunk loads a0/a1/b0/b1 once (4 loads instead of 8) and issues 4 vandq+vcntq
        // popcounts into 4 separate u32 accumulators. Combined with shift weights at
        // horizontal-sum time. On non-ARM this reduces to four scalar popcountAndPlane
        // calls, so the fused form is a NEON-only optimization.
        static inline uint64_t symmetricDibitDp(const uint8_t* a, const uint8_t* b, const uint64_t planeBytes) {
#ifdef __ARM_NEON
            const uint8_t* a0 = a;
            const uint8_t* a1 = a + planeBytes;
            const uint8_t* b0 = b;
            const uint8_t* b1 = b + planeBytes;

            uint32x4_t acc00 = vdupq_n_u32(0);
            uint32x4_t acc01 = vdupq_n_u32(0);
            uint32x4_t acc10 = vdupq_n_u32(0);
            uint32x4_t acc11 = vdupq_n_u32(0);

            uint64_t i = 0;
            for ( ; i + 16 <= planeBytes ; i += 16) {
                uint8x16_t va0 = vld1q_u8(a0 + i);
                uint8x16_t va1 = vld1q_u8(a1 + i);
                uint8x16_t vb0 = vld1q_u8(b0 + i);
                uint8x16_t vb1 = vld1q_u8(b1 + i);

                acc00 = vaddq_u32(acc00, vpaddlq_u16(vpaddlq_u8(vcntq_u8(vandq_u8(va0, vb0)))));
                acc01 = vaddq_u32(acc01, vpaddlq_u16(vpaddlq_u8(vcntq_u8(vandq_u8(va0, vb1)))));
                acc10 = vaddq_u32(acc10, vpaddlq_u16(vpaddlq_u8(vcntq_u8(vandq_u8(va1, vb0)))));
                acc11 = vaddq_u32(acc11, vpaddlq_u16(vpaddlq_u8(vcntq_u8(vandq_u8(va1, vb1)))));
            }

            uint64_t pc00 = vaddvq_u32(acc00);
            uint64_t pc01 = vaddvq_u32(acc01);
            uint64_t pc10 = vaddvq_u32(acc10);
            uint64_t pc11 = vaddvq_u32(acc11);

            for ( ; i < planeBytes ; ++i) {
                uint8_t x0 = a0[i], x1 = a1[i], y0 = b0[i], y1 = b1[i];
                pc00 += __builtin_popcount((x0 & y0) & 0xFF);
                pc01 += __builtin_popcount((x0 & y1) & 0xFF);
                pc10 += __builtin_popcount((x1 & y0) & 0xFF);
                pc11 += __builtin_popcount((x1 & y1) & 0xFF);
            }

            return pc00 + ((pc01 + pc10) << 1) + (pc11 << 2);
#else
            const uint64_t pc00 = popcountAndPlane(a,              b,              planeBytes);
            const uint64_t pc01 = popcountAndPlane(a,              b + planeBytes, planeBytes);
            const uint64_t pc10 = popcountAndPlane(a + planeBytes, b,              planeBytes);
            const uint64_t pc11 = popcountAndPlane(a + planeBytes, b + planeBytes, planeBytes);
            return pc00 + ((pc01 + pc10) << 1) + (pc11 << 2);
#endif
        }

        // Byte-wise dot product over Lucene's PACKED_NIBBLE doc layout.
        // packed[i] high nibble = element i, low nibble = element (packedBytes + i).
        // Mirrors Lucene's VectorUtil.int4DotProductBothPacked. For each byte we extract two
        // 4-bit values per side and accumulate two products.
        //
        // NEON path: per 16 packed bytes (32 elements) we issue 4 vmull_u8 widening
        // multiplies (aHi*bHi + aLo*bLo, low & high halves of the 16-lane register) and
        // accumulate into a u32 register via vpadalq_u16. Nibble values ∈ [0, 15], so per-
        // product max = 225 (u8-safe pre-widen); 32 accumulated per 16-byte chunk = 7200
        // per chunk, well within u32 headroom for realistic dims.
        //
        // Scalar path: two-per-byte scalar MUL loop. Compilers with -O2 auto-vectorize
        // this into PMULLW on x86 and NEON MUL on ARM builds without NEON intrinsics
        // enabled, but relying on that is fragile — the explicit NEON path guarantees it.
        static inline uint64_t bothPackedNibbleDp(const uint8_t* a, const uint8_t* b, const uint64_t packedBytes) {
#ifdef __ARM_NEON
            const uint8x16_t lowMask = vdupq_n_u8(0x0F);
            uint32x4_t acc = vdupq_n_u32(0);
            uint64_t i = 0;
            for ( ; i + 16 <= packedBytes ; i += 16) {
                uint8x16_t va = vld1q_u8(a + i);
                uint8x16_t vb = vld1q_u8(b + i);
                uint8x16_t aHi = vshrq_n_u8(va, 4);
                uint8x16_t aLo = vandq_u8(va, lowMask);
                uint8x16_t bHi = vshrq_n_u8(vb, 4);
                uint8x16_t bLo = vandq_u8(vb, lowMask);

                uint16x8_t hiProd0 = vmull_u8(vget_low_u8(aHi),  vget_low_u8(bHi));
                uint16x8_t hiProd1 = vmull_u8(vget_high_u8(aHi), vget_high_u8(bHi));
                uint16x8_t loProd0 = vmull_u8(vget_low_u8(aLo),  vget_low_u8(bLo));
                uint16x8_t loProd1 = vmull_u8(vget_high_u8(aLo), vget_high_u8(bLo));

                acc = vpadalq_u16(acc, hiProd0);
                acc = vpadalq_u16(acc, hiProd1);
                acc = vpadalq_u16(acc, loProd0);
                acc = vpadalq_u16(acc, loProd1);
            }
            uint64_t total = vaddvq_u32(acc);
            for ( ; i < packedBytes ; ++i) {
                const uint32_t aLo = a[i] & 0x0Fu;
                const uint32_t aHi = (a[i] >> 4) & 0x0Fu;
                const uint32_t bLo = b[i] & 0x0Fu;
                const uint32_t bHi = (b[i] >> 4) & 0x0Fu;
                total += aLo * bLo + aHi * bHi;
            }
            return total;
#else
            uint64_t total = 0;
            for (uint64_t i = 0; i < packedBytes; ++i) {
                const uint32_t aLo = a[i] & 0x0Fu;
                const uint32_t aHi = (a[i] >> 4) & 0x0Fu;
                const uint32_t bLo = b[i] & 0x0Fu;
                const uint32_t bHi = (b[i] >> 4) & 0x0Fu;
                total += aLo * bLo + aHi * bHi;
            }
            return total;
#endif
        }

        // Multi-bit dot product between two quantized codes. The kernel shape depends on docBits:
        //   B=1: single popcount(a AND b)                         — bit-plane (1 plane)
        //   B=2: 2x2 popcount-AND-shift double sum across planes  — bit-plane (2 planes)
        //   B=4: byte-wise nibble multiply-accumulate             — Lucene's PACKED_NIBBLE layout
        // Performance trade-off: for B=4 the bit-plane formulation needs 16 popcount calls per
        // distance, while the byte-wise multiply matches Lucene's int4DotProductBothPacked and
        // avoids both the popcount-shift dance and the ingest-time repack step. The math is the
        // same in either case (Σ A_n B_n with A_n, B_n in [0, 2^B - 1]); only the byte layout
        // differs. See claude_files/sq-mos-nibbles-explained.md for the full derivation.
        uint64_t multiBitDp(const uint8_t* a, const uint8_t* b) const {
            if (docBits == 1) {
                return popcountAndPlane(a, b, planeBytes);
            }
            if (docBits == 2) {
                // Fused 2x2 kernel: loads each plane chunk once instead of four times.
                return symmetricDibitDp(a, b, planeBytes);
            }
            if (docBits == 4) {
                return bothPackedNibbleDp(a, b, quantizedVectorBytes);
            }
            // Unsupported width — should have been rejected in InitFaissSQIndex.
            throw std::runtime_error("Unsupported docBits in multiBitDp: " + std::to_string(docBits));
        }

        void set_query(const float* x) final {
            // The query pointer comes from FaissSQFlat::quantizedVectorsAndCorrectionFactors
            // which uses NBytesAlignedAllocator<uint8_t, 8> (8-byte aligned base) with a
            // stride of oneElementByteSize that is always a multiple of 8 (quantizedVectorBytes
            // is a multiple of 8 by formula, plus 16 bytes of correction factors).
            // Therefore x is guaranteed 8-byte aligned when IsBytesMultipleOf8 is true.
            query = reinterpret_cast<const uint8_t*>(x);
            setCorrectionFactors(query, ay, ly, queryAdditional, y1);
        }

        void setCorrectionFactors(const void* target, float& lowerInterval, float& intervalLength, float& additionalCorrection, float& quantizedComponentSum) {
            const uint8_t* ptr = static_cast<const uint8_t*>(target) + quantizedVectorBytes;
            if constexpr (IsBytesMultipleOf8) {
                readCorrectionFactorsAligned(ptr, lowerInterval, intervalLength, additionalCorrection, quantizedComponentSum);
            } else {
                readCorrectionFactorsSafe(ptr, lowerInterval, intervalLength, additionalCorrection, quantizedComponentSum);
            }
            // Scale (upper - lower) by 1/(2^docBits - 1) so the reconstruction delta matches the
            // quantization level range [0, 2^docBits - 1]. For docBits == 1 this is a no-op.
            intervalLength *= dataScale;
        }

        float scoringSecondPart(const void* target, const float dp) {
            // Get correction factors
            float ax;
            float lx;
            float additional;
            float x1;
            setCorrectionFactors(target, ax, lx, additional, x1);

            // Scoring
            float score = ax * ay * dimension
                   + ay * lx * x1
                   + ax * ly * y1
                   + lx * ly * dp;

            if constexpr (IsMaxIP) {
                score += queryAdditional + additional - centroidDp;
                // Negate: Faiss HNSW always minimizes distance (CMax comparator).
                // For IP, higher score = more similar, so we negate so that
                // minimizing -score = maximizing score.
                return -score;
            } else {
                // L2: squared distance = ||q||² + ||x||² - 2*dot(q,x)
                // additionalCorrection values carry the squared norms of centroid-centered vectors.
                return queryAdditional + additional - 2.0f * score;
            }
        }

        /// compute distance of vector i to current query
        float operator()(faiss::idx_t i) final {
            const uint8_t* target = data + i * oneElementByteSize;
            const uint64_t dp = multiBitDp(query, target);
            return scoringSecondPart(target, static_cast<float>(dp));
        }

        /// compute distances of current query to 4 stored vectors.
        void distances_batch_4(
                const faiss::idx_t idx0,
                const faiss::idx_t idx1,
                const faiss::idx_t idx2,
                const faiss::idx_t idx3,
                float& dis0,
                float& dis1,
                float& dis2,
                float& dis3) final {
            const uint8_t* target1 = data + idx0 * oneElementByteSize;
            const uint8_t* target2 = data + idx1 * oneElementByteSize;
            const uint8_t* target3 = data + idx2 * oneElementByteSize;
            const uint8_t* target4 = data + idx3 * oneElementByteSize;

            dis0 = scoringSecondPart(target1, static_cast<float>(multiBitDp(query, target1)));
            dis1 = scoringSecondPart(target2, static_cast<float>(multiBitDp(query, target2)));
            dis2 = scoringSecondPart(target3, static_cast<float>(multiBitDp(query, target3)));
            dis3 = scoringSecondPart(target4, static_cast<float>(multiBitDp(query, target4)));
        }

        /// compute distance between two stored vectors
        float symmetric_dis(faiss::idx_t i, faiss::idx_t j) {
            const uint8_t* target1 = data + i * oneElementByteSize;
            const uint8_t* target2 = data + j * oneElementByteSize;

            const uint64_t dp = multiBitDp(target1, target2);

            // Get correction factors
            float ax, lx, additional, x1;
            setCorrectionFactors(target1, ax, lx, additional, x1);

            float az, lz, additionalz, z1;
            setCorrectionFactors(target2, az, lz, additionalz, z1);

            // Scoring
            float score = ax * az * dimension
                   + az * lx * x1
                   + ax * lz * z1
                   + lx * lz * static_cast<float>(dp);

            if constexpr (IsMaxIP) {
                // Negate: Faiss HNSW always minimizes distance (CMax comparator).
                // For IP, higher score = more similar, so we negate so that
                // minimizing -score = maximizing score.
                score += additional + additionalz - centroidDp;
                return -score;
            } else {
                return additional + additionalz - 2 * score;
            }
        }
    };

    struct FaissSQFlat final : faiss::IndexBinary {
        int64_t numVectors;
        int32_t quantizedVectorBytes;
        float centroidDp;
        int32_t oneElementSize;
        // For safely casting uint8_t* to float*, we should enforce 8-byte alignment for the vector.
        std::vector<uint8_t, knn_jni::NBytesAlignedAllocator<uint8_t, 8>> quantizedVectorsAndCorrectionFactors;
        int32_t dimension;
        // Document bit width (1, 2, or 4). quantizedVectorBytes == docBits * binaryCodeBytes.
        int32_t docBits;

        FaissSQFlat(int64_t _numVectors, int32_t _quantizedVectorBytes, float _centroidDp, int32_t _dimension, faiss::MetricType _metric, int32_t _docBits)
          : faiss::IndexBinary(_dimension, _metric, true),
            numVectors(_numVectors),
            quantizedVectorBytes(_quantizedVectorBytes),
            centroidDp(_centroidDp),
            oneElementSize(_quantizedVectorBytes + 3 * sizeof(float) + sizeof(int32_t)),
            // Pre allocate vector storage space
            quantizedVectorsAndCorrectionFactors(_numVectors * oneElementSize),
            dimension(_dimension),
            docBits(_docBits) {

            // Just changing the size, not shrinking, thus allocated memory capacity remains the same.
            // This is to avoid reallocations when adding elements later on since we know the exact required memory space upfront.
            quantizedVectorsAndCorrectionFactors.resize(0);
            // Rewriting code_size to the full element size so that hnsw_add_vertices
            // strides correctly through the packed buffer when computing:
            //   x + (pt_id - n0) * index_hnsw.code_size
            // Memory layout per element:
            // [Quantized Vector | lowerInterval (float) | upperInterval (float) | additionalCorrection (float) | quantizedComponentSum (int)]
            code_size = oneElementSize;
        }

        faiss::DistanceComputer* get_distance_computer() const {
            // When quantizedVectorBytes is a multiple of 8, oneElementSize is also a
            // multiple of 8 (quantizedVectorBytes + 16 bytes of correction factors),
            // so element starts are 8-byte aligned and we can use the fast
            // reinterpret_cast<uint64_t*> path. Otherwise we fall back to memcpy
            // with a byte remainder loop for the trailing bytes.
            const bool aligned = (oneElementSize % 8) == 0;
            if (metric_type == faiss::MetricType::METRIC_L2) {
                if (aligned) {
                    return new FaissSQDistanceComputer<false, true>(oneElementSize, quantizedVectorsAndCorrectionFactors.data(), centroidDp, dimension, numVectors, docBits);
                } else {
                    return new FaissSQDistanceComputer<false, false>(oneElementSize, quantizedVectorsAndCorrectionFactors.data(), centroidDp, dimension, numVectors, docBits);
                }
            } else if (metric_type == faiss::MetricType::METRIC_INNER_PRODUCT) {
                if (aligned) {
                    return new FaissSQDistanceComputer<true, true>(oneElementSize, quantizedVectorsAndCorrectionFactors.data(), centroidDp, dimension, numVectors, docBits);
                } else {
                    return new FaissSQDistanceComputer<true, false>(oneElementSize, quantizedVectorsAndCorrectionFactors.data(), centroidDp, dimension, numVectors, docBits);
                }
            }

            throw std::runtime_error("Unsupported metric type - " + std::to_string(metric_type));
        }

        void search(faiss::idx_t n,
                    const uint8_t* x,
                    faiss::idx_t k,
                    int32_t* distances,
                    faiss::idx_t* labels,
                    const faiss::SearchParameters* params = nullptr) const final {
            throw std::runtime_error("FaissSQFlat does not support search");
        }

        void reset() final {
            throw std::runtime_error("FaissSQFlat does not support reset");
        }

        void merge_from(faiss::IndexBinary& otherIndex, faiss::idx_t add_id = 0) final {
            throw std::runtime_error("FaissSQFlat does not support merge_from");
        };

        void add(faiss::idx_t n, const uint8_t* x) final {
            // We only increase ntotal here, as it does not actually own the binary quantized vectors.
            // They are in a separate files.
            ntotal += n;
        }
    };

}

#endif //KNNPLUGIN_JNI_FAISS_SQ_FLAT_H
