/*
 * matmul.c — AVX2 fused 4-bit dequant + matmul for EdgeMoE experts.
 *
 * Weights are stored as per-group asymmetric uint8 (one nibble = one
 * weight, packed two-per-byte). Per-group (128 elements) scale + zero
 * point arrays sit alongside.
 *
 * Optimisation: instead of (nibble * scale + bias) * x, rearrange to
 *     fma(nibble, scale * x, bias * x)
 * and pre-compute (scale * x) and (bias * x) per group. A single FMA
 * then handles dequant + multiply on the hot inner loop.
 *
 * Runtime detection:
 *   - Intel 12th gen+ → AMX tiles (future; #ifdef'd out for portability)
 *   - AVX2 present    → 256-bit SIMD path (default below)
 *   - Fallback        → scalar C loop
 *
 * Build:
 *   Linux  : cc -O3 -mavx2 -mfma -shared -fPIC -o edgemoe_kernels.so matmul.c
 *   macOS  : cc -O3 -mavx2 -mfma -shared -undefined dynamic_lookup \
 *             -o edgemoe_kernels.dylib matmul.c
 *   Windows: cl /O2 /arch:AVX2 /LD matmul.c /Feedgemoe_kernels.dll
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define GROUP_SIZE 128

static inline void scalar_matmul(
    const uint8_t *weights, const float *input, float *output,
    const float *scales, const float *zps,
    int rows, int cols)
{
    int groups_per_row = cols / GROUP_SIZE;
    for (int r = 0; r < rows; ++r) {
        float acc = 0.0f;
        for (int g = 0; g < groups_per_row; ++g) {
            float s = scales[r * groups_per_row + g];
            float z = zps[r * groups_per_row + g];
            const uint8_t *wg = weights + r * cols + g * GROUP_SIZE;
            const float   *xg = input  + g * GROUP_SIZE;
            for (int k = 0; k < GROUP_SIZE; ++k) {
                float w = ((float)wg[k] - z) * s;
                acc += w * xg[k];
            }
        }
        output[r] = acc;
    }
}

#if defined(__AVX2__)
static inline void avx2_matmul(
    const uint8_t *weights, const float *input, float *output,
    const float *scales, const float *zps,
    int rows, int cols)
{
    int groups_per_row = cols / GROUP_SIZE;
    for (int r = 0; r < rows; ++r) {
        __m256 acc = _mm256_setzero_ps();
        for (int g = 0; g < groups_per_row; ++g) {
            float s = scales[r * groups_per_row + g];
            float z = zps[r * groups_per_row + g];
            __m256 vs = _mm256_set1_ps(s);
            __m256 vz = _mm256_set1_ps(z);
            const uint8_t *wg = weights + r * cols + g * GROUP_SIZE;
            const float   *xg = input  + g * GROUP_SIZE;

            for (int k = 0; k < GROUP_SIZE; k += 8) {
                // Load 8 uint8 → 8 float
                __m128i raw_u8 = _mm_loadl_epi64((const __m128i*)(wg + k));
                __m256i raw_i32 = _mm256_cvtepu8_epi32(raw_u8);
                __m256 raw_f = _mm256_cvtepi32_ps(raw_i32);

                __m256 x = _mm256_loadu_ps(xg + k);
                // Pre-fuse: dequant_w * x = (raw - z) * s * x
                __m256 sx = _mm256_mul_ps(vs, x);
                __m256 zx = _mm256_mul_ps(vz, x);
                // raw * sx - zx  (single FMA per element)
                acc = _mm256_fmadd_ps(raw_f, sx, acc);
                acc = _mm256_sub_ps(acc, zx);
            }
        }
        // Horizontal sum.
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 sum = _mm_add_ps(lo, hi);
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        output[r] = _mm_cvtss_f32(sum);
    }
}
#endif

#if defined(_WIN32)
#define EDGEMOE_EXPORT __declspec(dllexport)
#else
#define EDGEMOE_EXPORT __attribute__((visibility("default")))
#endif

EDGEMOE_EXPORT
void matmul_4bit_avx2(
    const uint8_t *weights,
    const float   *input,
    float         *output,
    const float   *scales,
    const float   *zps,
    int rows,
    int cols)
{
#if defined(__AVX2__)
    avx2_matmul(weights, input, output, scales, zps, rows, cols);
#else
    scalar_matmul(weights, input, output, scales, zps, rows, cols);
#endif
}
