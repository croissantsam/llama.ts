/**
 * Matrix multiplication implementations.
 *
 * Multiple versions are kept for benchmarking comparison:
 * - V1: Naive triple loop
 * - V2: Cache-friendly ikj loop ordering
 * - V3: Blocked/tiled multiplication
 *
 * All signatures use flat Float32Array with explicit dimensions:
 *   A is (M x K), B is (K x N), C is (M x N)
 *
 * Row-major layout: A[i][j] = A[i * cols + j]
 */

/**
 * V1 — Naive matrix multiplication.
 * C = A × B where A is (M×K), B is (K×N), C is (M×N).
 */
export function matmulNaive(
  A: Float32Array,
  B: Float32Array,
  C: Float32Array,
  M: number,
  N: number,
  K: number,
): void {
  for (let i = 0; i < M; i++) {
    for (let j = 0; j < N; j++) {
      let sum = 0;
      for (let p = 0; p < K; p++) {
        sum += A[i * K + p] * B[p * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

/**
 * V2 — Cache-friendly ikj loop ordering.
 * Accesses B in row-major order for better cache utilization.
 */
export function matmulReordered(
  A: Float32Array,
  B: Float32Array,
  C: Float32Array,
  M: number,
  N: number,
  K: number,
): void {
  // Zero out C
  C.fill(0);

  for (let i = 0; i < M; i++) {
    for (let p = 0; p < K; p++) {
      const aVal = A[i * K + p];
      const bRow = p * N;
      const cRow = i * N;
      for (let j = 0; j < N; j++) {
        C[cRow + j] += aVal * B[bRow + j];
      }
    }
  }
}

/**
 * V3 — Tiled/blocked matrix multiplication.
 * Improves cache locality by processing tiles that fit in L1/L2 cache.
 */
export function matmulTiled(
  A: Float32Array,
  B: Float32Array,
  C: Float32Array,
  M: number,
  N: number,
  K: number,
  tileSize: number = 64,
): void {
  // Zero out C
  C.fill(0);

  for (let i0 = 0; i0 < M; i0 += tileSize) {
    const iEnd = Math.min(i0 + tileSize, M);
    for (let p0 = 0; p0 < K; p0 += tileSize) {
      const pEnd = Math.min(p0 + tileSize, K);
      for (let j0 = 0; j0 < N; j0 += tileSize) {
        const jEnd = Math.min(j0 + tileSize, N);

        // Process tile
        for (let i = i0; i < iEnd; i++) {
          for (let p = p0; p < pEnd; p++) {
            const aVal = A[i * K + p];
            const bRow = p * N;
            const cRow = i * N;
            for (let j = j0; j < jEnd; j++) {
              C[cRow + j] += aVal * B[bRow + j];
            }
          }
        }
      }
    }
  }
}

/**
 * Matrix-vector multiplication: out = A × x
 * A is (M×K), x is (K), out is (M).
 *
 * This is the most common operation during inference (one token at a time).
 */
export function matvecmul(
  A: Float32Array, aOffset: number,
  x: Float32Array, xOffset: number,
  out: Float32Array, outOffset: number,
  M: number,
  K: number,
): void {
  for (let i = 0; i < M; i++) {
    let sum = 0;
    const rowStart = aOffset + i * K;
    for (let j = 0; j < K; j++) {
      sum += A[rowStart + j] * x[xOffset + j];
    }
    out[outOffset + i] = sum;
  }
}

/**
 * Fast F16 → F32 conversion for scale factors.
 */
function f16ToF32(h: number): number {
  const sign = (h >> 15) & 0x1;
  const exponent = (h >> 10) & 0x1f;
  const mantissa = h & 0x3ff;
  if (exponent === 0) return mantissa === 0 ? (sign ? -0 : 0) : (sign ? -1 : 1) * Math.pow(2, -14) * (mantissa / 1024);
  if (exponent === 0x1f) return mantissa === 0 ? (sign ? -Infinity : Infinity) : NaN;
  const val = Math.pow(2, exponent - 15) * (1 + mantissa / 1024);
  return sign ? -val : val;
}

/**
 * V5 — Quantized Matrix-Vector Multiplication for Q8_0.
 * Multiplies Q8_0 quantized weight matrix A with FP32 vector x without full FP32 conversion.
 *
 * A layout: M rows, each row has (K / 32) blocks of 34 bytes (2-byte f16 scale + 32 int8 weights).
 */
export function matvecmulQ8_0(
  A: Uint8Array, aOffset: number,
  x: Float32Array, xOffset: number,
  out: Float32Array, outOffset: number,
  M: number,
  K: number,
): void {
  const numBlocksPerRow = K / 32;
  const bytesPerRow = numBlocksPerRow * 34;
  const view = new DataView(A.buffer, A.byteOffset, A.byteLength);
  const int8View = new Int8Array(A.buffer, A.byteOffset, A.byteLength);

  for (let i = 0; i < M; i++) {
    let totalSum = 0;
    const rowByteOffset = aOffset + i * bytesPerRow;

    for (let b = 0; b < numBlocksPerRow; b++) {
      const blockOffset = rowByteOffset + b * 34;
      const dRaw = view.getUint16(blockOffset, true);
      const scale = f16ToF32(dRaw);

      let blockSum = 0;
      const qsOffset = blockOffset + 2;
      const xStart = xOffset + b * 32;

      for (let j = 0; j < 32; j++) {
        blockSum += int8View[qsOffset + j] * x[xStart + j];
      }

      totalSum += scale * blockSum;
    }

    out[outOffset + i] = totalSum;
  }
}

/**
 * V5 — Quantized Matrix-Vector Multiplication for Q4_0.
 * Multiplies Q4_0 quantized weight matrix A with FP32 vector x without full FP32 conversion.
 *
 * A layout: M rows, each row has (K / 32) blocks of 18 bytes (2-byte f16 scale + 16 bytes = 32 nibbles).
 */
export function matvecmulQ4_0(
  A: Uint8Array, aOffset: number,
  x: Float32Array, xOffset: number,
  out: Float32Array, outOffset: number,
  M: number,
  K: number,
): void {
  const numBlocksPerRow = K / 32;
  const bytesPerRow = numBlocksPerRow * 18;
  const view = new DataView(A.buffer, A.byteOffset, A.byteLength);

  for (let i = 0; i < M; i++) {
    let totalSum = 0;
    const rowByteOffset = aOffset + i * bytesPerRow;

    for (let b = 0; b < numBlocksPerRow; b++) {
      const blockOffset = rowByteOffset + b * 18;
      const dRaw = view.getUint16(blockOffset, true);
      const scale = f16ToF32(dRaw);

      let blockSum = 0;
      const qsOffset = blockOffset + 2;
      const xStart = xOffset + b * 32;

      for (let j = 0; j < 16; j++) {
        const byte = A[qsOffset + j];
        const q0 = (byte & 0x0f) - 8;
        const q1 = (byte >> 4) - 8;
        blockSum += q0 * x[xStart + j] + q1 * x[xStart + j + 16];
      }

      totalSum += scale * blockSum;
    }

    out[outOffset + i] = totalSum;
  }
}

/**
 * Default matmul — uses V2 (reordered) as the best general-purpose version.
 */
export const matmul = matmulReordered;
