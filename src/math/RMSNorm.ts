/**
 * RMS (Root Mean Square) Layer Normalization.
 *
 * Used by LLaMA-style architectures instead of LayerNorm.
 * More efficient: only computes RMS (no mean subtraction).
 *
 * Formula: out[i] = (x[i] / sqrt(mean(x²) + eps)) * weight[i]
 */

/**
 * RMSNorm with pre-allocated output buffer.
 *
 * @param x       Input vector
 * @param xOffset Offset into x
 * @param weight  Learned scale parameters
 * @param wOffset Offset into weight
 * @param out     Output buffer
 * @param oOffset Offset into out
 * @param length  Vector length
 * @param eps     Epsilon for numerical stability (typically 1e-5 or 1e-6)
 */
export function rmsNorm(
  x: Float32Array, xOffset: number,
  weight: Float32Array, wOffset: number,
  out: Float32Array, oOffset: number,
  length: number,
  eps: number,
): void {
  // 1. Compute mean of squares
  let sumSq = 0;
  for (let i = 0; i < length; i++) {
    const val = x[xOffset + i];
    sumSq += val * val;
  }
  const rms = 1.0 / Math.sqrt(sumSq / length + eps);

  // 2. Normalize and scale
  for (let i = 0; i < length; i++) {
    out[oOffset + i] = x[xOffset + i] * rms * weight[wOffset + i];
  }
}
