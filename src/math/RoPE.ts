/**
 * Rotary Positional Embeddings (RoPE).
 *
 * Applied to Q and K vectors to encode positional information.
 * Pairs of dimensions are rotated by position-dependent angles.
 *
 * Parameters:
 *   - position: token position in the sequence
 *   - headDim: dimension of each attention head
 *   - theta: base frequency (default 10000, varies by model)
 *   - freqScale: frequency scaling factor for extended context
 */

/**
 * Apply RoPE to a vector in-place.
 *
 * The vector is expected to contain `nHeads` heads, each of `headDim` dimensions.
 * Rotation is applied to pairs of dimensions (2i, 2i+1).
 *
 * @param vec      The Q or K vector
 * @param offset   Offset into vec
 * @param position Token position
 * @param headDim  Dimension per head
 * @param nHeads   Number of heads
 * @param theta    Base frequency
 * @param freqScale Frequency scaling factor
 */
export function applyRoPE(
  vec: Float32Array,
  offset: number,
  position: number,
  headDim: number,
  nHeads: number,
  theta: number = 10000.0,
  freqScale: number = 1.0,
): void {
  const halfDim = headDim / 2;

  for (let h = 0; h < nHeads; h++) {
    const headOffset = offset + h * headDim;

    for (let i = 0; i < halfDim; i++) {
      // Compute frequency for this dimension pair
      const freq = 1.0 / Math.pow(theta, (2 * i) / headDim) * freqScale;
      const angle = position * freq;

      const cos = Math.cos(angle);
      const sin = Math.sin(angle);

      // Rotate adjacent pair (2i, 2i+1) matching GGML layout
      const idx0 = headOffset + 2 * i;
      const idx1 = headOffset + 2 * i + 1;

      const v0 = vec[idx0];
      const v1 = vec[idx1];

      vec[idx0] = v0 * cos - v1 * sin;
      vec[idx1] = v0 * sin + v1 * cos;
    }
  }
}

/**
 * Apply RoPE to Q and K vectors simultaneously.
 * More efficient than calling applyRoPE twice since frequencies are shared.
 */
export function applyRoPEQK(
  q: Float32Array, qOffset: number, nQHeads: number,
  k: Float32Array, kOffset: number, nKVHeads: number,
  position: number,
  headDim: number,
  theta: number = 10000.0,
  freqScale: number = 1.0,
): void {
  applyRoPE(q, qOffset, position, headDim, nQHeads, theta, freqScale);
  applyRoPE(k, kOffset, position, headDim, nKVHeads, theta, freqScale);
}
