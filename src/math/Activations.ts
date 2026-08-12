/**
 * Activation functions.
 *
 * SiLU (Sigmoid Linear Unit) is used by SwiGLU in LLaMA-style models.
 */

/**
 * SiLU activation: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
 */
export function silu(x: number): number {
  return x / (1.0 + Math.exp(-x));
}

/**
 * Apply SiLU in-place to a vector segment.
 */
export function siluInPlace(
  a: Float32Array,
  offset: number,
  length: number,
): void {
  for (let i = 0; i < length; i++) {
    const x = a[offset + i];
    a[offset + i] = x / (1.0 + Math.exp(-x));
  }
}

/**
 * SiLU and element-wise multiply combined (for SwiGLU).
 * out = silu(gate) * up
 *
 * This is the core SwiGLU operation: avoids a separate pass.
 */
export function siluElementwiseMul(
  gate: Float32Array, gOffset: number,
  up: Float32Array, uOffset: number,
  out: Float32Array, oOffset: number,
  length: number,
): void {
  for (let i = 0; i < length; i++) {
    const g = gate[gOffset + i];
    const siluG = g / (1.0 + Math.exp(-g));
    out[oOffset + i] = siluG * up[uOffset + i];
  }
}
