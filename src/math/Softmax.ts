/**
 * Numerically stable softmax implementation.
 *
 * Strategy:
 *   max → x - max → exp → sum → normalize
 *
 * Never applies exp() to unnormalized values.
 */

/**
 * Softmax over a contiguous segment of an array.
 * Modifies the array in-place.
 *
 * @param x      The array containing the values
 * @param offset Start index of the segment
 * @param length Number of elements in the segment
 */
export function softmax(
  x: Float32Array,
  offset: number,
  length: number,
): void {
  // 1. Find max for numerical stability
  let max = x[offset];
  for (let i = 1; i < length; i++) {
    const val = x[offset + i];
    if (val > max) max = val;
  }

  // 2. exp(x - max) and accumulate sum
  let sum = 0;
  for (let i = 0; i < length; i++) {
    const expVal = Math.exp(x[offset + i] - max);
    x[offset + i] = expVal;
    sum += expVal;
  }

  // 3. Normalize
  const invSum = 1.0 / sum;
  for (let i = 0; i < length; i++) {
    x[offset + i] *= invSum;
  }
}

/**
 * Softmax into a separate output array (non-destructive).
 */
export function softmaxTo(
  input: Float32Array, inOffset: number,
  output: Float32Array, outOffset: number,
  length: number,
): void {
  // 1. Find max
  let max = input[inOffset];
  for (let i = 1; i < length; i++) {
    const val = input[inOffset + i];
    if (val > max) max = val;
  }

  // 2. exp(x - max) and sum
  let sum = 0;
  for (let i = 0; i < length; i++) {
    const expVal = Math.exp(input[inOffset + i] - max);
    output[outOffset + i] = expVal;
    sum += expVal;
  }

  // 3. Normalize
  const invSum = 1.0 / sum;
  for (let i = 0; i < length; i++) {
    output[outOffset + i] *= invSum;
  }
}
