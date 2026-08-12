/**
 * Low-level vector operations on Float32Array segments.
 *
 * These are the fundamental primitives that could later be swapped
 * for SIMD implementations. All operations work on raw arrays with
 * explicit offset/length to avoid slice allocations.
 */

/** Dot product of two vectors. */
export function vecDot(
  a: Float32Array, aOffset: number,
  b: Float32Array, bOffset: number,
  length: number,
): number {
  let sum = 0;
  for (let i = 0; i < length; i++) {
    sum += a[aOffset + i] * b[bOffset + i];
  }
  return sum;
}

/** Element-wise addition: out = a + b */
export function vecAdd(
  a: Float32Array, aOffset: number,
  b: Float32Array, bOffset: number,
  out: Float32Array, outOffset: number,
  length: number,
): void {
  for (let i = 0; i < length; i++) {
    out[outOffset + i] = a[aOffset + i] + b[bOffset + i];
  }
}

/** Element-wise addition in-place: a += b */
export function vecAddInPlace(
  a: Float32Array, aOffset: number,
  b: Float32Array, bOffset: number,
  length: number,
): void {
  for (let i = 0; i < length; i++) {
    a[aOffset + i] += b[bOffset + i];
  }
}

/** Element-wise multiplication: out = a * b */
export function vecMul(
  a: Float32Array, aOffset: number,
  b: Float32Array, bOffset: number,
  out: Float32Array, outOffset: number,
  length: number,
): void {
  for (let i = 0; i < length; i++) {
    out[outOffset + i] = a[aOffset + i] * b[bOffset + i];
  }
}

/** Element-wise multiplication in-place: a *= b */
export function vecMulInPlace(
  a: Float32Array, aOffset: number,
  b: Float32Array, bOffset: number,
  length: number,
): void {
  for (let i = 0; i < length; i++) {
    a[aOffset + i] *= b[bOffset + i];
  }
}

/** Scale a vector: out = a * scalar */
export function vecScale(
  a: Float32Array, aOffset: number,
  scalar: number,
  out: Float32Array, outOffset: number,
  length: number,
): void {
  for (let i = 0; i < length; i++) {
    out[outOffset + i] = a[aOffset + i] * scalar;
  }
}

/** Scale in-place: a *= scalar */
export function vecScaleInPlace(
  a: Float32Array, aOffset: number,
  scalar: number,
  length: number,
): void {
  for (let i = 0; i < length; i++) {
    a[aOffset + i] *= scalar;
  }
}

/** Copy vector: out = a */
export function vecCopy(
  src: Float32Array, srcOffset: number,
  dst: Float32Array, dstOffset: number,
  length: number,
): void {
  for (let i = 0; i < length; i++) {
    dst[dstOffset + i] = src[srcOffset + i];
  }
}

/** Sum of all elements. */
export function vecSum(
  a: Float32Array, offset: number,
  length: number,
): number {
  let sum = 0;
  for (let i = 0; i < length; i++) {
    sum += a[offset + i];
  }
  return sum;
}

/** Maximum element value. */
export function vecMax(
  a: Float32Array, offset: number,
  length: number,
): number {
  let max = a[offset];
  for (let i = 1; i < length; i++) {
    const val = a[offset + i];
    if (val > max) max = val;
  }
  return max;
}

/** Index of maximum element (argmax). */
export function vecArgmax(
  a: Float32Array, offset: number,
  length: number,
): number {
  let maxIdx = 0;
  let maxVal = a[offset];
  for (let i = 1; i < length; i++) {
    const val = a[offset + i];
    if (val > maxVal) {
      maxVal = val;
      maxIdx = i;
    }
  }
  return maxIdx;
}

/** Fill a vector with a constant value. */
export function vecFill(
  a: Float32Array, offset: number,
  length: number,
  value: number,
): void {
  for (let i = 0; i < length; i++) {
    a[offset + i] = value;
  }
}
