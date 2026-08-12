/**
 * Tensor abstraction.
 *
 * Lightweight wrapper around Float32Array with shape tracking.
 * For Phase 1, all tensors are stored as F32.
 */

export class Tensor {
  /** Shape of the tensor (e.g., [768] for a vector, [768, 768] for a matrix). */
  readonly shape: readonly number[];

  /** Underlying data. */
  readonly data: Float32Array;

  constructor(data: Float32Array, shape: number[]) {
    this.data = data;
    this.shape = Object.freeze([...shape]);
  }

  /** Total number of elements. */
  get length(): number {
    return this.data.length;
  }

  /** Number of dimensions. */
  get ndim(): number {
    return this.shape.length;
  }

  /** Create a zero-filled tensor of the given shape. */
  static zeros(shape: number[]): Tensor {
    let size = 1;
    for (const d of shape) size *= d;
    return new Tensor(new Float32Array(size), shape);
  }

  /** Create a tensor from existing data with the given shape. */
  static from(data: Float32Array, shape: number[]): Tensor {
    let expectedSize = 1;
    for (const d of shape) expectedSize *= d;
    if (data.length !== expectedSize) {
      throw new Error(
        `Data length ${data.length} does not match shape [${shape}] (expected ${expectedSize})`
      );
    }
    return new Tensor(data, shape);
  }

  /** Create a 1D tensor from an array of numbers. */
  static fromArray(values: number[]): Tensor {
    return new Tensor(new Float32Array(values), [values.length]);
  }

  /** Reshape without copying data (view). */
  reshape(newShape: number[]): Tensor {
    let newSize = 1;
    for (const d of newShape) newSize *= d;
    if (newSize !== this.data.length) {
      throw new Error(
        `Cannot reshape tensor of size ${this.data.length} to shape [${newShape}]`
      );
    }
    return new Tensor(this.data, newShape);
  }

  /** Get a string representation. */
  toString(): string {
    const shapeStr = `[${this.shape.join(", ")}]`;
    if (this.data.length <= 8) {
      return `Tensor(${shapeStr}, [${Array.from(this.data).map(v => v.toFixed(4)).join(", ")}])`;
    }
    const first = Array.from(this.data.slice(0, 4)).map(v => v.toFixed(4)).join(", ");
    const last = Array.from(this.data.slice(-2)).map(v => v.toFixed(4)).join(", ");
    return `Tensor(${shapeStr}, [${first}, ..., ${last}])`;
  }
}
