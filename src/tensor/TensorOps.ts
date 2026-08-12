/**
 * Higher-level tensor operations built on VectorOps.
 */
import { Tensor } from "./Tensor.js";
import {
  vecAdd, vecMul, vecScale, vecDot, vecSum, vecMax, vecArgmax,
} from "../math/VectorOps.js";

/** Element-wise addition: result = a + b */
export function add(a: Tensor, b: Tensor, out?: Tensor): Tensor {
  if (a.length !== b.length) throw new Error("Tensor size mismatch for add");
  const result = out ?? Tensor.zeros([...a.shape]);
  vecAdd(a.data, 0, b.data, 0, result.data, 0, a.length);
  return result;
}

/** Element-wise multiplication: result = a * b */
export function mul(a: Tensor, b: Tensor, out?: Tensor): Tensor {
  if (a.length !== b.length) throw new Error("Tensor size mismatch for mul");
  const result = out ?? Tensor.zeros([...a.shape]);
  vecMul(a.data, 0, b.data, 0, result.data, 0, a.length);
  return result;
}

/** Scale: result = a * scalar */
export function scale(a: Tensor, scalar: number, out?: Tensor): Tensor {
  const result = out ?? Tensor.zeros([...a.shape]);
  vecScale(a.data, 0, scalar, result.data, 0, a.length);
  return result;
}

/** Dot product of two 1D tensors. */
export function dot(a: Tensor, b: Tensor): number {
  if (a.length !== b.length) throw new Error("Tensor size mismatch for dot");
  return vecDot(a.data, 0, b.data, 0, a.length);
}

/** Sum of all elements. */
export function sum(a: Tensor): number {
  return vecSum(a.data, 0, a.length);
}

/** Mean of all elements. */
export function mean(a: Tensor): number {
  return vecSum(a.data, 0, a.length) / a.length;
}

/** Max element value. */
export function max(a: Tensor): number {
  return vecMax(a.data, 0, a.length);
}

/** Argmax — index of maximum element. */
export function argmax(a: Tensor): number {
  return vecArgmax(a.data, 0, a.length);
}
