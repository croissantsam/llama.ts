import test from "node:test";
import assert from "node:assert/strict";
import {
  vecDot,
  vecAdd,
  vecAddInPlace,
  vecMul,
  vecMulInPlace,
  vecScale,
  vecCopy,
  vecSum,
  vecMax,
  vecArgmax,
} from "../../src/math/VectorOps.js";

function assertNear(actual: number, expected: number, eps = 1e-5): void {
  assert.ok(
    Math.abs(actual - expected) < eps,
    `Expected ${expected}, got ${actual} (diff: ${Math.abs(actual - expected)})`
  );
}

test("VectorOps - vecDot", () => {
  const a = new Float32Array([1, 2, 3, 4]);
  const b = new Float32Array([0.5, 2, 1.5, 1]);
  const res = vecDot(a, 0, b, 0, 4);
  // 1*0.5 + 2*2 + 3*1.5 + 4*1 = 0.5 + 4 + 4.5 + 4 = 13
  assertNear(res, 13);
});

test("VectorOps - vecAdd & vecAddInPlace", () => {
  const a = new Float32Array([1, 2, 3]);
  const b = new Float32Array([10, 20, 30]);
  const out = new Float32Array(3);

  vecAdd(a, 0, b, 0, out, 0, 3);
  assertNear(out[0], 11);
  assertNear(out[1], 22);
  assertNear(out[2], 33);

  vecAddInPlace(a, 0, b, 0, 3);
  assertNear(a[0], 11);
  assertNear(a[1], 22);
  assertNear(a[2], 33);
});

test("VectorOps - vecMul & vecMulInPlace", () => {
  const a = new Float32Array([2, 3, 4]);
  const b = new Float32Array([5, 6, 7]);
  const out = new Float32Array(3);

  vecMul(a, 0, b, 0, out, 0, 3);
  assertNear(out[0], 10);
  assertNear(out[1], 18);
  assertNear(out[2], 28);

  vecMulInPlace(a, 0, b, 0, 3);
  assertNear(a[0], 10);
  assertNear(a[1], 18);
  assertNear(a[2], 28);
});

test("VectorOps - vecScale & vecCopy & vecSum & vecMax & vecArgmax", () => {
  const a = new Float32Array([1, 5, 3, 2]);
  const out = new Float32Array(4);

  vecScale(a, 0, 2, out, 0, 4);
  assertNear(out[0], 2);
  assertNear(out[1], 10);

  vecCopy(a, 0, out, 0, 4);
  assertNear(out[1], 5);

  assertNear(vecSum(a, 0, 4), 11);
  assertNear(vecMax(a, 0, 4), 5);
  assert.equal(vecArgmax(a, 0, 4), 1);
});
