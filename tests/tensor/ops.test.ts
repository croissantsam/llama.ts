import test from "node:test";
import assert from "node:assert/strict";
import { Tensor } from "../../src/tensor/Tensor.js";
import { add, mul, scale, dot, sum, mean, max, argmax } from "../../src/tensor/TensorOps.js";

function assertNear(actual: number, expected: number, eps = 1e-5): void {
  assert.ok(
    Math.abs(actual - expected) < eps,
    `Expected ${expected}, got ${actual} (diff: ${Math.abs(actual - expected)})`
  );
}

test("TensorOps - basic tensor ops", () => {
  const t1 = Tensor.fromArray([1, 2, 3]);
  const t2 = Tensor.fromArray([4, 5, 6]);

  const resAdd = add(t1, t2);
  assert.deepEqual(Array.from(resAdd.data), [5, 7, 9]);

  const resMul = mul(t1, t2);
  assert.deepEqual(Array.from(resMul.data), [4, 10, 18]);

  const resScale = scale(t1, 3);
  assert.deepEqual(Array.from(resScale.data), [3, 6, 9]);

  assertNear(dot(t1, t2), 32);
  assertNear(sum(t1), 6);
  assertNear(mean(t1), 2);
  assertNear(max(t1), 3);
  assert.equal(argmax(t1), 2);
});
