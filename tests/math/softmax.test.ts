import test from "node:test";
import assert from "node:assert/strict";
import { softmax, softmaxTo } from "../../src/math/Softmax.js";

function assertNear(actual: number, expected: number, eps = 1e-5): void {
  assert.ok(
    Math.abs(actual - expected) < eps,
    `Expected ${expected}, got ${actual} (diff: ${Math.abs(actual - expected)})`
  );
}

test("Softmax - basic normalization and stability", () => {
  const input = new Float32Array([1, 2, 3]);
  // exp(1)=2.71828, exp(2)=7.38905, exp(3)=20.08553
  // sum = 30.19286
  // p = [0.09003, 0.24473, 0.66524]

  const out = new Float32Array(3);
  softmaxTo(input, 0, out, 0, 3);

  assertNear(out[0], 0.09003057);
  assertNear(out[1], 0.24472847);
  assertNear(out[2], 0.66524096);

  // Sum must be 1.0
  const sum = out[0] + out[1] + out[2];
  assertNear(sum, 1.0);

  // Test in-place with large values (testing numerical stability)
  const largeInput = new Float32Array([1000, 1001, 1002]);
  softmax(largeInput, 0, 3);
  assertNear(largeInput[0], 0.09003057);
  assertNear(largeInput[1], 0.24472847);
  assertNear(largeInput[2], 0.66524096);
});
