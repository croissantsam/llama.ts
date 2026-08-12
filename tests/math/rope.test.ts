import test from "node:test";
import assert from "node:assert/strict";
import { applyRoPE } from "../../src/math/RoPE.js";

function assertNear(actual: number, expected: number, eps = 1e-4): void {
  assert.ok(
    Math.abs(actual - expected) < eps,
    `Expected ${expected}, got ${actual} (diff: ${Math.abs(actual - expected)})`
  );
}

test("RoPE - identity rotation at position 0", () => {
  const vec = new Float32Array([1, 2, 3, 4]); // headDim = 4, nHeads = 1
  applyRoPE(vec, 0, 0, 4, 1, 10000);

  // Position 0 angle is 0, cos(0)=1, sin(0)=0 -> vector should remain unchanged
  assertNear(vec[0], 1);
  assertNear(vec[1], 2);
  assertNear(vec[2], 3);
  assertNear(vec[3], 4);
});

test("RoPE - rotation at position 1", () => {
  const vec = new Float32Array([1, 0, 0, 0]); // headDim = 4
  // pair 0: idx0=0, idx1=1 (v0=1, v1=0)
  // angle = 1 * (10000^0) = 1 rad
  // v0' = 1*cos(1) - 0*sin(1) = cos(1) ≈ 0.54030
  // v1' = 1*sin(1) + 0*cos(1) = sin(1) ≈ 0.84147

  applyRoPE(vec, 0, 1, 4, 1, 10000);

  assertNear(vec[0], 0.54030);
  assertNear(vec[1], 0.84147);
});
