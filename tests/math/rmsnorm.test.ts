import test from "node:test";
import assert from "node:assert/strict";
import { rmsNorm } from "../../src/math/RMSNorm.js";

function assertNear(actual: number, expected: number, eps = 1e-5): void {
  assert.ok(
    Math.abs(actual - expected) < eps,
    `Expected ${expected}, got ${actual} (diff: ${Math.abs(actual - expected)})`
  );
}

test("RMSNorm - basic normalization", () => {
  const x = new Float32Array([1, 2, 3, 4]);
  const weight = new Float32Array([1, 1, 1, 1]);
  const out = new Float32Array(4);

  // sumSq = 1+4+9+16 = 30
  // meanSq = 30 / 4 = 7.5
  // rms = 1 / sqrt(7.5 + 1e-5) ≈ 1 / 2.7386127 ≈ 0.365148
  // out[0] = 1 * rms ≈ 0.365148
  // out[1] = 2 * rms ≈ 0.730296
  // out[2] = 3 * rms ≈ 1.095445
  // out[3] = 4 * rms ≈ 1.460593

  rmsNorm(x, 0, weight, 0, out, 0, 4, 1e-5);

  assertNear(out[0], 0.365148);
  assertNear(out[1], 0.730296);
  assertNear(out[2], 1.095445);
  assertNear(out[3], 1.460593);
});
