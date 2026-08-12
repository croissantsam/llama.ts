import test from "node:test";
import assert from "node:assert/strict";
import { silu, siluInPlace, siluElementwiseMul } from "../../src/math/Activations.js";

function assertNear(actual: number, expected: number, eps = 1e-5): void {
  assert.ok(
    Math.abs(actual - expected) < eps,
    `Expected ${expected}, got ${actual} (diff: ${Math.abs(actual - expected)})`
  );
}

test("Activations - silu calculation", () => {
  // silu(0) = 0 / 2 = 0
  assertNear(silu(0), 0);

  // silu(1) = 1 / (1 + exp(-1)) = 1 / (1 + 0.367879) ≈ 0.731058
  assertNear(silu(1), 0.731058);

  const arr = new Float32Array([0, 1]);
  siluInPlace(arr, 0, 2);
  assertNear(arr[0], 0);
  assertNear(arr[1], 0.731058);
});

test("Activations - siluElementwiseMul (fused SwiGLU)", () => {
  const gate = new Float32Array([1, 2]);
  const up = new Float32Array([3, 4]);
  const out = new Float32Array(2);

  // gate[0]=1 -> silu(1) ≈ 0.731058 * up[0](3) ≈ 2.193175
  // gate[1]=2 -> silu(2) = 2 / (1 + exp(-2)) ≈ 2 / 1.135335 ≈ 1.761594 * up[1](4) ≈ 7.046376

  siluElementwiseMul(gate, 0, up, 0, out, 0, 2);

  assertNear(out[0], 2.193175);
  assertNear(out[1], 7.046376);
});
