import test from "node:test";
import assert from "node:assert/strict";
import {
  matmulNaive,
  matmulReordered,
  matmulTiled,
  matvecmul,
  matvecmulQ8_0,
  matvecmulQ4_0,
} from "../../src/math/MatMul.js";

function assertNear(actual: number, expected: number, eps = 1e-5): void {
  assert.ok(
    Math.abs(actual - expected) < eps,
    `Expected ${expected}, got ${actual} (diff: ${Math.abs(actual - expected)})`
  );
}

test("MatMul - Naive, Reordered, Tiled correctness", () => {
  // A is 2x3
  // B is 3x2
  const A = new Float32Array([
    1, 2, 3,
    4, 5, 6
  ]);
  const B = new Float32Array([
    7, 8,
    9, 1,
    2, 3
  ]);

  // Expected C (2x2):
  // Row 0: [1*7 + 2*9 + 3*2, 1*8 + 2*1 + 3*3] = [7+18+6, 8+2+9] = [31, 19]
  // Row 1: [4*7 + 5*9 + 6*2, 4*8 + 5*1 + 6*3] = [28+45+12, 32+5+18] = [85, 55]
  const expected = [31, 19, 85, 55];

  const C1 = new Float32Array(4);
  matmulNaive(A, B, C1, 2, 2, 3);
  for (let i = 0; i < 4; i++) assertNear(C1[i], expected[i]);

  const C2 = new Float32Array(4);
  matmulReordered(A, B, C2, 2, 2, 3);
  for (let i = 0; i < 4; i++) assertNear(C2[i], expected[i]);

  const C3 = new Float32Array(4);
  matmulTiled(A, B, C3, 2, 2, 3, 2); // tile size 2
  for (let i = 0; i < 4; i++) assertNear(C3[i], expected[i]);
});

test("MatMul - Matrix-Vector multiply", () => {
  // A is 2x3, x is 3
  const A = new Float32Array([
    1, 2, 3,
    4, 5, 6
  ]);
  const x = new Float32Array([0.5, 1, 2]);

  // Expected out:
  // [1*0.5 + 2*1 + 3*2, 4*0.5 + 5*1 + 6*2] = [0.5+2+6, 2+5+12] = [8.5, 19]
  const out = new Float32Array(2);
  matvecmul(A, 0, x, 0, out, 0, 2, 3);

  assertNear(out[0], 8.5);
  assertNear(out[1], 19.0);
});

test("MatMul - Quantized Q8_0 Matrix-Vector multiply", () => {
  // M=1, K=32 (1 block)
  // Scale = 1.0 (in float16: 0x3C00)
  // 32 int8 weights: [1, 2, ..., 32]
  const rawQ8 = new Uint8Array(34);
  const view = new DataView(rawQ8.buffer);
  view.setUint16(0, 0x3C00, true); // scale = 1.0
  for (let i = 0; i < 32; i++) {
    rawQ8[2 + i] = (i + 1);
  }

  const x = new Float32Array(32).fill(1.0);
  const out = new Float32Array(1);

  matvecmulQ8_0(rawQ8, 0, x, 0, out, 0, 1, 32);

  // Sum of 1..32 = 32 * 33 / 2 = 528
  assertNear(out[0], 528.0);
});

test("MatMul - Quantized Q4_0 Matrix-Vector multiply", () => {
  // M=1, K=32 (1 block, 18 bytes)
  // Scale = 1.0 (in float16: 0x3C00)
  // Nibbles = 8 (which maps to 8 - 8 = 0)
  const rawQ4 = new Uint8Array(18);
  const view = new DataView(rawQ4.buffer);
  view.setUint16(0, 0x3C00, true); // scale = 1.0
  // Each byte contains two nibbles 0x88 -> (8-8)=0, (8-8)=0
  for (let i = 0; i < 16; i++) {
    rawQ4[2 + i] = 0x88;
  }

  const x = new Float32Array(32).fill(2.0);
  const out = new Float32Array(1);

  matvecmulQ4_0(rawQ4, 0, x, 0, out, 0, 1, 32);

  assertNear(out[0], 0.0);
});
