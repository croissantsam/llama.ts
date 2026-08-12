/**
 * Kernel Benchmarks for llama.ts
 *
 * Measures kernel execution speed for:
 * - MatMul (Naive vs Reordered vs Tiled)
 * - Matrix-Vector multiply (inference hot-path)
 * - RMSNorm
 * - Softmax
 * - RoPE
 * - SwiGLU activation
 */
import { performance } from "node:perf_hooks";
import {
  matmulNaive,
  matmulReordered,
  matmulTiled,
  matvecmul,
} from "../src/math/MatMul.js";
import { rmsNorm } from "../src/math/RMSNorm.js";
import { softmax } from "../src/math/Softmax.js";
import { applyRoPE } from "../src/math/RoPE.js";
import { siluElementwiseMul } from "../src/math/Activations.js";

function bench(name: string, fn: () => void, iterations: number): void {
  // Warmup
  for (let i = 0; i < Math.min(iterations, 10); i++) fn();

  const start = performance.now();
  for (let i = 0; i < iterations; i++) {
    fn();
  }
  const end = performance.now();
  const totalMs = end - start;
  const avgUs = (totalMs / iterations) * 1000;
  const opsPerSec = (iterations / totalMs) * 1000;

  console.log(
    `  ${name.padEnd(32)} ${avgUs.toFixed(2).padStart(10)} µs/op  ${opsPerSec.toFixed(1).padStart(10)} ops/s`
  );
}

console.log("==================================================");
console.log("llama-ts Kernel Benchmarks");
console.log("==================================================");
console.log();

// ─── 1. MatMul Benchmarks ──────────────────────────────────────
console.log("1. Matrix Multiplication (M=256, N=256, K=256)");
console.log("--------------------------------------------------");
{
  const M = 256, N = 256, K = 256;
  const A = new Float32Array(M * K).fill(0.5);
  const B = new Float32Array(K * N).fill(0.5);
  const C = new Float32Array(M * N);
  const iters = 20;

  bench("MatMul V1 (Naive)", () => matmulNaive(A, B, C, M, N, K), iters);
  bench("MatMul V2 (Reordered ikj)", () => matmulReordered(A, B, C, M, N, K), iters);
  bench("MatMul V3 (Tiled 64)", () => matmulTiled(A, B, C, M, N, K, 64), iters);
}
console.log();

// ─── 2. Mat-Vec Multiply (Single token inference pass) ─────────
console.log("2. Matrix-Vector Multiply (M=896, K=896 - Qwen 0.5B size)");
console.log("--------------------------------------------------");
{
  const M = 896, K = 896;
  const A = new Float32Array(M * K).fill(0.1);
  const x = new Float32Array(K).fill(1.0);
  const out = new Float32Array(M);
  const iters = 5000;

  bench("MatVecMul (896x896)", () => matvecmul(A, 0, x, 0, out, 0, M, K), iters);
}
console.log();

// ─── 3. RMSNorm Benchmark ──────────────────────────────────────
console.log("3. RMSNorm (dim=896)");
console.log("--------------------------------------------------");
{
  const dim = 896;
  const x = new Float32Array(dim).fill(1.5);
  const weight = new Float32Array(dim).fill(1.0);
  const out = new Float32Array(dim);
  const iters = 50000;

  bench("RMSNorm (896)", () => rmsNorm(x, 0, weight, 0, out, 0, dim, 1e-5), iters);
}
console.log();

// ─── 4. Softmax Benchmark ──────────────────────────────────────
console.log("4. Softmax (seq_len=512)");
console.log("--------------------------------------------------");
{
  const len = 512;
  const x = new Float32Array(len);
  for (let i = 0; i < len; i++) x[i] = Math.sin(i);
  const iters = 20000;

  bench("Softmax (512)", () => softmax(x, 0, len), iters);
}
console.log();

// ─── 5. RoPE Benchmark ─────────────────────────────────────────
console.log("5. RoPE (nHeads=14, headDim=64, pos=100)");
console.log("--------------------------------------------------");
{
  const nHeads = 14, headDim = 64;
  const q = new Float32Array(nHeads * headDim).fill(0.8);
  const iters = 20000;

  bench("RoPE (14 heads x 64)", () => applyRoPE(q, 0, 100, headDim, nHeads, 10000), iters);
}
console.log();

// ─── 6. SwiGLU Benchmark ───────────────────────────────────────
console.log("6. SwiGLU Fused Activation (hiddenDim=4864)");
console.log("--------------------------------------------------");
{
  const hiddenDim = 4864;
  const gate = new Float32Array(hiddenDim).fill(0.5);
  const up = new Float32Array(hiddenDim).fill(1.2);
  const out = new Float32Array(hiddenDim);
  const iters = 10000;

  bench("SwiGLU (4864)", () => siluElementwiseMul(gate, 0, up, 0, out, 0, hiddenDim), iters);
}
console.log();

console.log("==================================================");
