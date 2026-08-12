import test from "node:test";
import assert from "node:assert/strict";
import { sampleGreedy, sampleWithParams, PRNG } from "../../src/generation/Sampler.js";

test("Sampler - PRNG determinism", () => {
  const rng1 = new PRNG(12345);
  const rng2 = new PRNG(12345);

  const val1 = [rng1.nextFloat(), rng1.nextFloat(), rng1.nextFloat()];
  const val2 = [rng2.nextFloat(), rng2.nextFloat(), rng2.nextFloat()];

  assert.deepEqual(val1, val2);
});

test("Sampler - sampleGreedy", () => {
  const logits = new Float32Array([1.0, 5.0, 2.0, 0.5]);
  assert.equal(sampleGreedy(logits), 1);
});

test("Sampler - sampleWithParams deterministic behavior", () => {
  const logits = new Float32Array([1.0, 5.0, 2.0, 0.5]);
  const rng = new PRNG(42);

  // With very low temperature (near 0), it should behave like greedy
  const sample = sampleWithParams(new Float32Array(logits), 0.001, 0, 1.0, rng);
  assert.equal(sample, 1);
});
