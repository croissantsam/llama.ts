/**
 * Sampling strategies for token generation.
 *
 * Supported:
 * - Greedy (argmax)
 * - Temperature scaling
 * - Top-K filtering
 * - Top-P (nucleus) filtering
 * - Seeded PRNG for reproducibility
 */
import { softmax } from "../math/Softmax.js";

/**
 * Simple xorshift128 PRNG for reproducible sampling.
 */
export class PRNG {
  private state: Uint32Array;

  constructor(seed: number) {
    this.state = new Uint32Array(4);
    // Initialize state from seed
    this.state[0] = seed;
    this.state[1] = seed ^ 0xDEADBEEF;
    this.state[2] = seed ^ 0x12345678;
    this.state[3] = seed ^ 0xCAFEBABE;
    // Warm up
    for (let i = 0; i < 16; i++) this.next();
  }

  /** Returns a random uint32. */
  next(): number {
    let t = this.state[3];
    const s = this.state[0];
    this.state[3] = this.state[2];
    this.state[2] = this.state[1];
    this.state[1] = s;
    t ^= t << 11;
    t ^= t >>> 8;
    this.state[0] = t ^ s ^ (s >>> 19);
    return this.state[0] >>> 0;
  }

  /** Returns a random float in [0, 1). */
  nextFloat(): number {
    return this.next() / 4294967296.0;
  }
}

/**
 * Greedy sampling: returns the token with the highest logit.
 */
export function sampleGreedy(logits: Float32Array): number {
  let maxIdx = 0;
  let maxVal = logits[0];
  for (let i = 1; i < logits.length; i++) {
    if (logits[i] > maxVal) {
      maxVal = logits[i];
      maxIdx = i;
    }
  }
  return maxIdx;
}

/**
 * Temperature + optional top-k + optional top-p sampling.
 *
 * This modifies the logits array in-place (temperature scaling, softmax).
 */
export function sampleWithParams(
  logits: Float32Array,
  temperature: number,
  topK: number,
  topP: number,
  rng: PRNG,
): number {
  const vocabSize = logits.length;

  // Temperature scaling
  if (temperature !== 1.0 && temperature > 0) {
    const invTemp = 1.0 / temperature;
    for (let i = 0; i < vocabSize; i++) {
      logits[i] *= invTemp;
    }
  }

  // Apply softmax to convert logits to probabilities
  softmax(logits, 0, vocabSize);

  // Top-K filtering
  if (topK > 0 && topK < vocabSize) {
    applyTopK(logits, topK);
  }

  // Top-P (nucleus) filtering
  if (topP > 0 && topP < 1.0) {
    applyTopP(logits, topP);
  }

  // Sample from the probability distribution
  return sampleFromProbs(logits, rng);
}

/**
 * Top-K: keep only the K highest-probability tokens, zero the rest.
 */
function applyTopK(probs: Float32Array, k: number): void {
  const n = probs.length;

  // Find the k-th highest probability using a partial sort approach
  // For small k, a selection-based approach is fine
  const indices = new Uint32Array(n);
  for (let i = 0; i < n; i++) indices[i] = i;

  // Sort indices by probability descending (top-k partial sort)
  indices.sort((a, b) => probs[b] - probs[a]);

  // Zero out everything beyond top-k
  for (let i = k; i < n; i++) {
    probs[indices[i]] = 0;
  }

  // Renormalize
  let sum = 0;
  for (let i = 0; i < n; i++) sum += probs[i];
  if (sum > 0) {
    const invSum = 1.0 / sum;
    for (let i = 0; i < n; i++) probs[i] *= invSum;
  }
}

/**
 * Top-P (nucleus): keep tokens whose cumulative probability reaches P.
 */
function applyTopP(probs: Float32Array, p: number): void {
  const n = probs.length;

  // Sort indices by probability descending
  const indices = new Uint32Array(n);
  for (let i = 0; i < n; i++) indices[i] = i;
  indices.sort((a, b) => probs[b] - probs[a]);

  // Find cutoff
  let cumSum = 0;
  let cutoff = n;
  for (let i = 0; i < n; i++) {
    cumSum += probs[indices[i]];
    if (cumSum >= p) {
      cutoff = i + 1;
      break;
    }
  }

  // Zero out everything beyond cutoff
  for (let i = cutoff; i < n; i++) {
    probs[indices[i]] = 0;
  }

  // Renormalize
  let sum = 0;
  for (let i = 0; i < n; i++) sum += probs[i];
  if (sum > 0) {
    const invSum = 1.0 / sum;
    for (let i = 0; i < n; i++) probs[i] *= invSum;
  }
}

/**
 * Sample a token from a probability distribution.
 */
function sampleFromProbs(probs: Float32Array, rng: PRNG): number {
  const r = rng.nextFloat();
  let cumSum = 0;
  for (let i = 0; i < probs.length; i++) {
    cumSum += probs[i];
    if (r < cumSum) {
      return i;
    }
  }
  // Fallback: return last non-zero token
  return probs.length - 1;
}
