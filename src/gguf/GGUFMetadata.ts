/**
 * Typed GGUF metadata extraction helpers.
 *
 * Provides convenient access to model architecture parameters
 * from raw GGUF metadata key-value pairs.
 */
import type { GGUFValue, GGUFFile } from "./GGUFTypes.js";

/**
 * Helper to get a typed metadata value, with an optional default.
 */
function getString(metadata: Map<string, GGUFValue>, key: string, defaultValue?: string): string {
  const val = metadata.get(key);
  if (val === undefined) {
    if (defaultValue !== undefined) return defaultValue;
    throw new Error(`Missing required GGUF metadata key: ${key}`);
  }
  return String(val);
}

function getNumber(metadata: Map<string, GGUFValue>, key: string, defaultValue?: number): number {
  const val = metadata.get(key);
  if (val === undefined) {
    if (defaultValue !== undefined) return defaultValue;
    throw new Error(`Missing required GGUF metadata key: ${key}`);
  }
  return Number(val);
}

function getArray(metadata: Map<string, GGUFValue>, key: string): GGUFValue[] {
  const val = metadata.get(key);
  if (val === undefined || !Array.isArray(val)) {
    throw new Error(`Missing or invalid GGUF metadata array key: ${key}`);
  }
  return val;
}

/**
 * Extract the model architecture name (e.g., "llama", "qwen2").
 */
export function getArchitecture(metadata: Map<string, GGUFValue>): string {
  return getString(metadata, "general.architecture");
}

/**
 * Build a prefixed key for architecture-specific metadata.
 * e.g., "llama.context_length" or "qwen2.context_length"
 */
function archKey(metadata: Map<string, GGUFValue>, suffix: string): string {
  const arch = getArchitecture(metadata);
  return `${arch}.${suffix}`;
}

export function getContextLength(metadata: Map<string, GGUFValue>): number {
  return getNumber(metadata, archKey(metadata, "context_length"));
}

export function getEmbeddingLength(metadata: Map<string, GGUFValue>): number {
  return getNumber(metadata, archKey(metadata, "embedding_length"));
}

export function getBlockCount(metadata: Map<string, GGUFValue>): number {
  return getNumber(metadata, archKey(metadata, "block_count"));
}

export function getHeadCount(metadata: Map<string, GGUFValue>): number {
  return getNumber(metadata, archKey(metadata, "attention.head_count"));
}

export function getKVHeadCount(metadata: Map<string, GGUFValue>): number {
  const arch = getArchitecture(metadata);
  const key = `${arch}.attention.head_count_kv`;
  const val = metadata.get(key);
  if (val === undefined) {
    return getHeadCount(metadata);
  }
  if (Array.isArray(val)) {
    const nums = val.map(v => Number(v));
    const nonZero = nums.filter(n => n > 0);
    return nonZero.length > 0 ? Math.max(...nonZero) : getHeadCount(metadata);
  }
  return Number(val);
}

export function getFeedForwardLength(metadata: Map<string, GGUFValue>): number {
  return getNumber(metadata, archKey(metadata, "feed_forward_length"));
}

export function getRopeTheta(metadata: Map<string, GGUFValue>): number {
  return getNumber(metadata, archKey(metadata, "rope.freq_base"), 10000.0);
}

export function getRopeScalingFactor(metadata: Map<string, GGUFValue>): number {
  return getNumber(metadata, archKey(metadata, "rope.scaling.factor"), 1.0);
}

export function getNormEps(metadata: Map<string, GGUFValue>): number {
  return getNumber(metadata, archKey(metadata, "attention.layer_norm_rms_epsilon"), 1e-5);
}

export function getVocabSize(metadata: Map<string, GGUFValue>): number {
  // Vocabulary can be in tokenizer metadata
  const tokens = metadata.get("tokenizer.ggml.tokens");
  if (Array.isArray(tokens)) {
    return tokens.length;
  }
  throw new Error("Cannot determine vocabulary size from metadata");
}

export function getModelName(metadata: Map<string, GGUFValue>): string {
  return getString(metadata, "general.name", "unknown");
}

/**
 * Get tokenizer vocabulary from GGUF metadata.
 */
export function getTokenizerVocab(metadata: Map<string, GGUFValue>): string[] {
  const tokens = getArray(metadata, "tokenizer.ggml.tokens");
  return tokens.map(t => String(t));
}

export function getTokenizerScores(metadata: Map<string, GGUFValue>): Float32Array {
  const scores = metadata.get("tokenizer.ggml.scores");
  if (!Array.isArray(scores)) {
    return new Float32Array(0);
  }
  return new Float32Array(scores.map(s => Number(s)));
}

export function getTokenizerMerges(metadata: Map<string, GGUFValue>): string[] | null {
  const merges = metadata.get("tokenizer.ggml.merges");
  if (!Array.isArray(merges)) return null;
  return merges.map(m => String(m));
}

export function getTokenizerType(metadata: Map<string, GGUFValue>): string {
  return getString(metadata, "tokenizer.ggml.model", "llama");
}

export function getBOSTokenId(metadata: Map<string, GGUFValue>): number {
  return getNumber(metadata, "tokenizer.ggml.bos_token_id", 1);
}

export function getEOSTokenId(metadata: Map<string, GGUFValue>): number {
  return getNumber(metadata, "tokenizer.ggml.eos_token_id", 2);
}

/**
 * Print a summary card of the model.
 */
export function printModelInfo(file: GGUFFile): void {
  const m = file.metadata;
  const arch = getArchitecture(m);

  console.log("Model");
  console.log("─────────────────────────────");
  console.log(`  Name:          ${getModelName(m)}`);
  console.log(`  Architecture:  ${arch}`);
  console.log(`  Layers:        ${getBlockCount(m)}`);
  console.log(`  Embedding:     ${getEmbeddingLength(m)}`);
  console.log(`  Heads:         ${getHeadCount(m)}`);
  console.log(`  KV Heads:      ${getKVHeadCount(m)}`);
  console.log(`  Context:       ${getContextLength(m)}`);
  console.log(`  Vocabulary:    ${getVocabSize(m)}`);
  console.log(`  FF Length:     ${getFeedForwardLength(m)}`);
  console.log(`  RoPE Theta:    ${getRopeTheta(m)}`);
  console.log(`  Norm Eps:      ${getNormEps(m)}`);
  console.log(`  Tensors:       ${file.tensors.length}`);
  console.log(`  GGUF Version:  ${file.header.version}`);
  console.log("─────────────────────────────");
}
