/**
 * Model configuration extracted from GGUF metadata.
 */
import type { GGUFFile, GGUFValue } from "../gguf/GGUFTypes.js";
import {
  getArchitecture,
  getContextLength,
  getEmbeddingLength,
  getBlockCount,
  getHeadCount,
  getKVHeadCount,
  getFeedForwardLength,
  getRopeTheta,
  getRopeScalingFactor,
  getNormEps,
  getVocabSize,
} from "../gguf/GGUFMetadata.js";

export interface ModelConfig {
  /** Model architecture name (e.g., "llama", "qwen2"). */
  architecture: string;
  /** Embedding dimension (d_model). */
  dim: number;
  /** Feed-forward hidden dimension. */
  hiddenDim: number;
  /** Number of transformer layers. */
  nLayers: number;
  /** Number of attention query heads. */
  nHeads: number;
  /** Number of key/value heads (GQA). */
  nKVHeads: number;
  /** Vocabulary size. */
  vocabSize: number;
  /** Maximum context length. */
  contextLength: number;
  /** RoPE base frequency. */
  ropeTheta: number;
  /** RoPE frequency scaling factor. */
  ropeFreqScale: number;
  /** RMSNorm epsilon. */
  normEps: number;
  /** Dimension per attention head. */
  headDim: number;
  /** Key/value dimension (nKVHeads * headDim). */
  kvDim: number;
}

/**
 * Extract model configuration from parsed GGUF file.
 */
export function configFromGGUF(gguf: GGUFFile): ModelConfig {
  const m = gguf.metadata;

  const architecture = getArchitecture(m);
  const dim = getEmbeddingLength(m);
  const nHeads = getHeadCount(m);
  const headDim = dim / nHeads;
  const nKVHeads = getKVHeadCount(m);
  const kvDim = nKVHeads * headDim;

  return {
    architecture,
    dim,
    hiddenDim: getFeedForwardLength(m),
    nLayers: getBlockCount(m),
    nHeads,
    nKVHeads,
    vocabSize: getVocabSize(m),
    contextLength: getContextLength(m),
    ropeTheta: getRopeTheta(m),
    ropeFreqScale: getRopeScalingFactor(m),
    normEps: getNormEps(m),
    headDim,
    kvDim,
  };
}
