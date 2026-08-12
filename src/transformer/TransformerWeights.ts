/**
 * Transformer weight structures and GGUF loading.
 *
 * Maps GGUF tensor names to the weight arrays used by the transformer.
 */
import type { ModelConfig } from "./ModelConfig.js";
import type { TensorLoader } from "../gguf/TensorLoader.js";

/** Weights for a single transformer layer. */
export interface LayerWeights {
  /** RMSNorm weight before attention. */
  attnNorm: Float32Array;
  /** Query projection weight (dim × dim). */
  wq: Float32Array;
  /** Key projection weight (dim × kvDim). */
  wk: Float32Array;
  /** Value projection weight (dim × kvDim). */
  wv: Float32Array;
  /** Output projection weight (dim × dim). */
  wo: Float32Array;
  /** RMSNorm weight before FFN. */
  ffnNorm: Float32Array;
  /** Gate projection weight for SwiGLU (dim × hiddenDim). */
  w1: Float32Array;
  /** Down projection weight (hiddenDim × dim). */
  w2: Float32Array;
  /** Up projection weight (dim × hiddenDim). */
  w3: Float32Array;
}

/** All weights for the transformer model. */
export interface TransformerWeights {
  /** Token embedding matrix (vocabSize × dim). */
  tokenEmbedding: Float32Array;
  /** Per-layer weights. */
  layers: LayerWeights[];
  /** Final RMSNorm weight. */
  finalNorm: Float32Array;
  /** Output projection / LM head (vocabSize × dim). */
  output: Float32Array;
}

/**
 * Load all transformer weights from a GGUF TensorLoader.
 *
 * Supports common GGUF tensor naming conventions:
 * - llama.cpp style: "blk.{i}.attn_q.weight", etc.
 * - also: "token_embd.weight", "output_norm.weight", "output.weight"
 */
export function loadWeights(
  loader: TensorLoader,
  config: ModelConfig,
): TransformerWeights {
  const startTime = performance.now();

  // Token embedding
  const tokenEmbedding = loader.loadTensorF32("token_embd.weight");

  // Per-layer weights
  const layers: LayerWeights[] = new Array(config.nLayers);
  for (let i = 0; i < config.nLayers; i++) {
    const prefix = `blk.${i}`;
    layers[i] = {
      attnNorm: loader.loadTensorF32(`${prefix}.attn_norm.weight`),
      wq:       loader.loadTensorF32(`${prefix}.attn_q.weight`),
      wk:       loader.loadTensorF32(`${prefix}.attn_k.weight`),
      wv:       loader.loadTensorF32(`${prefix}.attn_v.weight`),
      wo:       loader.loadTensorF32(`${prefix}.attn_output.weight`),
      ffnNorm:  loader.loadTensorF32(`${prefix}.ffn_norm.weight`),
      w1:       loader.loadTensorF32(`${prefix}.ffn_gate.weight`),
      w2:       loader.loadTensorF32(`${prefix}.ffn_down.weight`),
      w3:       loader.loadTensorF32(`${prefix}.ffn_up.weight`),
    };

    if ((i + 1) % 4 === 0 || i === config.nLayers - 1) {
      const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);
      console.log(`  Loaded layer ${i + 1}/${config.nLayers} (${elapsed}s)`);
    }
  }

  // Final norm
  const finalNorm = loader.loadTensorF32("output_norm.weight");

  // Output / LM head — some models tie this to the embedding
  let output: Float32Array;
  try {
    output = loader.loadTensorF32("output.weight");
  } catch {
    // Tied embeddings: reuse token embedding as LM head
    console.log("  Note: output.weight not found, using tied embeddings");
    output = tokenEmbedding;
  }

  const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);
  console.log(`  All weights loaded in ${elapsed}s`);

  return { tokenEmbedding, layers, finalNorm, output };
}
