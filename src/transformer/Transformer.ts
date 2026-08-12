/**
 * Transformer — full forward pass for one token.
 *
 * Architecture (LLaMA-style):
 *   embedding lookup
 *   for each layer:
 *     RMSNorm → Attention → residual
 *     RMSNorm → FeedForward (SwiGLU) → residual
 *   final RMSNorm
 *   output projection → logits
 */
import type { ModelConfig } from "./ModelConfig.js";
import type { TransformerWeights } from "./TransformerWeights.js";
import { RunState } from "./RunState.js";
import { KVCache } from "./KVCache.js";
import { rmsNorm } from "../math/RMSNorm.js";
import { applyRoPE } from "../math/RoPE.js";
import { softmax } from "../math/Softmax.js";
import { siluElementwiseMul } from "../math/Activations.js";
import { matvecmul } from "../math/MatMul.js";
import { vecAddInPlace, vecDot, vecCopy } from "../math/VectorOps.js";

export class Transformer {
  readonly config: ModelConfig;
  readonly weights: TransformerWeights;
  readonly state: RunState;
  readonly kvCache: KVCache;

  constructor(config: ModelConfig, weights: TransformerWeights) {
    this.config = config;
    this.weights = weights;
    this.state = new RunState(config);
    this.kvCache = new KVCache(config);
  }

  /**
   * Forward pass for a single token at a given position.
   * Returns the logits array (owned by RunState — do not retain across calls).
   *
   * @param token    Token ID
   * @param position Position in the sequence (0-indexed)
   * @returns Float32Array of logits (length = vocabSize)
   */
  forward(token: number, position: number): Float32Array {
    const { config, weights, state, kvCache } = this;
    const { dim, nHeads, nKVHeads, headDim, kvDim, hiddenDim, vocabSize } = config;

    // ─── Embedding lookup ──────────────────────────────────
    // Copy the embedding for this token into the activation vector
    const embOffset = token * dim;
    for (let i = 0; i < dim; i++) {
      state.x[i] = weights.tokenEmbedding[embOffset + i];
    }

    // ─── Transformer layers ────────────────────────────────
    for (let l = 0; l < config.nLayers; l++) {
      const layer = weights.layers[l];

      // --- Attention block ---

      // 1. RMSNorm before attention
      rmsNorm(state.x, 0, layer.attnNorm, 0, state.xb, 0, dim, config.normEps);

      // 2. QKV projections (mat-vec multiply)
      matvecmul(layer.wq, 0, state.xb, 0, state.q, 0, dim, dim);
      matvecmul(layer.wk, 0, state.xb, 0, state.k, 0, kvDim, dim);
      matvecmul(layer.wv, 0, state.xb, 0, state.v, 0, kvDim, dim);

      // 3. Apply RoPE to Q and K
      applyRoPE(state.q, 0, position, headDim, nHeads, config.ropeTheta, config.ropeFreqScale);
      applyRoPE(state.k, 0, position, headDim, nKVHeads, config.ropeTheta, config.ropeFreqScale);

      // 4. Store K and V in cache
      kvCache.store(l, position, state.k, 0, state.v, 0);

      // 5. Multi-head attention with GQA
      const kCache = kvCache.getKeys(l);
      const vCache = kvCache.getValues(l);
      const seqLen = position + 1; // attend to all positions up to and including current

      // How many Q heads share each KV head
      const kvGroupSize = nHeads / nKVHeads;

      for (let h = 0; h < nHeads; h++) {
        // Which KV head does this Q head use?
        const kvHead = Math.floor(h / kvGroupSize);

        const qOffset = h * headDim;
        const attOffset = h * config.contextLength;

        // Compute attention scores: Q · K^T / sqrt(headDim)
        const scale = 1.0 / Math.sqrt(headDim);
        for (let t = 0; t < seqLen; t++) {
          const kOffset = t * kvDim + kvHead * headDim;
          let score = 0;
          for (let d = 0; d < headDim; d++) {
            score += state.q[qOffset + d] * kCache[kOffset + d];
          }
          state.att[attOffset + t] = score * scale;
        }

        // Softmax over attention scores
        softmax(state.att, attOffset, seqLen);

        // Weighted sum of values
        // Clear the output for this head
        const xbOffset = h * headDim;
        for (let d = 0; d < headDim; d++) {
          state.xb[xbOffset + d] = 0;
        }
        for (let t = 0; t < seqLen; t++) {
          const weight = state.att[attOffset + t];
          const vOffset = t * kvDim + kvHead * headDim;
          for (let d = 0; d < headDim; d++) {
            state.xb[xbOffset + d] += weight * vCache[vOffset + d];
          }
        }
      }

      // 6. Output projection
      // state.xb now contains the concatenated head outputs [dim]
      // Project back: xb2 = Wo × xb
      matvecmul(layer.wo, 0, state.xb, 0, state.xb2, 0, dim, dim);

      // 7. Residual connection
      vecAddInPlace(state.x, 0, state.xb2, 0, dim);

      // --- Feed-Forward block ---

      // 1. RMSNorm before FFN
      rmsNorm(state.x, 0, layer.ffnNorm, 0, state.xb, 0, dim, config.normEps);

      // 2. SwiGLU FFN
      // gate = W1 × xb
      matvecmul(layer.w1, 0, state.xb, 0, state.hb, 0, hiddenDim, dim);
      // up = W3 × xb
      matvecmul(layer.w3, 0, state.xb, 0, state.hb2, 0, hiddenDim, dim);

      // 3. SiLU(gate) * up
      siluElementwiseMul(state.hb, 0, state.hb2, 0, state.hb, 0, hiddenDim);

      // 4. Down projection: xb = W2 × hb
      matvecmul(layer.w2, 0, state.hb, 0, state.xb, 0, dim, hiddenDim);

      // 5. Residual connection
      vecAddInPlace(state.x, 0, state.xb, 0, dim);
    }

    // ─── Final norm ────────────────────────────────────────
    rmsNorm(state.x, 0, weights.finalNorm, 0, state.x, 0, dim, config.normEps);

    // ─── Output projection → logits ────────────────────────
    matvecmul(weights.output, 0, state.x, 0, state.logits, 0, vocabSize, dim);

    return state.logits;
  }

  /**
   * Reset the KV cache for a new generation session.
   */
  reset(): void {
    this.kvCache.reset();
  }
}
