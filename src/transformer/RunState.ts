/**
 * Pre-allocated activation buffers for the transformer forward pass.
 *
 * Zero allocations per token — all buffers created once at initialization.
 * This is critical for performance: avoiding GC pressure in the hot loop.
 */
import type { ModelConfig } from "./ModelConfig.js";

export class RunState {
  /** Current activation vector [dim]. */
  x: Float32Array;
  /** After RMSNorm [dim]. */
  xb: Float32Array;
  /** After second RMSNorm [dim]. */
  xb2: Float32Array;
  /** Query projection [dim]. */
  q: Float32Array;
  /** Key projection [kvDim]. */
  k: Float32Array;
  /** Value projection [kvDim]. */
  v: Float32Array;
  /** Attention scores [nHeads × contextLength]. */
  att: Float32Array;
  /** Hidden buffer 1 for FFN [hiddenDim]. */
  hb: Float32Array;
  /** Hidden buffer 2 for FFN [hiddenDim]. */
  hb2: Float32Array;
  /** Output logits [vocabSize]. */
  logits: Float32Array;

  constructor(config: ModelConfig) {
    this.x      = new Float32Array(config.dim);
    this.xb     = new Float32Array(config.dim);
    this.xb2    = new Float32Array(config.dim);
    this.q      = new Float32Array(config.dim);
    this.k      = new Float32Array(config.kvDim);
    this.v      = new Float32Array(config.kvDim);
    this.att    = new Float32Array(config.nHeads * config.contextLength);
    this.hb     = new Float32Array(config.hiddenDim);
    this.hb2    = new Float32Array(config.hiddenDim);
    this.logits = new Float32Array(config.vocabSize);
  }
}
