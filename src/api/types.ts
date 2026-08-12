/**
 * Public API types.
 */

export interface GenerateOptions {
  /** The input prompt text. */
  prompt: string;
  /** Maximum number of tokens to generate. */
  maxTokens: number;
  /** Temperature for sampling (0 = greedy, default 1.0). */
  temperature?: number;
  /** Top-K filtering (0 = disabled). */
  topK?: number;
  /** Top-P nucleus filtering (1.0 = disabled). */
  topP?: number;
  /** Random seed for reproducible sampling. */
  seed?: number;
}

export interface GenerateResult {
  /** The generated text. */
  text: string;
  /** Number of tokens generated. */
  tokenCount: number;
  /** Number of prompt tokens. */
  promptTokens: number;
  /** Prompt processing time (ms). */
  promptTimeMs: number;
  /** Generation time (ms). */
  generateTimeMs: number;
  /** Tokens per second during generation. */
  tokensPerSecond: number;
  /** Time to first token (ms). */
  firstTokenMs: number;
}
