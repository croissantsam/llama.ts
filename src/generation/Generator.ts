/**
 * Text generation loop.
 *
 * Pipeline:
 *   prompt → tokenize → prefill → sample → generate → decode
 *
 * Supports both batch result and streaming via AsyncGenerator.
 */
import type { Tokenizer } from "../tokenizer/Tokenizer.js";
import type { Transformer } from "../transformer/Transformer.js";
import { sampleGreedy, sampleWithParams, PRNG } from "./Sampler.js";

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
  /** Whether to prepend BOS token. */
  addBOS?: boolean;
}

export interface GenerateResult {
  /** The generated text (excluding the prompt). */
  text: string;
  /** Total number of tokens generated. */
  tokenCount: number;
  /** Number of prompt tokens processed. */
  promptTokens: number;
  /** Time to process the prompt (ms). */
  promptTimeMs: number;
  /** Time to generate all tokens (ms). */
  generateTimeMs: number;
  /** Tokens per second during generation. */
  tokensPerSecond: number;
  /** Time to first generated token (ms). */
  firstTokenMs: number;
}

/**
 * Generator — drives the autoregressive generation loop.
 */
export class Generator {
  private transformer: Transformer;
  private tokenizer: Tokenizer;

  constructor(transformer: Transformer, tokenizer: Tokenizer) {
    this.transformer = transformer;
    this.tokenizer = tokenizer;
  }

  /**
   * Generate text and return the complete result.
   */
  async generate(options: GenerateOptions): Promise<GenerateResult> {
    const tokens: string[] = [];
    let promptTokens = 0;
    let promptTimeMs = 0;
    let firstTokenMs = 0;
    const genStart = performance.now();

    let isFirst = true;
    for await (const event of this.generateEvents(options)) {
      if (event.type === "prefill_done") {
        promptTokens = event.promptTokens;
        promptTimeMs = event.timeMs;
      } else if (event.type === "token") {
        if (isFirst) {
          firstTokenMs = performance.now() - genStart;
          isFirst = false;
        }
        tokens.push(event.text);
      }
    }

    const totalTime = performance.now() - genStart;
    const generateTimeMs = totalTime - promptTimeMs;
    const tokenCount = tokens.length;

    return {
      text: tokens.join(""),
      tokenCount,
      promptTokens,
      promptTimeMs,
      generateTimeMs,
      tokensPerSecond: tokenCount > 0 ? (tokenCount / generateTimeMs) * 1000 : 0,
      firstTokenMs,
    };
  }

  /**
   * Streaming generation — yields tokens as they are generated.
   */
  async *generateStream(options: GenerateOptions): AsyncGenerator<string> {
    for await (const event of this.generateEvents(options)) {
      if (event.type === "token") {
        yield event.text;
      }
    }
  }

  /**
   * Internal event-based generation loop.
   */
  private async *generateEvents(options: GenerateOptions): AsyncGenerator<GenerateEvent> {
    const {
      prompt,
      maxTokens,
      temperature = 1.0,
      topK = 0,
      topP = 1.0,
      seed = 42,
      addBOS = true,
    } = options;

    const rng = new PRNG(seed);

    // Reset KV cache for new generation
    this.transformer.reset();

    // Tokenize prompt
    let promptTokens = this.tokenizer.encode(prompt);
    if (addBOS) {
      promptTokens = [this.tokenizer.bosTokenId, ...promptTokens];
    }

    // ─── Prefill: process all prompt tokens ────────────────
    const prefillStart = performance.now();

    for (let i = 0; i < promptTokens.length; i++) {
      this.transformer.forward(promptTokens[i], i);
    }

    const prefillTime = performance.now() - prefillStart;
    yield {
      type: "prefill_done",
      promptTokens: promptTokens.length,
      timeMs: prefillTime,
    };

    // ─── Generation: sample new tokens ─────────────────────
    let position = promptTokens.length;
    let prevToken = promptTokens[promptTokens.length - 1];

    // Get logits from the last prefill step
    let logits = this.transformer.state.logits;

    for (let i = 0; i < maxTokens; i++) {
      // Sample next token from logits
      let nextToken: number;

      if (temperature === 0 || temperature <= 1e-8) {
        nextToken = sampleGreedy(logits);
      } else {
        // Clone logits because sampling modifies them in-place
        const logitsCopy = new Float32Array(logits);
        nextToken = sampleWithParams(logitsCopy, temperature, topK, topP, rng);
      }

      // Check for EOS
      if (nextToken === this.tokenizer.eosTokenId) {
        break;
      }

      // Decode and yield
      const text = this.tokenizer.decodeToken(nextToken);
      yield { type: "token", text, tokenId: nextToken };

      // Forward pass for the new token
      logits = this.transformer.forward(nextToken, position);
      position++;
      prevToken = nextToken;
    }
  }
}

type GenerateEvent =
  | { type: "prefill_done"; promptTokens: number; timeMs: number }
  | { type: "token"; text: string; tokenId: number };
