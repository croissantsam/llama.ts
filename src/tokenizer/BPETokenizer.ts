/**
 * BPE (Byte-Pair Encoding) tokenizer loaded from GGUF metadata.
 *
 * Supports the tokenizer format used by LLaMA / Qwen models:
 * - Vocabulary loaded from "tokenizer.ggml.tokens"
 * - Scores from "tokenizer.ggml.scores"
 * - Merges from "tokenizer.ggml.merges" (optional)
 *
 * Two modes:
 * 1. Score-based BPE (like SentencePiece): uses token scores to find best merges
 * 2. Merge-based BPE (like GPT/Qwen): uses explicit merge rules with priority ordering
 */
import type { Tokenizer } from "./Tokenizer.js";
import type { GGUFValue } from "../gguf/GGUFTypes.js";
import {
  getTokenizerVocab,
  getTokenizerScores,
  getTokenizerMerges,
  getBOSTokenId,
  getEOSTokenId,
  getTokenizerType,
} from "../gguf/GGUFMetadata.js";

/**
 * Create a BPE tokenizer from GGUF metadata.
 */
export function createBPETokenizer(metadata: Map<string, GGUFValue>): BPETokenizer {
  const vocab = getTokenizerVocab(metadata);
  const scores = getTokenizerScores(metadata);
  const merges = getTokenizerMerges(metadata);
  const bosTokenId = getBOSTokenId(metadata);
  const eosTokenId = getEOSTokenId(metadata);
  const tokenizerType = getTokenizerType(metadata);

  return new BPETokenizer(vocab, scores, merges, bosTokenId, eosTokenId, tokenizerType);
}

/**
 * Build GPT-2 / Qwen byte-to-unicode and unicode-to-byte mappings.
 */
function buildByteUnicodeMaps(): { b2u: Map<number, string>; u2b: Map<string, number> } {
  const bs: number[] = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);

  const cs = [...bs];
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  }

  const b2u = new Map<number, string>();
  const u2b = new Map<string, number>();
  for (let i = 0; i < 256; i++) {
    const ch = String.fromCharCode(cs[i]);
    b2u.set(bs[i], ch);
    u2b.set(ch, bs[i]);
  }

  return { b2u, u2b };
}

const { b2u, u2b } = buildByteUnicodeMaps();

export class BPETokenizer implements Tokenizer {
  /** Token ID → string piece. */
  private vocab: string[];
  /** Token scores (for score-based BPE). */
  private scores: Float32Array;
  /** Merge rules: "piece1 piece2" → priority (lower = higher priority). */
  private mergeRanks: Map<string, number> | null;
  /** String piece → token ID (reverse lookup). */
  private tokenToId: Map<string, number>;

  readonly bosTokenId: number;
  readonly eosTokenId: number;
  readonly vocabSize: number;

  private tokenizerType: string;

  constructor(
    vocab: string[],
    scores: Float32Array,
    merges: string[] | null,
    bosTokenId: number,
    eosTokenId: number,
    tokenizerType: string,
  ) {
    this.vocab = vocab;
    this.scores = scores;
    this.bosTokenId = bosTokenId;
    this.eosTokenId = eosTokenId;
    this.vocabSize = vocab.length;
    this.tokenizerType = tokenizerType;

    // Build reverse lookup
    this.tokenToId = new Map();
    for (let i = 0; i < vocab.length; i++) {
      this.tokenToId.set(vocab[i], i);
    }

    // Build merge ranks if merges are provided
    if (merges && merges.length > 0) {
      this.mergeRanks = new Map();
      for (let i = 0; i < merges.length; i++) {
        this.mergeRanks.set(merges[i], i);
      }
    } else {
      this.mergeRanks = null;
    }
  }

  /**
   * Encode text into token IDs.
   */
  encode(text: string): number[] {
    if (text.length === 0) return [];

    // Step 1: Convert to UTF-8 bytes
    const encoder = new TextEncoder();
    const bytes = encoder.encode(text);

    // Step 2: Initialize with byte tokens via byte-to-unicode map
    const tokens: number[] = [];
    for (let i = 0; i < bytes.length; i++) {
      const b = bytes[i];
      const char = b2u.get(b);
      let id = char !== undefined ? this.tokenToId.get(char) : undefined;
      if (id === undefined) {
        id = this.findByteToken(b);
      }
      tokens.push(id !== -1 && id !== undefined ? id : 0);
    }

    // Step 3: Iteratively merge using BPE
    if (this.mergeRanks) {
      this.applyMergeBPE(tokens);
    } else {
      this.applyScoreBPE(tokens);
    }

    return tokens;
  }

  /**
   * Score-based BPE merging (SentencePiece-style).
   */
  private applyScoreBPE(tokens: number[]): void {
    while (tokens.length >= 2) {
      let bestScore = -Infinity;
      let bestIdx = -1;
      let bestId = -1;

      for (let i = 0; i < tokens.length - 1; i++) {
        const merged = this.vocab[tokens[i]] + this.vocab[tokens[i + 1]];
        const id = this.tokenToId.get(merged);
        if (id !== undefined && this.scores[id] > bestScore) {
          bestScore = this.scores[id];
          bestIdx = i;
          bestId = id;
        }
      }

      if (bestIdx === -1) break;

      tokens[bestIdx] = bestId;
      tokens.splice(bestIdx + 1, 1);
    }
  }

  /**
   * Merge-based BPE (GPT/Qwen-style).
   */
  private applyMergeBPE(tokens: number[]): void {
    if (!this.mergeRanks) return;

    while (tokens.length >= 2) {
      let bestRank = Infinity;
      let bestIdx = -1;

      for (let i = 0; i < tokens.length - 1; i++) {
        const pair = `${this.vocab[tokens[i]]} ${this.vocab[tokens[i + 1]]}`;
        const rank = this.mergeRanks.get(pair);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestIdx = i;
        }
      }

      if (bestIdx === -1) break;

      const merged = this.vocab[tokens[bestIdx]] + this.vocab[tokens[bestIdx + 1]];
      const mergedId = this.tokenToId.get(merged);
      if (mergedId === undefined) break;

      tokens[bestIdx] = mergedId;
      tokens.splice(bestIdx + 1, 1);
    }
  }

  /**
   * Find a byte fallback token for a given byte value.
   */
  private findByteToken(byte: number): number {
    const hex = byte.toString(16).toUpperCase().padStart(2, "0");
    const patterns = [`<0x${hex}>`, `<0x${hex.toLowerCase()}>`];

    for (const pattern of patterns) {
      const id = this.tokenToId.get(pattern);
      if (id !== undefined) return id;
    }
    return -1;
  }

  /**
   * Decode token IDs to text.
   */
  decode(tokens: number[]): string {
    const byteBuf: number[] = [];

    for (const token of tokens) {
      if (token < 0 || token >= this.vocab.length) continue;
      const piece = this.vocab[token];

      // Handle byte fallback tokens: <0xHH>
      if (piece.startsWith("<0x") && piece.endsWith(">")) {
        const hexStr = piece.slice(3, -1);
        const byte = parseInt(hexStr, 16);
        if (!isNaN(byte)) {
          byteBuf.push(byte);
          continue;
        }
      }

      // Convert unicode chars back to raw bytes using u2b map
      for (let i = 0; i < piece.length; i++) {
        const ch = piece[i];
        const byte = u2b.get(ch);
        if (byte !== undefined) {
          byteBuf.push(byte);
        } else {
          // SentencePiece space marker or standard character
          const code = ch === "▁" || ch === " " ? 32 : ch.charCodeAt(0);
          byteBuf.push(code);
        }
      }
    }

    return new TextDecoder("utf-8").decode(new Uint8Array(byteBuf));
  }

  /**
   * Decode a single token ID to its string piece.
   */
  decodeToken(token: number): string {
    return this.decode([token]);
  }
}
