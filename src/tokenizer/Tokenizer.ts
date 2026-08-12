/**
 * Tokenizer interface.
 */
export interface Tokenizer {
  /** Encode text into token IDs. */
  encode(text: string): number[];

  /** Decode token IDs back into text. */
  decode(tokens: number[]): string;

  /** Decode a single token ID. */
  decodeToken(token: number): string;

  /** Beginning-of-sequence token ID. */
  readonly bosTokenId: number;

  /** End-of-sequence token ID. */
  readonly eosTokenId: number;

  /** Vocabulary size. */
  readonly vocabSize: number;
}
