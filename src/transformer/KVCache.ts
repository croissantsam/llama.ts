/**
 * Key-Value cache for autoregressive generation.
 *
 * Stores K and V for all past positions, per layer.
 * Never recalculates historical attention.
 */
import type { ModelConfig } from "./ModelConfig.js";

export class KVCache {
  /** Key cache: [nLayers][contextLength * kvDim]. */
  readonly keyCache: Float32Array[];
  /** Value cache: [nLayers][contextLength * kvDim]. */
  readonly valueCache: Float32Array[];

  readonly kvDim: number;
  readonly contextLength: number;
  readonly nLayers: number;

  constructor(config: ModelConfig) {
    this.kvDim = config.kvDim;
    this.contextLength = config.contextLength;
    this.nLayers = config.nLayers;

    const layerCacheSize = config.contextLength * config.kvDim;

    this.keyCache = new Array(config.nLayers);
    this.valueCache = new Array(config.nLayers);

    for (let l = 0; l < config.nLayers; l++) {
      this.keyCache[l] = new Float32Array(layerCacheSize);
      this.valueCache[l] = new Float32Array(layerCacheSize);
    }
  }

  /**
   * Store K and V for a given layer and position.
   */
  store(
    layer: number,
    position: number,
    k: Float32Array, kOffset: number,
    v: Float32Array, vOffset: number,
  ): void {
    const cacheOffset = position * this.kvDim;
    const kCache = this.keyCache[layer];
    const vCache = this.valueCache[layer];

    for (let i = 0; i < this.kvDim; i++) {
      kCache[cacheOffset + i] = k[kOffset + i];
      vCache[cacheOffset + i] = v[vOffset + i];
    }
  }

  /**
   * Get the key cache for a layer. Returns the full buffer;
   * caller must use position * kvDim to index.
   */
  getKeys(layer: number): Float32Array {
    return this.keyCache[layer];
  }

  /**
   * Get the value cache for a layer.
   */
  getValues(layer: number): Float32Array {
    return this.valueCache[layer];
  }

  /**
   * Reset the cache (for a new generation session).
   */
  reset(): void {
    for (let l = 0; l < this.nLayers; l++) {
      this.keyCache[l].fill(0);
      this.valueCache[l].fill(0);
    }
  }

  /**
   * Estimated memory usage in bytes.
   */
  memoryUsage(): number {
    return this.nLayers * 2 * this.contextLength * this.kvDim * 4;
  }
}
