/**
 * Llama — Public API for the llama.ts inference engine.
 *
 * Usage:
 *   const model = await Llama.load("./models/model.gguf");
 *   const result = await model.generate({ prompt: "Hello", maxTokens: 100 });
 *   console.log(result.text);
 */
import { NodeRuntime } from "../runtime/NodeRuntime.js";
import type { Runtime } from "../runtime/Runtime.js";
import { readGGUF } from "../gguf/GGUFReader.js";
import { TensorLoader } from "../gguf/TensorLoader.js";
import { printModelInfo } from "../gguf/GGUFMetadata.js";
import { configFromGGUF, type ModelConfig } from "../transformer/ModelConfig.js";
import { loadWeights } from "../transformer/TransformerWeights.js";
import { Transformer } from "../transformer/Transformer.js";
import { createBPETokenizer } from "../tokenizer/BPETokenizer.js";
import type { Tokenizer } from "../tokenizer/Tokenizer.js";
import { Generator, type GenerateResult } from "../generation/Generator.js";
import type { GenerateOptions } from "./types.js";

export class Llama {
  readonly config: ModelConfig;
  private transformer: Transformer;
  private tokenizer: Tokenizer;
  private generator: Generator;

  private constructor(
    config: ModelConfig,
    transformer: Transformer,
    tokenizer: Tokenizer,
  ) {
    this.config = config;
    this.transformer = transformer;
    this.tokenizer = tokenizer;
    this.generator = new Generator(transformer, tokenizer);
  }

  /**
   * Load a GGUF model file.
   *
   * @param path    Path to the .gguf file
   * @param runtime Optional runtime override (defaults to NodeRuntime)
   */
  static async load(path: string, runtime?: Runtime): Promise<Llama> {
    const rt = runtime ?? new NodeRuntime();

    console.log(`Loading model: ${path}`);
    const loadStart = rt.now();

    // 1. Parse GGUF
    console.log("  Parsing GGUF...");
    const gguf = readGGUF(path, rt);

    // 2. Extract config
    const config = configFromGGUF(gguf);
    console.log(`  Architecture: ${config.architecture}`);
    console.log(`  ${config.nLayers} layers, dim=${config.dim}, heads=${config.nHeads}, kv_heads=${config.nKVHeads}`);
    console.log(`  Vocab: ${config.vocabSize}, Context: ${config.contextLength}`);

    // 3. Load weights
    console.log("  Loading weights...");
    const tensorLoader = new TensorLoader(path, gguf, rt);
    const weights = loadWeights(tensorLoader, config);
    tensorLoader.close();

    // 4. Build transformer
    const transformer = new Transformer(config, weights);

    // 5. Build tokenizer
    console.log("  Building tokenizer...");
    const tokenizer = createBPETokenizer(gguf.metadata);

    const loadTime = ((rt.now() - loadStart) / 1000).toFixed(1);
    console.log(`  Model loaded in ${loadTime}s`);
    console.log();

    return new Llama(config, transformer, tokenizer);
  }

  /**
   * Generate text from a prompt.
   */
  async generate(options: GenerateOptions): Promise<GenerateResult> {
    return this.generator.generate({
      ...options,
      addBOS: true,
    });
  }

  /**
   * Stream generated tokens.
   */
  async *generateStream(options: GenerateOptions): AsyncGenerator<string> {
    yield* this.generator.generateStream({
      ...options,
      addBOS: true,
    });
  }

  /**
   * Print model information card.
   */
  printInfo(path: string, runtime?: Runtime): void {
    const rt = runtime ?? new NodeRuntime();
    const gguf = readGGUF(path, rt);
    printModelInfo(gguf);
  }
}
