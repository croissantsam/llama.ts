#!/usr/bin/env node
/**
 * llama-ts CLI
 *
 * Usage:
 *   llama-ts --model ./model.gguf --prompt "Hello" --max-tokens 100
 *   llama-ts --model ./model.gguf --info
 *   llama-ts --model ./model.gguf --benchmark
 */
import { NodeRuntime } from "../runtime/NodeRuntime.js";
import { readGGUF } from "../gguf/GGUFReader.js";
import { printModelInfo } from "../gguf/GGUFMetadata.js";
import { Llama } from "../api/Llama.js";

// ─── Argument parsing ──────────────────────────────────────

interface CLIArgs {
  model: string;
  prompt: string;
  maxTokens: number;
  temperature: number;
  topK: number;
  topP: number;
  seed: number;
  info: boolean;
  benchmark: boolean;
}

function parseArgs(args: string[]): CLIArgs {
  const result: CLIArgs = {
    model: "",
    prompt: "",
    maxTokens: 128,
    temperature: 0.7,
    topK: 40,
    topP: 0.9,
    seed: 42,
    info: false,
    benchmark: false,
  };

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case "--model":
        result.model = args[++i] ?? "";
        break;
      case "--prompt":
        result.prompt = args[++i] ?? "";
        break;
      case "--max-tokens":
        result.maxTokens = parseInt(args[++i] ?? "128", 10);
        break;
      case "--temperature":
        result.temperature = parseFloat(args[++i] ?? "0.7");
        break;
      case "--top-k":
        result.topK = parseInt(args[++i] ?? "40", 10);
        break;
      case "--top-p":
        result.topP = parseFloat(args[++i] ?? "0.9");
        break;
      case "--seed":
        result.seed = parseInt(args[++i] ?? "42", 10);
        break;
      case "--info":
        result.info = true;
        break;
      case "--benchmark":
        result.benchmark = true;
        break;
      case "--help":
      case "-h":
        printHelp();
        process.exit(0);
    }
  }

  return result;
}

function printHelp(): void {
  console.log(`
llama-ts — TypeScript LLM inference engine

Usage:
  llama-ts --model <path> [options]

Options:
  --model <path>       Path to GGUF model file (required)
  --prompt <text>      Input prompt text
  --max-tokens <n>     Maximum tokens to generate (default: 128)
  --temperature <f>    Sampling temperature (default: 0.7)
  --top-k <n>          Top-K filtering (default: 40)
  --top-p <f>          Top-P nucleus filtering (default: 0.9)
  --seed <n>           Random seed (default: 42)
  --info               Show model information and exit
  --benchmark          Run benchmark mode
  --help, -h           Show this help message

Examples:
  llama-ts --model model.gguf --info
  llama-ts --model model.gguf --prompt "What is TypeScript?" --max-tokens 100
  llama-ts --model model.gguf --benchmark
`);
}

// ─── Main ──────────────────────────────────────────────────

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));

  if (!args.model) {
    console.error("Error: --model is required");
    printHelp();
    process.exit(1);
  }

  const runtime = new NodeRuntime();

  // ─── Info mode ─────────────────────────────
  if (args.info) {
    const gguf = readGGUF(args.model, runtime);
    printModelInfo(gguf);
    return;
  }

  // ─── Load model ────────────────────────────
  const model = await Llama.load(args.model, runtime);

  // ─── Benchmark mode ────────────────────────
  if (args.benchmark) {
    const benchPrompt = args.prompt || "The meaning of life is";
    const benchMaxTokens = args.maxTokens || 128;

    console.log("Running benchmark...");
    console.log(`  Prompt: "${benchPrompt}"`);
    console.log(`  Max tokens: ${benchMaxTokens}`);
    console.log();

    const result = await model.generate({
      prompt: benchPrompt,
      maxTokens: benchMaxTokens,
      temperature: 0,
      seed: args.seed,
    });

    console.log();
    console.log("══════════════════════════════════════════");
    console.log("llama-ts benchmark");
    console.log("══════════════════════════════════════════");
    console.log(`  Prompt tokens:   ${result.promptTokens}`);
    console.log(`  Generated:       ${result.tokenCount} tokens`);
    console.log(`  Prompt time:     ${result.promptTimeMs.toFixed(1)} ms`);
    console.log(`  Generation time: ${result.generateTimeMs.toFixed(1)} ms`);
    console.log(`  First token:     ${result.firstTokenMs.toFixed(1)} ms`);
    console.log(`  Speed:           ${result.tokensPerSecond.toFixed(1)} tok/s`);
    console.log(`  Prompt speed:    ${(result.promptTokens / result.promptTimeMs * 1000).toFixed(1)} tok/s`);
    console.log("══════════════════════════════════════════");
    console.log();
    console.log("Generated text:");
    console.log(result.text);
    return;
  }

  // ─── Generation mode ──────────────────────
  if (!args.prompt) {
    console.error("Error: --prompt is required for generation mode");
    process.exit(1);
  }

  console.log("Generating...\n");

  // Stream tokens to stdout
  for await (const token of model.generateStream({
    prompt: args.prompt,
    maxTokens: args.maxTokens,
    temperature: args.temperature,
    topK: args.topK,
    topP: args.topP,
    seed: args.seed,
  })) {
    process.stdout.write(token);
  }
  console.log("\n");
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
