# llama.ts

> A 100% TypeScript LLM inference engine, targeting both Node.js and ScriptC.

## What is this?

**llama.ts** is an experimental project to build an LLM inference engine entirely in TypeScript. It can load GGUF model files and generate text — no C++, Rust, Go, Python, or native bindings required.

The central question:

> **How far can TypeScript compiled natively go for building an LLM inference engine?**

## Quick Start

```bash
npm install
npm run build

# Show model info
node dist/cli/main.js --model ./models/model.gguf --info

# Generate text
node dist/cli/main.js --model ./models/model.gguf --prompt "Hello, world!" --max-tokens 100

# Benchmark
node dist/cli/main.js --model ./models/model.gguf --benchmark
```

## API

```ts
import { Llama } from "./src/api/Llama.js";

const model = await Llama.load("./models/model.gguf");

const result = await model.generate({
  prompt: "Explain quantum computing in simple terms.",
  maxTokens: 100,
  temperature: 0.7,
});

console.log(result.text);
```

### Streaming

```ts
for await (const token of model.generateStream({ prompt: "Hello", maxTokens: 50 })) {
  process.stdout.write(token);
}
```

## Architecture

```
src/
├── api/          # Public API (Llama, types)
├── cli/          # CLI entry point
├── gguf/         # GGUF file reader
├── tensor/       # Tensor abstraction
├── math/         # Math kernels (MatMul, Softmax, RMSNorm, RoPE)
├── transformer/  # Transformer blocks (Attention, FFN, KV Cache)
├── tokenizer/    # BPE tokenizer
├── generation/   # Sampling and generation loop
└── runtime/      # Runtime abstraction (Node / ScriptC)
```

## Models

Place GGUF models in the `./models/` directory. Recommended first model:

- **Qwen2.5-0.5B-Instruct** (F16 GGUF) — small, well-documented, standard architecture

## Development

```bash
npm run build     # Compile TypeScript
npm run dev       # Watch mode
npm test          # Run tests
```

## License

MIT
