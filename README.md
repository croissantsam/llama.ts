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

## Benchmarks

Run the kernel benchmark suite:

```bash
npm run bench
```

### Kernel Performance Results

| Kernel | Matrix / Tensor Shape | Execution Time | Throughput | Notes |
|---|---|---|---|---|
| **MatVecMul (FP32)** | $896 \times 896$ | $978.43\ \mu\text{s/op}$ | $1,022.0\ \text{ops/s}$ | Single-token inference pass |
| **MatVecMul (Q8_0)** | $896 \times 896$ | $563.23\ \mu\text{s/op}$ | **$1,775.5\ \text{ops/s}$** | **1.73x speedup vs FP32** |
| **MatVecMul (Q4_0)** | $896 \times 896$ | $563.73\ \mu\text{s/op}$ | **$1,773.9\ \text{ops/s}$** | **1.73x speedup vs FP32** |
| **RMSNorm** | $\text{dim}=896$ | $1.05\ \mu\text{s/op}$ | $952,853.6\ \text{ops/s}$ | Layer pre-normalization |
| **Softmax** | $\text{seq\_len}=512$ | $1.29\ \mu\text{s/op}$ | $773,850.3\ \text{ops/s}$ | Attention score normalization |
| **RoPE** | $14\ \text{heads} \times 64\ \text{dim}$ | $7.65\ \mu\text{s/op}$ | $130,664.2\ \text{ops/s}$ | Rotary positional embeddings |
| **SwiGLU Activation** | $\text{hidden\_dim}=4864$ | $17.66\ \mu\text{s/op}$ | $56,613.5\ \text{ops/s}$ | Fused SiLU elementwise mul |
| **MatMul V2 (Reordered)** | $256 \times 256 \times 256$ | $14,848.38\ \mu\text{s/op}$ | $67.3\ \text{ops/s}$ | Reordered $ikj$ loop |
| **MatMul V3 (Tiled 64)** | $256 \times 256 \times 256$ | $15,071.47\ \mu\text{s/op}$ | $66.4\ \text{ops/s}$ | L1/L2 blocked matrix product |

### End-to-End Model Benchmark

Run an end-to-end inference benchmark on a loaded GGUF model:

```bash
npm start -- --model ./models/qwen2.5-0.5b-instruct-q8_0.gguf --benchmark --max-tokens 100
```

---

## License

MIT
