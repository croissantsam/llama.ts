# Instruction — llama.ts : moteur LLM natif en TypeScript + ScriptC

## 0. Vision

Construire un moteur d'inférence LLM **100 % TypeScript**, compilable avec **ScriptC**, inspiré de l'architecture de `llama.cpp`.

Le projet doit permettre de charger un modèle au format GGUF et de générer du texte localement, sans dépendre de :

- C++
- Rust
- Go
- Python pour le runtime
- llama.cpp
- bindings vers llama.cpp

L'objectif est expérimental :

> **Démontrer jusqu'où TypeScript + ScriptC peuvent aller pour construire un moteur d'inférence LLM natif.**

Le projet ne doit pas chercher à reproduire immédiatement toutes les fonctionnalités de llama.cpp.

La priorité est :

1. correction ;
2. simplicité ;
3. architecture propre ;
4. TypeScript pur ;
5. compilation ScriptC ;
6. benchmarks ;
7. optimisation progressive.

---

# 1. Objectif final

L'API utilisateur doit ressembler à :

```ts
import { Llama } from "./llama";

const model = await Llama.load("./models/model.gguf");

const result = await model.generate({
  prompt: "Explain quantum computing in simple terms.",
  maxTokens: 100,
  temperature: 0.7,
});

console.log(result.text);
```

Le moteur doit pouvoir fonctionner en deux modes :

```text
TypeScript / Node
        │
        ▼
   llama.ts

TypeScript / ScriptC
        │
        ▼
   llama native
```

Le même code source doit être réutilisé autant que possible.

---

# 2. Architecture

Structure recommandée :

```text
llama-ts/
│
├── src/
│   ├── api/
│   │   ├── Llama.ts
│   │   ├── Model.ts
│   │   └── Generation.ts
│   │
│   ├── gguf/
│   │   ├── GGUFReader.ts
│   │   ├── GGUFHeader.ts
│   │   ├── GGUFMetadata.ts
│   │   └── TensorReader.ts
│   │
│   ├── tensor/
│   │   ├── Tensor.ts
│   │   ├── TensorShape.ts
│   │   ├── TensorType.ts
│   │   └── TensorOps.ts
│   │
│   ├── quantization/
│   │   ├── Q4.ts
│   │   ├── Q5.ts
│   │   ├── Q8.ts
│   │   └── Dequantize.ts
│   │
│   ├── math/
│   │   ├── MatMul.ts
│   │   ├── Softmax.ts
│   │   ├── RMSNorm.ts
│   │   ├── RoPE.ts
│   │   └── VectorOps.ts
│   │
│   ├── transformer/
│   │   ├── Transformer.ts
│   │   ├── Attention.ts
│   │   ├── FeedForward.ts
│   │   ├── Layer.ts
│   │   └── KVCache.ts
│   │
│   ├── tokenizer/
│   │   ├── Tokenizer.ts
│   │   └── SentencePiece.ts
│   │
│   ├── generation/
│   │   ├── Sampler.ts
│   │   ├── Temperature.ts
│   │   ├── TopK.ts
│   │   ├── TopP.ts
│   │   └── Generator.ts
│   │
│   └── runtime/
│       ├── Runtime.ts
│       ├── NodeRuntime.ts
│       └── ScriptCRuntime.ts
│
├── benchmarks/
│   ├── startup/
│   ├── prompt-processing/
│   ├── generation/
│   ├── memory/
│   └── kernels/
│
├── tests/
│   ├── gguf/
│   ├── tensors/
│   ├── quantization/
│   ├── math/
│   ├── transformer/
│   └── generation/
│
├── models/
│   └── README.md
│
├── package.json
├── tsconfig.json
└── README.md
```

---

# 3. Règle absolue : TypeScript uniquement

Le code du moteur doit être exclusivement :

```text
.ts
```

ou éventuellement :

```text
.tsx
```

pour une interface future.

Interdit :

```text
.cpp
.c
.rs
.go
.py
```

Ne pas utiliser de bibliothèque native externe pour les opérations principales.

Si une optimisation n'est pas possible avec les capacités actuelles de ScriptC, elle doit être documentée plutôt que contournée avec du C++ ou Rust.

---

# 4. Runtime abstraction

Créer une abstraction minimale pour les opérations système :

```ts
export interface Runtime {
  readFile(path: string): Uint8Array;

  openFile(path: string): FileHandle;

  now(): number;

  exit(code: number): never;
}
```

Créer :

```text
NodeRuntime
ScriptCRuntime
```

L'objectif est de ne pas coupler le moteur aux APIs Node.

Le cœur mathématique du moteur ne doit dépendre d'aucun runtime.

---

# 5. GGUF

Le premier objectif fonctionnel est de lire correctement un fichier GGUF.

Implémenter :

```text
GGUF header
metadata
tensor information
tensor offsets
tensor data
```

Le reader doit vérifier :

- magic ;
- version ;
- nombre de tensors ;
- nombre de metadata entries ;
- offsets ;
- tailles ;
- types.

Exemple conceptuel :

```ts
interface GGUFHeader {
  magic: number;
  version: number;
  tensorCount: bigint;
  metadataCount: bigint;
}
```

Ne pas utiliser de librairie GGUF externe.

---

# 6. Metadata

Lire notamment les informations nécessaires à l'architecture du modèle :

```text
architecture
context length
embedding length
block count
attention heads
key/value heads
feed-forward length
rope parameters
vocabulary size
```

Le moteur doit être capable d'afficher :

```text
Model
-----------------------------
Architecture: ...
Layers:       ...
Embedding:    ...
Heads:        ...
KV Heads:     ...
Context:      ...
Vocabulary:   ...
```

---

# 7. Tensor abstraction

Créer une abstraction légère :

```ts
export interface Tensor {
  shape: number[];
  type: TensorType;
  data: Uint8Array;
}
```

Ne pas convertir systématiquement tous les tensors en `Float32Array`.

Pour les modèles quantifiés, conserver les données dans leur représentation native.

Exemple :

```text
Q4
 ↓
packed bytes
 ↓
quantized matmul
```

plutôt que :

```text
Q4
 ↓
Float32
 ↓
MatMul
```

L'objectif est de limiter :

- mémoire ;
- allocations ;
- copies ;
- bandwidth.

---

# 8. Commencer avec FP32

Avant d'implémenter les formats quantifiés, supporter un chemin FP32 simple.

Pipeline :

```text
GGUF
 ↓
FP32 tensors
 ↓
Transformer
 ↓
logits
 ↓
sampling
 ↓
token
```

Cela permet de valider le moteur indépendamment de la complexité de la quantification.

---

# 9. Tensor operations

Implémenter les opérations fondamentales :

```text
add
sub
mul
scale
dot
sum
mean
max
matmul
```

Exemple :

```ts
export function dot(
  a: Float32Array,
  b: Float32Array,
): number {
  let result = 0;

  for (let i = 0; i < a.length; i++) {
    result += a[i] * b[i];
  }

  return result;
}
```

Les implémentations doivent être simples au départ.

Chaque opération doit avoir des tests numériques.

---

# 10. MatMul

`MatMul` est un des benchmarks les plus importants du projet.

Commencer par :

```ts
C = A × B
```

avec une implémentation naïve.

Puis créer progressivement :

```text
MatMul V1
→ naïve

MatMul V2
→ loop ordering

MatMul V3
→ blocking / tiling

MatMul V4
→ SIMD si supporté

MatMul V5
→ quantized kernels
```

Chaque version doit être benchmarkée.

Ne pas remplacer une version fonctionnelle sans conserver la possibilité de comparer les performances.

---

# 11. RMSNorm

Implémenter :

```text
RMSNorm(x)
```

avec la formule standard utilisée par les architectures ciblées.

Créer des tests contre des valeurs de référence.

---

# 12. RoPE

Implémenter Rotary Positional Embeddings.

Prévoir :

```text
RoPE
 ├── position
 ├── dimension
 ├── theta
 └── scaling
```

Ne pas supposer qu'un seul modèle utilise exactement les mêmes paramètres.

---

# 13. Softmax

Implémenter une version numériquement stable.

Ne jamais faire naïvement :

```ts
Math.exp(x)
```

sur des valeurs arbitraires sans normalisation.

Utiliser une stratégie :

```text
max
 ↓
x - max
 ↓
exp
 ↓
sum
 ↓
normalize
```

Ajouter des tests pour :

- valeurs positives ;
- valeurs négatives ;
- grands écarts ;
- petits écarts ;
- vecteurs longs.

---

# 14. Attention

Implémenter progressivement :

```text
Q
K
V
 ↓
QKᵀ
 ↓
scale
 ↓
mask
 ↓
softmax
 ↓
V
```

Commencer avec une implémentation simple et correcte.

---

# 15. KV Cache

La génération autoregressive doit utiliser une KV cache.

Structure :

```text
KVCache
├── layer 0
│   ├── K
│   └── V
├── layer 1
│   ├── K
│   └── V
└── ...
```

Ne jamais recalculer toute l'attention historique à chaque token si le modèle permet l'utilisation d'une KV cache.

---

# 16. Transformer block

Construire :

```text
Input
  ↓
RMSNorm
  ↓
Attention
  ↓
Residual
  ↓
RMSNorm
  ↓
Feed Forward
  ↓
Residual
```

L'architecture doit être modulaire.

Ne pas hardcoder les dimensions.

---

# 17. Feed Forward

Commencer avec l'architecture correspondant au premier modèle ciblé.

Le code doit séparer :

```text
gate
up
activation
down
```

L'activation doit être configurable selon l'architecture.

---

# 18. Premier modèle cible

Ne pas viser immédiatement les gros modèles.

Commencer par un petit modèle compatible GGUF :

```text
0.5B – 1B paramètres
```

Le modèle doit :

- tenir facilement en RAM ;
- avoir une architecture documentée ;
- être disponible en GGUF ;
- permettre des tests rapides.

Le modèle exact doit être choisi avant l'implémentation finale de l'architecture.

---

# 19. Tokenizer

Implémenter le tokenizer requis par le premier modèle.

Séparer :

```text
encode(text)
decode(tokens)
```

API :

```ts
interface Tokenizer {
  encode(text: string): number[];
  decode(tokens: number[]): string;
}
```

Le tokenizer doit être testé indépendamment du transformer.

---

# 20. Generation

API publique :

```ts
interface GenerateOptions {
  prompt: string;
  maxTokens: number;
  temperature?: number;
  topK?: number;
  topP?: number;
  seed?: number;
}
```

Pipeline :

```text
prompt
 ↓
tokenizer
 ↓
tokens
 ↓
model
 ↓
logits
 ↓
sampler
 ↓
next token
 ↓
KV cache
 ↓
repeat
```

---

# 21. Sampling

Implémenter :

```text
greedy
temperature
top-k
top-p
seeded random
```

Commencer par greedy decoding pour valider le moteur.

Puis ajouter les autres stratégies.

---

# 22. Streaming

Prévoir :

```ts
for await (const token of model.generateStream(...)) {
  process.stdout.write(token);
}
```

Le moteur doit pouvoir retourner les tokens au fur et à mesure.

Ne pas attendre la fin complète de la génération.

---

# 23. Quantization

Une fois FP32 fonctionnel, implémenter progressivement :

```text
Q8
Q4
```

Priorité au format réellement utilisé par le modèle de benchmark.

Architecture :

```text
QuantizedTensor
       ↓
QuantizedKernel
       ↓
Float output
```

Ne pas déquantifier tout le modèle en mémoire sauf pour les benchmarks de référence.

---

# 24. Quantized MatMul

Le véritable objectif de performance est :

```text
Q4 weights
     ↓
quantized matmul
     ↓
FP output
```

et non :

```text
Q4
 ↓
FP32 copy
 ↓
MatMul
```

Le kernel quantifié doit être benchmarké séparément.

---

# 25. Memory management

Éviter les allocations dans la boucle token-by-token.

Interdit autant que possible :

```ts
new Float32Array(...)
```

à chaque token.

Préallouer :

```text
KV cache
activations
temporary buffers
logits
```

et les réutiliser.

Objectif :

```text
generate()
 ↓
allocations minimales
 ↓
stable memory usage
```

---

# 26. Performance architecture

Le moteur doit séparer :

```text
Model loading
Prompt processing
Token generation
Sampling
```

Mesurer séparément :

```text
model load time
prompt tokens/sec
generation tokens/sec
first-token latency
memory usage
```

---

# 27. Benchmark obligatoire

Créer un benchmark standard :

```text
Model:
Prompt:
Context:
Generation:
```

Mesurer :

```text
Startup
Model load
Prompt processing
Tokens/sec
First token
Peak RSS
Model memory
Binary size
```

---

# 28. Comparaison Node vs ScriptC

Le même moteur TypeScript doit être exécuté :

```text
Node
```

puis :

```text
ScriptC
```

Ne pas modifier l'algorithme entre les deux benchmarks.

Comparer :

```text
Node
vs
ScriptC
```

sur :

```text
GGUF parsing
MatMul
RMSNorm
RoPE
Softmax
Attention
Generation
```

---

# 29. Comparaison llama.cpp

Le benchmark de référence doit également utiliser llama.cpp.

La comparaison doit préciser :

```text
hardware
OS
CPU
RAM
model
quantization
context
threads
prompt
generation length
```

Ne jamais comparer deux configurations différentes sans le signaler.

---

# 30. Benchmark format

Utiliser un format comme :

```text
==================================================
llama-ts benchmark
==================================================

Model:        ...
Quantization: Q4
Context:      2048
Prompt:       256 tokens
Generation:   128 tokens

Engine              tok/s       RAM
--------------------------------------------------
Node llama-ts       ...
ScriptC llama-ts    ...
llama.cpp            ...
--------------------------------------------------
```

---

# 31. Kernel benchmarks

Créer des benchmarks indépendants :

```text
bench-matmul
bench-rmsnorm
bench-softmax
bench-rope
bench-attention
bench-quantized-matmul
```

Cela permettra de savoir précisément où les performances sont perdues.

---

# 32. Profiling

Le code doit permettre d'activer :

```text
LLAMA_TS_PROFILE=1
```

Résultat :

```text
MatMul       42.3%
Attention    21.1%
RMSNorm       4.2%
RoPE          2.1%
Sampling      0.4%
Other        30.0%
```

L'objectif est de concentrer les optimisations sur les vrais bottlenecks.

---

# 33. Optimisation ScriptC

Ne jamais supposer qu'une optimisation est utile.

Workflow obligatoire :

```text
baseline
   ↓
benchmark
   ↓
optimization
   ↓
benchmark
   ↓
keep/revert
```

Chaque optimisation doit avoir un commit identifiable.

---

# 34. SIMD

Étudier les capacités réelles de ScriptC avant d'implémenter une couche SIMD.

Si une fonctionnalité SIMD est disponible, créer une abstraction :

```ts
VectorOps
```

avec :

```text
scalar implementation
SIMD implementation
```

Le moteur doit pouvoir fallback sur la version scalaire.

Ne jamais ajouter de code C/C++ uniquement pour obtenir SIMD.

---

# 35. Multithreading

Étudier les primitives de concurrence réellement disponibles dans ScriptC.

Si les workers/threads sont supportés :

```text
MatMul
 ├── worker 1
 ├── worker 2
 ├── worker 3
 └── worker 4
```

Tester l'évolution :

```text
1 thread
2 threads
4 threads
8 threads
```

Mesurer le scaling.

---

# 36. Electron future integration

Le moteur doit pouvoir être utilisé plus tard depuis Electron.

Prévoir une API CLI :

```bash
llama-ts \
  --model model.gguf \
  --prompt "Hello" \
  --max-tokens 100
```

Puis une API processus :

```text
Electron
    ↓
spawn llama-ts
    ↓
JSON Lines
    ↓
stream tokens
```

Cela permettra d'intégrer le moteur dans une application Electron sans modifier le cœur du moteur.

---

# 37. API future Electron

Prévoir à terme :

```ts
const engine = new NativeLlamaEngine();

const stream = engine.generate({
  prompt: "...",
});

for await (const token of stream) {
  // afficher token
}
```

Le moteur Electron ne doit jamais connaître les détails GGUF, MatMul ou quantization.

---

# 38. Tests

Chaque composant mathématique doit avoir des tests.

Minimum :

```text
GGUF reader
Tensor
MatMul
Dot
Softmax
RMSNorm
RoPE
Attention
KV cache
Tokenizer
Sampling
Generation
```

Les tests doivent utiliser des valeurs de référence calculées indépendamment lorsque possible.

---

# 39. Tests de précision

Comparer les résultats avec une référence.

Pour les opérations numériques, utiliser des tolérances :

```text
absolute error
relative error
```

Ne jamais utiliser uniquement :

```ts
a === b
```

pour des résultats flottants.

---

# 40. Golden tests

Conserver quelques petits modèles / fixtures de test adaptés à leur licence.

Un test doit pouvoir vérifier :

```text
model
+
prompt
→
expected token sequence
```

ou une tolérance sur les logits.

---

# 41. CLI

Créer une CLI :

```bash
llama-ts --model ./model.gguf
```

Options :

```text
--model
--prompt
--max-tokens
--temperature
--top-k
--top-p
--seed
--threads
--context
--benchmark
--profile
```

---

# 42. Exemple

Commande :

```bash
llama-ts \
  --model ./models/model-q4.gguf \
  --prompt "What is TypeScript?" \
  --max-tokens 100
```

Output :

```text
TypeScript is ...
```

Avec :

```bash
llama-ts \
  --model ./models/model-q4.gguf \
  --benchmark
```

Output :

```text
Model load:       812 ms
Prompt:           54.2 tok/s
Generation:       31.7 tok/s
First token:      41 ms
Peak RSS:         428 MB
```

---

# 43. Phases de développement

## Phase 1 — Infrastructure

- [ ] TypeScript
- [ ] build Node
- [ ] build ScriptC
- [ ] runtime abstraction
- [ ] CLI
- [ ] tests

## Phase 2 — GGUF

- [ ] header
- [ ] metadata
- [ ] tensor metadata
- [ ] tensor loading
- [ ] FP32

## Phase 3 — Math

- [ ] vector operations
- [ ] MatMul
- [ ] Softmax
- [ ] RMSNorm
- [ ] RoPE

## Phase 4 — Transformer

- [ ] attention
- [ ] KV cache
- [ ] feed-forward
- [ ] transformer blocks
- [ ] logits

## Phase 5 — Tokenizer

- [ ] encode
- [ ] decode
- [ ] special tokens

## Phase 6 — Generation

- [ ] greedy
- [ ] temperature
- [ ] top-k
- [ ] top-p
- [ ] streaming

## Phase 7 — Quantization

- [ ] Q8
- [ ] Q4
- [ ] quantized matmul

## Phase 8 — Native optimization

- [ ] profiling
- [ ] allocation reduction
- [ ] cache optimization
- [ ] SIMD investigation
- [ ] threading
- [ ] kernel optimization

## Phase 9 — Benchmark

- [ ] Node
- [ ] ScriptC
- [ ] llama.cpp
- [ ] CPU
- [ ] RAM
- [ ] startup
- [ ] tokens/sec

---

# 44. Contraintes importantes

Ne pas :

- copier du code de llama.cpp ;
- dépendre de llama.cpp ;
- appeler llama.cpp ;
- écrire des bindings C++ ;
- utiliser Rust pour les kernels ;
- utiliser Python pour l'inférence ;
- optimiser avant d'avoir une baseline correcte ;
- sacrifier la lisibilité sans benchmark.

Le projet peut s'inspirer des concepts et architectures connus des moteurs LLM open source, mais l'implémentation doit être originale en TypeScript.

---

# 45. Philosophie de développement

Toujours privilégier :

```text
Correct
   ↓
Simple
   ↓
Measured
   ↓
Optimized
```

et jamais :

```text
Optimized
   ↓
Impossible to debug
```

Le moteur doit rester compréhensible par un développeur TypeScript.

---

# 46. Critère de réussite minimal

Le premier milestone est atteint lorsque :

```text
GGUF
 ↓
Tokenizer
 ↓
Transformer
 ↓
KV cache
 ↓
Sampling
 ↓
Generated text
```

fonctionne avec un petit modèle.

Le deuxième milestone est atteint lorsque le même code fonctionne sous ScriptC.

Le troisième est atteint lorsqu'une comparaison directe Node vs ScriptC peut être produite.

Le quatrième est atteint lorsqu'une comparaison avec llama.cpp est disponible.

---

# 47. Ambition finale

Le projet doit viser à démontrer :

> **Un moteur d'inférence LLM local écrit entièrement en TypeScript, compilé nativement avec ScriptC, capable de charger des modèles GGUF et de générer du texte avec des performances mesurables.**

La question scientifique/technique centrale du projet est :

> **Jusqu'où peut-on pousser du TypeScript compilé nativement avant d'atteindre les limites de ScriptC par rapport à un moteur C/C++ spécialisé comme llama.cpp ?**

La performance n'est pas supposée à l'avance.

Elle doit être mesurée.