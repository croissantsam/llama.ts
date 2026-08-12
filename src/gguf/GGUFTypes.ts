/**
 * GGUF type definitions.
 *
 * Based on the GGUF specification:
 * https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
 */

// ─── GGML tensor types ───────────────────────────────────────────────

/** Tensor data type enum matching the GGUF spec exactly. */
export enum GGMLType {
  F32      = 0,
  F16      = 1,
  Q4_0     = 2,
  Q4_1     = 3,
  // Q4_2 = 4,  removed
  // Q4_3 = 5,  removed
  Q5_0     = 6,
  Q5_1     = 7,
  Q8_0     = 8,
  Q8_1     = 9,
  Q2_K     = 10,
  Q3_K     = 11,
  Q4_K     = 12,
  Q5_K     = 13,
  Q6_K     = 14,
  Q8_K     = 15,
  IQ2_XXS  = 16,
  IQ2_XS   = 17,
  IQ3_XXS  = 18,
  IQ1_S    = 19,
  IQ4_NL   = 20,
  IQ3_S    = 21,
  IQ2_S    = 22,
  IQ4_XS   = 23,
  I8       = 24,
  I16      = 25,
  I32      = 26,
  I64      = 27,
  F64      = 28,
  IQ1_M    = 29,
  BF16     = 30,
  TQ1_0    = 34,
  TQ2_0    = 35,
  MXFP4    = 39,
}

/** Number of elements per block for quantized types. */
export const GGML_BLOCK_SIZES: Partial<Record<GGMLType, number>> = {
  [GGMLType.F32]:    1,
  [GGMLType.F16]:    1,
  [GGMLType.Q4_0]:   32,
  [GGMLType.Q4_1]:   32,
  [GGMLType.Q5_0]:   32,
  [GGMLType.Q5_1]:   32,
  [GGMLType.Q8_0]:   32,
  [GGMLType.Q8_1]:   32,
  [GGMLType.Q2_K]:   256,
  [GGMLType.Q3_K]:   256,
  [GGMLType.Q4_K]:   256,
  [GGMLType.Q5_K]:   256,
  [GGMLType.Q6_K]:   256,
  [GGMLType.Q8_K]:   256,
  [GGMLType.BF16]:   1,
  [GGMLType.I8]:     1,
  [GGMLType.I16]:    1,
  [GGMLType.I32]:    1,
  [GGMLType.I64]:    1,
  [GGMLType.F64]:    1,
};

/** Byte size per block for quantized types. */
export const GGML_TYPE_SIZES: Partial<Record<GGMLType, number>> = {
  [GGMLType.F32]:    4,
  [GGMLType.F16]:    2,
  [GGMLType.Q4_0]:   18,   // 2 (scale) + 16 (32 * 4 bits)
  [GGMLType.Q4_1]:   20,   // 2 (scale) + 2 (min) + 16
  [GGMLType.Q5_0]:   22,   // 2 + 4 + 16
  [GGMLType.Q5_1]:   24,   // 2 + 2 + 4 + 16
  [GGMLType.Q8_0]:   34,   // 2 (scale) + 32 (32 * 8 bits)
  [GGMLType.Q8_1]:   36,   // 2 + 2 + 32
  [GGMLType.Q2_K]:   84,
  [GGMLType.Q3_K]:   110,
  [GGMLType.Q4_K]:   144,
  [GGMLType.Q5_K]:   176,
  [GGMLType.Q6_K]:   210,
  [GGMLType.Q8_K]:   292,
  [GGMLType.BF16]:   2,
  [GGMLType.I8]:     1,
  [GGMLType.I16]:    2,
  [GGMLType.I32]:    4,
  [GGMLType.I64]:    8,
  [GGMLType.F64]:    8,
};

// ─── GGUF metadata value types ───────────────────────────────────────

/** Metadata value type enum. */
export enum GGUFMetadataValueType {
  UINT8   = 0,
  INT8    = 1,
  UINT16  = 2,
  INT16   = 3,
  UINT32  = 4,
  INT32   = 5,
  FLOAT32 = 6,
  BOOL    = 7,
  STRING  = 8,
  ARRAY   = 9,
  UINT64  = 10,
  INT64   = 11,
  FLOAT64 = 12,
}

// ─── GGUF file structures ────────────────────────────────────────────

/** GGUF magic number: "GGUF" in little-endian. */
export const GGUF_MAGIC = 0x46554747;

/** Default alignment in bytes. */
export const GGUF_DEFAULT_ALIGNMENT = 32;

/** Parsed GGUF header. */
export interface GGUFHeader {
  magic: number;
  version: number;
  tensorCount: bigint;
  metadataCount: bigint;
}

/** Metadata value — can be a primitive, string, or nested array. */
export type GGUFValue =
  | number
  | bigint
  | boolean
  | string
  | GGUFValue[];

/** Information about a single tensor in the file. */
export interface GGUFTensorInfo {
  name: string;
  nDimensions: number;
  dimensions: number[];
  type: GGMLType;
  offset: bigint;
}

/** Fully parsed GGUF file (without tensor data). */
export interface GGUFFile {
  header: GGUFHeader;
  metadata: Map<string, GGUFValue>;
  tensors: GGUFTensorInfo[];
  /** Absolute byte offset where tensor data begins in the file. */
  dataOffset: number;
}

/**
 * Compute the total number of elements in a tensor from its dimensions.
 */
export function tensorElementCount(dimensions: number[]): number {
  let count = 1;
  for (let i = 0; i < dimensions.length; i++) {
    count *= dimensions[i];
  }
  return count;
}

/**
 * Compute the byte size of a tensor's data given its element count and type.
 */
export function tensorByteSize(elementCount: number, type: GGMLType): number {
  const blockSize = GGML_BLOCK_SIZES[type];
  const typeSize = GGML_TYPE_SIZES[type];
  if (blockSize === undefined || typeSize === undefined) {
    throw new Error(`Unsupported tensor type: ${type}`);
  }
  if (elementCount % blockSize !== 0) {
    throw new Error(
      `Element count ${elementCount} is not divisible by block size ${blockSize} for type ${type}`
    );
  }
  return (elementCount / blockSize) * typeSize;
}
