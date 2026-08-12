/**
 * Tensor data loader for GGUF files.
 *
 * Loads tensor data lazily from the GGUF file using a FileHandle.
 * For Phase 1, only FP32 and F16 are supported.
 * F16 tensors are dequantized to FP32 on load.
 */
import type { Runtime, FileHandle } from "../runtime/Runtime.js";
import {
  GGMLType,
  type GGUFFile,
  type GGUFTensorInfo,
  tensorElementCount,
  tensorByteSize,
} from "./GGUFTypes.js";

/**
 * Dequantize F16 (IEEE 754 half-precision) to F32.
 */
function f16ToF32(h: number): number {
  const sign = (h >> 15) & 0x1;
  const exponent = (h >> 10) & 0x1f;
  const mantissa = h & 0x3ff;

  if (exponent === 0) {
    // Subnormal or zero
    if (mantissa === 0) {
      return sign ? -0 : 0;
    }
    // Subnormal: value = (-1)^sign * 2^(-14) * (mantissa / 1024)
    const val = Math.pow(2, -14) * (mantissa / 1024);
    return sign ? -val : val;
  }

  if (exponent === 0x1f) {
    // Infinity or NaN
    if (mantissa === 0) {
      return sign ? -Infinity : Infinity;
    }
    return NaN;
  }

  // Normal number: value = (-1)^sign * 2^(exponent-15) * (1 + mantissa/1024)
  const val = Math.pow(2, exponent - 15) * (1 + mantissa / 1024);
  return sign ? -val : val;
}

/**
 * Dequantize BF16 (bfloat16) to F32.
 * BF16 is simply the upper 16 bits of a F32.
 */
function bf16ToF32(h: number): number {
  // Shift left 16 bits to get the F32 representation
  const buffer = new ArrayBuffer(4);
  const view = new DataView(buffer);
  view.setUint32(0, h << 16, false);
  return view.getFloat32(0, false);
}

/**
 * Loads tensors from a GGUF file on demand.
 */
export class TensorLoader {
  private fileHandle: FileHandle;
  private gguf: GGUFFile;
  private tensorMap: Map<string, GGUFTensorInfo>;

  constructor(path: string, gguf: GGUFFile, runtime: Runtime) {
    this.fileHandle = runtime.openFile(path);
    this.gguf = gguf;
    this.tensorMap = new Map();
    for (const t of gguf.tensors) {
      this.tensorMap.set(t.name, t);
    }
  }

  /** Get tensor info by name. */
  getTensorInfo(name: string): GGUFTensorInfo | undefined {
    return this.tensorMap.get(name);
  }

  /** List all tensor names. */
  getTensorNames(): string[] {
    return Array.from(this.tensorMap.keys());
  }

  /**
   * Load a tensor as Float32Array.
   *
   * For F32: direct copy.
   * For F16: dequantize each element.
   * For BF16: dequantize each element.
   */
  loadTensorF32(name: string): Float32Array {
    const info = this.tensorMap.get(name);
    if (!info) {
      throw new Error(`Tensor not found: ${name}`);
    }

    const elementCount = tensorElementCount(info.dimensions);
    const byteSize = tensorByteSize(elementCount, info.type);

    // Read raw bytes from file
    const rawBytes = new Uint8Array(byteSize);
    const fileOffset = this.gguf.dataOffset + Number(info.offset);
    this.fileHandle.read(rawBytes, 0, byteSize, fileOffset);

    switch (info.type) {
      case GGMLType.F32: {
        // Direct interpretation
        return new Float32Array(rawBytes.buffer, rawBytes.byteOffset, elementCount);
      }

      case GGMLType.F16: {
        // Dequantize F16 → F32
        const f16View = new Uint16Array(rawBytes.buffer, rawBytes.byteOffset, elementCount);
        const f32 = new Float32Array(elementCount);
        for (let i = 0; i < elementCount; i++) {
          f32[i] = f16ToF32(f16View[i]);
        }
        return f32;
      }

      case GGMLType.BF16: {
        // Dequantize BF16 → F32
        const bf16View = new Uint16Array(rawBytes.buffer, rawBytes.byteOffset, elementCount);
        const f32 = new Float32Array(elementCount);
        for (let i = 0; i < elementCount; i++) {
          f32[i] = bf16ToF32(bf16View[i]);
        }
        return f32;
      }

      case GGMLType.Q8_0: {
        // Dequantize Q8_0 → F32 (block size 32, type size 34 bytes)
        const f32 = new Float32Array(elementCount);
        const numBlocks = elementCount / 32;
        const view = new DataView(rawBytes.buffer, rawBytes.byteOffset, byteSize);
        const int8View = new Int8Array(rawBytes.buffer, rawBytes.byteOffset, byteSize);

        for (let b = 0; b < numBlocks; b++) {
          const blockByteOffset = b * 34;
          const dRaw = view.getUint16(blockByteOffset, true);
          const scale = f16ToF32(dRaw);
          const outOffset = b * 32;
          const qsOffset = blockByteOffset + 2;

          for (let i = 0; i < 32; i++) {
            f32[outOffset + i] = scale * int8View[qsOffset + i];
          }
        }
        return f32;
      }

      case GGMLType.Q4_0: {
        // Dequantize Q4_0 → F32 (block size 32, type size 18 bytes)
        const f32 = new Float32Array(elementCount);
        const numBlocks = elementCount / 32;
        const view = new DataView(rawBytes.buffer, rawBytes.byteOffset, byteSize);

        for (let b = 0; b < numBlocks; b++) {
          const blockByteOffset = b * 18;
          const dRaw = view.getUint16(blockByteOffset, true);
          const scale = f16ToF32(dRaw);
          const outOffset = b * 32;
          const qsOffset = blockByteOffset + 2;

          for (let i = 0; i < 16; i++) {
            const byte = rawBytes[qsOffset + i];
            const q0 = (byte & 0x0f) - 8;
            const q1 = (byte >> 4) - 8;
            f32[outOffset + i] = scale * q0;
            f32[outOffset + i + 16] = scale * q1;
          }
        }
        return f32;
      }

      default:
        throw new Error(
          `Unsupported tensor type for F32 loading: ${GGMLType[info.type] ?? info.type}. ` +
          `Only F32, F16, and BF16 are supported in Phase 1.`
        );
    }
  }

  /** Close the underlying file handle. */
  close(): void {
    this.fileHandle.close();
  }
}
