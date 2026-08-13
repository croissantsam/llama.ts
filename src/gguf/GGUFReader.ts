/**
 * GGUF binary file reader.
 *
 * Parses GGUF format files using DataView for endian-safe reads.
 * Based on the spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
 */
import type { Runtime } from "../runtime/Runtime.js";
import {
  GGUF_MAGIC,
  GGUF_DEFAULT_ALIGNMENT,
  GGUFMetadataValueType,
  type GGUFHeader,
  type GGUFTensorInfo,
  type GGUFFile,
  type GGUFValue,
  type GGMLType,
} from "./GGUFTypes.js";

/**
 * Cursor-based binary reader wrapping a DataView.
 * All reads advance the cursor position.
 */
class BinaryReader {
  private view: DataView;
  private pos: number;

  /**
   * Accept either an ArrayBuffer (or ArrayBufferLike) or a Uint8Array so
   * that callers that have a subarray (with non-zero byteOffset) are handled
   * correctly. This prevents DataView/TypedArray range errors when the
   * underlying ArrayBuffer is larger than the intended slice.
   */
  constructor(buffer: ArrayBufferLike | Uint8Array) {
    if (buffer instanceof Uint8Array) {
      this.view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
    } else {
      this.view = new DataView(buffer as ArrayBuffer);
    }
    this.pos = 0;
  }

  get position(): number {
    return this.pos;
  }

  set position(p: number) {
    this.pos = p;
  }

  readUint8(): number {
    const val = this.view.getUint8(this.pos);
    this.pos += 1;
    return val;
  }

  readInt8(): number {
    const val = this.view.getInt8(this.pos);
    this.pos += 1;
    return val;
  }

  readUint16(): number {
    const val = this.view.getUint16(this.pos, true); // little-endian
    this.pos += 2;
    return val;
  }

  readInt16(): number {
    const val = this.view.getInt16(this.pos, true);
    this.pos += 2;
    return val;
  }

  readUint32(): number {
    const val = this.view.getUint32(this.pos, true);
    this.pos += 4;
    return val;
  }

  readInt32(): number {
    const val = this.view.getInt32(this.pos, true);
    this.pos += 4;
    return val;
  }

  readUint64(): bigint {
    const val = this.view.getBigUint64(this.pos, true);
    this.pos += 8;
    return val;
  }

  readInt64(): bigint {
    const val = this.view.getBigInt64(this.pos, true);
    this.pos += 8;
    return val;
  }

  readFloat32(): number {
    const val = this.view.getFloat32(this.pos, true);
    this.pos += 4;
    return val;
  }

  readFloat64(): number {
    const val = this.view.getFloat64(this.pos, true);
    this.pos += 8;
    return val;
  }

  readBool(): boolean {
    const val = this.readUint8();
    return val !== 0;
  }

  /**
   * Read a GGUF string: uint64 length prefix + UTF-8 bytes (no null terminator).
   */
  readString(): string {
    const length = Number(this.readUint64());
    const bytes = new Uint8Array(this.view.buffer as ArrayBuffer, this.view.byteOffset + this.pos, length);
    this.pos += length;
    return new TextDecoder("utf-8").decode(bytes);
  }

  /**
   * Read a metadata value based on its type tag.
   */
  readMetadataValue(type: GGUFMetadataValueType): GGUFValue {
    switch (type) {
      case GGUFMetadataValueType.UINT8:   return this.readUint8();
      case GGUFMetadataValueType.INT8:    return this.readInt8();
      case GGUFMetadataValueType.UINT16:  return this.readUint16();
      case GGUFMetadataValueType.INT16:   return this.readInt16();
      case GGUFMetadataValueType.UINT32:  return this.readUint32();
      case GGUFMetadataValueType.INT32:   return this.readInt32();
      case GGUFMetadataValueType.FLOAT32: return this.readFloat32();
      case GGUFMetadataValueType.BOOL:    return this.readBool();
      case GGUFMetadataValueType.STRING:  return this.readString();
      case GGUFMetadataValueType.UINT64:  return this.readUint64();
      case GGUFMetadataValueType.INT64:   return this.readInt64();
      case GGUFMetadataValueType.FLOAT64: return this.readFloat64();
      case GGUFMetadataValueType.ARRAY: {
        const elemType = this.readUint32() as GGUFMetadataValueType;
        const count = Number(this.readUint64());
        const arr: GGUFValue[] = new Array(count);
        for (let i = 0; i < count; i++) {
          arr[i] = this.readMetadataValue(elemType);
        }
        return arr;
      }
      default:
        throw new Error(`Unknown metadata value type: ${type}`);
    }
  }
}

/**
 * Read and parse a GGUF file.
 *
 * This reads the entire file into memory to parse header, metadata,
 * and tensor info. Tensor DATA is not loaded here — use TensorLoader
 * for lazy/on-demand tensor loading.
 */
export function readGGUF(path: string, runtime: Runtime): GGUFFile {
  const fileData = runtime.readFile(path);
  const reader = new BinaryReader(fileData.buffer);

  // ─── Header ──────────────────────────────────────────────
  const magic = reader.readUint32();
  if (magic !== GGUF_MAGIC) {
    throw new Error(
      `Invalid GGUF magic: expected 0x${GGUF_MAGIC.toString(16)}, got 0x${magic.toString(16)}`
    );
  }

  const version = reader.readUint32();
  if (version < 2 || version > 3) {
    throw new Error(`Unsupported GGUF version: ${version} (expected 2 or 3)`);
  }

  const tensorCount = reader.readUint64();
  const metadataCount = reader.readUint64();

  const header: GGUFHeader = { magic, version, tensorCount, metadataCount };

  // ─── Metadata KV pairs ───────────────────────────────────
  const metadata = new Map<string, GGUFValue>();
  const metaCount = Number(metadataCount);

  for (let i = 0; i < metaCount; i++) {
    const key = reader.readString();
    const valueType = reader.readUint32() as GGUFMetadataValueType;
    const value = reader.readMetadataValue(valueType);
    metadata.set(key, value);
  }

  // ─── Tensor info entries ─────────────────────────────────
  const tensCount = Number(tensorCount);
  const tensors: GGUFTensorInfo[] = new Array(tensCount);

  for (let i = 0; i < tensCount; i++) {
    const name = reader.readString();
    const nDimensions = reader.readUint32();
    const dimensions: number[] = new Array(nDimensions);
    for (let d = 0; d < nDimensions; d++) {
      dimensions[d] = Number(reader.readUint64());
    }
    const type = reader.readUint32() as GGMLType;
    const offset = reader.readUint64();

    tensors[i] = { name, nDimensions, dimensions, type, offset };
  }

  // ─── Compute tensor data start offset ────────────────────
  // The tensor data block is aligned to the specified alignment.
  const alignment = Number(metadata.get("general.alignment") ?? GGUF_DEFAULT_ALIGNMENT);
  const currentPos = reader.position;
  const dataOffset = Math.ceil(currentPos / alignment) * alignment;

  return { header, metadata, tensors, dataOffset };
}
