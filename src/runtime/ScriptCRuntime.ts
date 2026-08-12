/**
 * ScriptC implementation of the Runtime interface.
 *
 * ScriptC aims for Node.js API compatibility, so this initially
 * mirrors NodeRuntime. As ScriptC's capabilities are better understood,
 * this implementation will diverge where needed.
 *
 * TODO: Investigate ScriptC-specific optimizations:
 * - Direct memory mapping via FFI
 * - SIMD intrinsics
 * - Native threading primitives
 */
import * as fs from "node:fs";
import { performance } from "node:perf_hooks";
import type { Runtime, FileHandle } from "./Runtime.js";

class ScriptCFileHandle implements FileHandle {
  private fd: number;

  constructor(path: string) {
    this.fd = fs.openSync(path, "r");
  }

  read(buffer: Uint8Array, offset: number, length: number, position: number): number {
    return fs.readSync(this.fd, buffer, offset, length, position);
  }

  close(): void {
    fs.closeSync(this.fd);
  }
}

export class ScriptCRuntime implements Runtime {
  readFile(path: string): Uint8Array {
    return new Uint8Array(fs.readFileSync(path));
  }

  openFile(path: string): FileHandle {
    return new ScriptCFileHandle(path);
  }

  now(): number {
    return performance.now();
  }

  exit(code: number): never {
    process.exit(code);
    throw new Error(`Process exited with code ${code}`);
  }
}
