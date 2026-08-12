/**
 * Runtime abstraction — decouples the engine from Node.js APIs.
 *
 * The math core of the engine must depend on NO runtime.
 * Only I/O and system operations go through this interface.
 */

/**
 * Handle to an open file for streaming reads.
 */
export interface FileHandle {
  /**
   * Read bytes from the file into a buffer.
   * @param buffer  Destination buffer
   * @param offset  Offset in buffer to start writing
   * @param length  Number of bytes to read
   * @param position  Position in file to read from
   * @returns Number of bytes actually read
   */
  read(buffer: Uint8Array, offset: number, length: number, position: number): number;

  /** Close the file handle. */
  close(): void;
}

/**
 * Minimal runtime interface for system operations.
 */
export interface Runtime {
  /** Read an entire file into memory. */
  readFile(path: string): Uint8Array;

  /** Open a file for streaming/random-access reads. */
  openFile(path: string): FileHandle;

  /** High-resolution timestamp in milliseconds. */
  now(): number;

  /** Exit the process. */
  exit(code: number): never;
}
