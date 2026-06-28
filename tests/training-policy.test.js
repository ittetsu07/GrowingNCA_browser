import { describe, expect, it } from "vitest";
import { batchSizeForBackend } from "../src/training-policy.js";

describe("batchSizeForBackend", () => {
  it("batches more optimizer updates on WebGPU", () => {
    expect(batchSizeForBackend("webgpu")).toBe(8);
  });

  it("keeps Wasm batches small enough to preserve responsiveness", () => {
    expect(batchSizeForBackend("wasm")).toBe(2);
  });
});
