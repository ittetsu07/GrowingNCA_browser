import { describe, expect, it } from "vitest";
import { initializePreferredBackend, initializeWasm } from "../src/backend.js";

describe("initializeWasm", () => {
  it("initializes and selects only the Wasm backend", async () => {
    const initialized = [];
    const selected = [];

    const device = await initializeWasm({
      init: async (...devices) => {
        initialized.push(devices);
        return ["cpu", "wasm"];
      },
      defaultDevice: (value) => selected.push(value),
    });

    expect(device).toBe("wasm");
    expect(initialized).toEqual([["wasm"]]);
    expect(selected).toEqual(["wasm"]);
  });
});

describe("initializePreferredBackend", () => {
  it("selects WebGPU when it initializes before the timeout", async () => {
    const selected = [];
    const device = await initializePreferredBackend({
      init: async () => ["cpu", "wasm", "webgpu"],
      defaultDevice: (value) => selected.push(value),
      timeoutMs: 1,
    });

    expect(device).toBe("webgpu");
    expect(selected).toEqual(["webgpu"]);
  });

  it("falls back to Wasm when WebGPU is unavailable", async () => {
    const selected = [];
    const device = await initializePreferredBackend({
      init: async () => ["cpu", "wasm"],
      defaultDevice: (value) => selected.push(value),
    });

    expect(device).toBe("wasm");
    expect(selected).toEqual(["wasm"]);
  });
});
