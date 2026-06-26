import { describe, expect, it } from "vitest";
import { initializeBackend } from "../src/backend.js";

describe("initializeBackend", () => {
  it("starts the auto mode on Wasm without requesting WebGPU", async () => {
    const initialized = [];
    const selected = [];

    const device = await initializeBackend("auto", {
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
