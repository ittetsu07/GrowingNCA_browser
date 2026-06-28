export async function initializeWasm({ init, defaultDevice }) {
  await init("wasm");
  defaultDevice("wasm");
  return "wasm";
}

export async function initializePreferredBackend({ init, defaultDevice, timeoutMs = 2000 }) {
  try {
    const devices = await Promise.race([
      init("webgpu"),
      new Promise((resolve) => setTimeout(() => resolve([]), timeoutMs)),
    ]);
    if (devices.includes("webgpu")) {
      defaultDevice("webgpu");
      return "webgpu";
    }
  } catch {
    // WebGPU is optional. Fall through to the portable backend.
  }
  return initializeWasm({ init, defaultDevice });
}
