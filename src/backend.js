export async function initializeBackend(preference, { init, defaultDevice }) {
  if (preference === "webgpu") {
    const devices = await init("webgpu");
    if (devices.includes("webgpu")) {
      defaultDevice("webgpu");
      return "webgpu";
    }
  } else {
    await init("wasm");
  }

  defaultDevice("wasm");
  return "wasm";
}
