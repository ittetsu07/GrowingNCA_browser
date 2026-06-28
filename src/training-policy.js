export function batchSizeForBackend(backend) {
  return backend === "webgpu" ? 8 : 2;
}
