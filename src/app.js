import { defaultDevice, init, tree } from "@jax-js/jax";
import { CHANNELS, createModel, createSeed, createTarget, createTrainer, readPixels, rollout, trainStep } from "./nca.js";

const SIZE = 32;
const targetCanvas = document.querySelector("#target");
const worldCanvas = document.querySelector("#world");
const status = document.querySelector("#status");
const toggle = document.querySelector("#toggle");
const resetModel = document.querySelector("#reset-model");
const resetState = document.querySelector("#reset-state");
const backend = document.querySelector("#backend");
const controls = [toggle, resetModel, resetState, backend];

let model;
let trainer;
let target;
let training = false;
let busy = false;
let iteration = 0;
let lastLoss = NaN;
let activeDevice = "wasm";

function paint(canvas, values) {
  const context = canvas.getContext("2d");
  const image = context.createImageData(SIZE, SIZE);
  for (let pixel = 0; pixel < SIZE * SIZE; pixel += 1) {
    const source = pixel * 4;
    const targetIndex = source;
    image.data[targetIndex] = Math.round(Math.max(0, Math.min(1, values[source])) * 255);
    image.data[targetIndex + 1] = Math.round(Math.max(0, Math.min(1, values[source + 1])) * 255);
    image.data[targetIndex + 2] = Math.round(Math.max(0, Math.min(1, values[source + 2])) * 255);
    image.data[targetIndex + 3] = Math.round(Math.max(0, Math.min(1, values[source + 3])) * 255);
  }
  context.putImageData(image, 0, 0);
}

async function renderTarget() {
  paint(targetCanvas, await readPixels(target));
}

async function renderWorld() {
  const state = rollout(tree.ref(model), createSeed(SIZE, CHANNELS));
  const visible = state.slice([], [], [], [0, 4]);
  paint(worldCanvas, await readPixels(visible));
  visible.dispose();
}

function updateStatus() {
  const loss = Number.isFinite(lastLoss) ? ` · loss ${lastLoss.toFixed(5)}` : "";
  status.value = `${activeDevice.toUpperCase()} · ${iteration} gradient steps${loss}`;
}

async function chooseBackend(preference) {
  const devices = await init();
  activeDevice = preference === "webgpu" && devices.includes("webgpu") ? "webgpu" : "wasm";
  if (preference === "auto") activeDevice = devices.includes("webgpu") ? "webgpu" : "wasm";
  if (!devices.includes(activeDevice)) activeDevice = devices[0] ?? "cpu";
  defaultDevice(activeDevice);
}

function freshModel() {
  model = createModel();
  trainer = createTrainer(model);
  iteration = 0;
  lastLoss = NaN;
}

async function trainOnce() {
  if (!training || busy) return;
  busy = true;
  const result = trainStep(model, trainer, createSeed(SIZE, CHANNELS), target.ref);
  model = result.model;
  trainer = result.trainer;
  lastLoss = Number((await readPixels(result.loss))[0]);
  result.loss.dispose();
  iteration += 1;
  if (iteration % 2 === 0) await renderWorld();
  updateStatus();
  busy = false;
}

async function frame() {
  await trainOnce();
  requestAnimationFrame(frame);
}

toggle.addEventListener("click", () => {
  training = !training;
  toggle.textContent = training ? "Pause training" : "Start training";
});

resetModel.addEventListener("click", async () => {
  training = false;
  toggle.textContent = "Start training";
  freshModel();
  await renderWorld();
  updateStatus();
});

resetState.addEventListener("click", async () => {
  await renderWorld();
});

backend.addEventListener("change", async () => {
  training = false;
  toggle.textContent = "Start training";
  await chooseBackend(backend.value);
  freshModel();
  await renderTarget();
  await renderWorld();
  updateStatus();
});

try {
  await chooseBackend("auto");
  target = createTarget(SIZE);
  freshModel();
  await renderTarget();
  await renderWorld();
  controls.forEach((control) => { control.disabled = false; });
  updateStatus();
  requestAnimationFrame(frame);
} catch (error) {
  status.value = `Unable to initialize JAX-js: ${error instanceof Error ? error.message : String(error)}`;
}
