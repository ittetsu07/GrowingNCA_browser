import { defaultDevice, init, tree } from "@jax-js/jax";
import { initializePreferredBackend } from "./backend.js";
import { CHANNELS, createModel, createSeed, createTarget, createTrainer, readPixels, rollout, trainStep } from "./nca.js";
import { batchSizeForBackend } from "./training-policy.js";

const SIZE = 32;
const targetCanvas = document.querySelector("#target");
const targetPreview = document.querySelector("#target-preview");
const worldCanvas = document.querySelector("#world");
const status = document.querySelector("#status");
const steps = document.querySelector("#steps");
const lossValue = document.querySelector("#loss");
const runtime = document.querySelector("#runtime");
const throughput = document.querySelector("#throughput");
const toggle = document.querySelector("#toggle");
const stepButton = document.querySelector("#step");
const resetModel = document.querySelector("#reset-model");
const controls = [toggle, stepButton, resetModel];

let model;
let trainer;
let target;
let training = false;
let busy = false;
let iteration = 0;
let lastLoss = NaN;
let activeDevice = "wasm";
let updatesPerSecond = NaN;

function paint(canvas, values) {
  const context = canvas.getContext("2d");
  const image = context.createImageData(SIZE, SIZE);
  for (let pixel = 0; pixel < SIZE * SIZE; pixel += 1) {
    const i = pixel * 4;
    image.data[i] = Math.round(Math.max(0, Math.min(1, values[i])) * 255);
    image.data[i + 1] = Math.round(Math.max(0, Math.min(1, values[i + 1])) * 255);
    image.data[i + 2] = Math.round(Math.max(0, Math.min(1, values[i + 2])) * 255);
    image.data[i + 3] = Math.round(Math.max(0, Math.min(1, values[i + 3])) * 255);
  }
  context.putImageData(image, 0, 0);
}

async function renderTarget() {
  const pixels = await readPixels(target);
  paint(targetCanvas, pixels);
  paint(targetPreview, pixels);
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
  steps.textContent = String(iteration);
  lossValue.textContent = Number.isFinite(lastLoss) ? lastLoss.toFixed(5) : "—";
  runtime.textContent = activeDevice === "webgpu" ? "WebGPU" : "Wasm";
  throughput.textContent = Number.isFinite(updatesPerSecond) ? `${updatesPerSecond.toFixed(1)}/s` : "—";
}

function freshModel() {
  model = createModel();
  trainer = createTrainer(model);
  iteration = 0;
  lastLoss = NaN;
  updatesPerSecond = NaN;
}

async function trainBatch(force = false, requestedBatchSize = batchSizeForBackend(activeDevice)) {
  if ((!training && !force) || busy) return;
  busy = true;
  const startedAt = performance.now();
  let finalLoss = null;
  try {
    for (let update = 0; update < requestedBatchSize; update += 1) {
      const result = trainStep(model, trainer, createSeed(SIZE, CHANNELS), target.ref);
      model = result.model;
      trainer = result.trainer;
      finalLoss?.dispose();
      finalLoss = result.loss;
      iteration += 1;
    }
    lastLoss = Number((await readPixels(finalLoss))[0]);
    finalLoss.dispose();
    await renderWorld();
    const elapsedSeconds = (performance.now() - startedAt) / 1000;
    updatesPerSecond = requestedBatchSize / Math.max(elapsedSeconds, 0.001);
    updateStatus();
  } finally {
    busy = false;
  }
}

async function frame() {
  await trainBatch();
  requestAnimationFrame(frame);
}

toggle.addEventListener("click", () => {
  training = !training;
  toggle.textContent = training ? "Ⅱ" : "▶";
  toggle.setAttribute("aria-label", training ? "Pause training" : "Play training");
});
stepButton.addEventListener("click", () => trainBatch(true, 1));
resetModel.addEventListener("click", async () => {
  training = false;
  toggle.textContent = "▶";
  freshModel();
  await renderWorld();
  updateStatus();
});

try {
  activeDevice = await initializePreferredBackend({ init, defaultDevice });
  status.value = `${activeDevice.toUpperCase()} · preparing target…`;
  target = createTarget(SIZE);
  freshModel();
  await renderTarget();
  status.value = `${activeDevice.toUpperCase()} · compiling first NCA rollout…`;
  await new Promise(requestAnimationFrame);
  await renderWorld();
  controls.forEach((control) => { control.disabled = false; });
  updateStatus();
  requestAnimationFrame(frame);
} catch (error) {
  status.value = `Unable to initialize JAX-js: ${error instanceof Error ? error.message : String(error)}`;
}
