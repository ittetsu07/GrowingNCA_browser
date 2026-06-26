import { jit, numpy as np, tree, valueAndGrad } from "@jax-js/jax";
import { adam, applyUpdates } from "@jax-js/optax";

export const CHANNELS = 16;
const HIDDEN = 64;
const STEPS = 32;

function seededValues(length, seed, scale = 0.1) {
  let value = seed >>> 0;
  const values = new Float32Array(length);
  for (let i = 0; i < length; i += 1) {
    value = (value * 1664525 + 1013904223) >>> 0;
    values[i] = ((value / 0xffffffff) * 2 - 1) * scale;
  }
  return values;
}

export function createSeed(size, channels = CHANNELS) {
  const values = new Float32Array(size * size * channels);
  values[((Math.floor(size / 2) * size + Math.floor(size / 2)) * channels) + 3] = 1;
  return np.array(values).reshape([1, size, size, channels]);
}

export function createTarget(size) {
  const pixels = new Float32Array(size * size * 4);
  const center = (size - 1) / 2;
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const dx = (x - center) / center;
      const dy = (y - center) / center;
      const radius = Math.hypot(dx, dy);
      const angle = Math.atan2(dy, dx);
      const petal = Math.max(0, 1 - Math.abs(radius - (0.43 + 0.15 * Math.cos(angle * 6))) * 8);
      const core = Math.max(0, 1 - radius * 7);
      const leaf = Math.max(0, 1 - Math.hypot(dx * 1.8, (dy + 0.47) * 3) * 3);
      const index = (y * size + x) * 4;
      pixels[index] = 0.93 * petal + 0.98 * core;
      pixels[index + 1] = 0.22 * petal + 0.74 * core + 0.55 * leaf;
      pixels[index + 2] = 0.45 * petal + 0.1 * core + 0.22 * leaf;
      pixels[index + 3] = Math.max(petal, core, leaf);
    }
  }
  return np.array(pixels).reshape([1, size, size, 4]);
}

export function readPixels(array) {
  return array.ref.data();
}

export function createModel(channels = CHANNELS, hidden = HIDDEN, seed = 7) {
  return {
    w1: np.array(seededValues(channels * 4 * hidden, seed)).reshape([channels * 4, hidden]),
    b1: np.zeros([hidden]),
    w2: np.array(seededValues(hidden * channels, seed + 1, 0.01)).reshape([hidden, channels]),
    b2: np.zeros([channels]),
  };
}

function perceive(state) {
  const north = np.roll(state.ref, -1, 1);
  const south = np.roll(state.ref, 1, 1);
  const west = np.roll(state.ref, -1, 2);
  const east = np.roll(state.ref, 1, 2);
  const sobelX = east.ref.sub(west.ref);
  const sobelY = south.ref.sub(north.ref);
  const laplacian = north.add(south).add(east).sub(west.ref.mul(4));
  return np.concatenate([state.ref, sobelX, sobelY, laplacian], 3);
}

export function step(params, state) {
  const [batch, height, width] = state.shape;
  const features = perceive(state).reshape([batch * height * width, params.w1.shape[0]]);
  const hidden = np.tanh(np.matmul(features, params.w1.ref).add(params.b1));
  const updates = np.matmul(hidden, params.w2.ref).add(params.b2).reshape([batch, height, width, params.b2.shape[0]]);
  return updates.mul(0.1).add(state);
}

export const rollout = jit((params, initialState) => {
  let state = initialState;
  for (let i = 0; i < STEPS; i += 1) state = step(tree.ref(params), state);
  return state;
});

export function loss(params, initialState, target) {
  const result = rollout(params, initialState);
  const image = result.slice([], [], [], [0, 4]);
  return np.square(image.sub(target)).mean();
}

export function createTrainer(model, learningRate = 0.003) {
  const optimizer = adam(learningRate);
  return { optimizer, optimizerState: optimizer.init(tree.ref(model)) };
}

export function trainStep(model, trainer, initialState, target) {
  const [value, gradients] = valueAndGrad(loss)(tree.ref(model), initialState, target.ref);
  const [updates, optimizerState] = trainer.optimizer.update(
    gradients,
    trainer.optimizerState,
    tree.ref(model),
  );
  return {
    model: applyUpdates(model, updates),
    trainer: { ...trainer, optimizerState },
    loss: value,
  };
}
