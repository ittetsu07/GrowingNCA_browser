import { describe, expect, it } from "vitest";
import { createModel, createSeed, createTarget, createTrainer, readPixels, trainStep } from "../src/nca.js";

describe("GrowingNCA inputs", () => {
  it("places an alive seed at the center of the state grid", () => {
    const seed = createSeed(5, 8);

    expect(seed.shape).toEqual([1, 5, 5, 8]);
    expect(seed.dataSync()[((2 * 5 + 2) * 8) + 3]).toBe(1);
  });

  it("creates an RGBA target with values in the display range", () => {
    const target = createTarget(8);
    const pixels = target.dataSync();

    expect(target.shape).toEqual([1, 8, 8, 4]);
    expect(Math.min(...pixels)).toBeGreaterThanOrEqual(0);
    expect(Math.max(...pixels)).toBeLessThanOrEqual(1);
  });

  it("reads display pixels without consuming a retained target", async () => {
    const target = createTarget(8);

    expect((await readPixels(target)).length).toBe(8 * 8 * 4);
    expect(target.refCount).toBe(1);
  });

  it("returns a finite loss and an updated model after one gradient step", () => {
    const model = createModel(8, 12, 4);
    const result = trainStep(model, createTrainer(model, 0.001), createSeed(8, 8), createTarget(8));

    expect(Number.isFinite(Number(result.loss.dataSync()[0]))).toBe(true);
    expect(result.model.w1.shape).toEqual([32, 12]);
  });
});
