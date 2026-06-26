# Browser Growing NCA

A compact Growing Neural Cellular Automaton trained entirely in the browser with
[@jax-js/jax](https://github.com/ekzhang/jax-js). JAX-js initializes WebGPU when the browser
supports it and otherwise uses its Wasm backend. The NCA rollout is JIT-compiled; gradients and
Adam updates run locally in the selected backend.

## Run

```sh
npm install
npm run dev
```

Open the local URL printed by Vite. Use **Start training** to learn the procedural flower target.

## Verify

```sh
npm test
npm run build
```
