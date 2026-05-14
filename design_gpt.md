# Draft Design Section

**Design.** Our design targets the decode stage of MoE serving, where GPU memory is too small to hold all expert weights and naive offloading quickly turns expert selection into a CPU-GPU bandwidth bottleneck. We therefore keep non-expert parameters on GPU, place expert weights in CPU memory, and expose a bounded GPU-resident expert cache shared across all MoE layers. Prefill uses a simple full-tensor materialization path; the main challenge is decode, where each layer touches only a small, rapidly changing subset of experts. The key systems question is thus not how to execute an MoE layer, but which cached experts should survive across layers and decode steps so that future misses are minimized.

Our key idea is to separate admission from eviction. When layer `l` at decode step `s` is about to execute, the runtime uses the routing plan available for that layer to determine the experts that must be materialized now. Those experts are admitted on demand and protected while the layer is being constructed. Prediction is used only for eviction: if the cache is full, the system chooses which previously cached expert from another layer to evict. This layer-local formulation is important because it does not require the cache to hold the union of all experts used anywhere in the full decode step. Instead, we only require that the current layer fit, and we use prediction to decide which non-current-layer residents are least valuable to keep.

To drive these decisions, we construct a step-level ranking over `(layer, expert)` pairs. The underlying predictor operates per request/token, but the cache needs a batch-level view that approximates the union of experts likely to be used by any live request. We obtain this by aggregating the per-request logits of the active requests with an elementwise max, applying a per-layer softmax, and then globally sorting all experts across all layers. Intuitively, an expert ranks high if any active request strongly prefers it, which matches the goal of preserving experts likely to appear in the batch union. However, raw predictor rank is not yet a reuse probability. We therefore learn held-out calibration curves

```text
g_h(r) = P(expert is actually active | predicted rank r)
```

for each prediction horizon `h`, and use these curves to convert rank into an empirical probability of future use.

Our deployed policy uses two horizons. A primary model predicts reuse at the current token (`h=0`), and a secondary model predicts reuse at the next token (`h=1`). For each cached expert `i` at step `s`, we look up its ranks `r_0` and `r_1`, map them to calibrated probabilities `p_0 = g_0(r_0)` and `p_1 = g_1(r_1)`, and assign the fused value

```text
S_i(s) = p_0 + lambda * (1 - p_0) * p_1
```

The first term captures immediate value, while the second captures short-term future value without double-counting experts that are already likely to be needed now; `lambda in [0,1]` discounts the noisier next-token signal. On a miss, the cache evicts the resident expert from another layer with the smallest `S_i(s)`. This gives a practical online approximation to Belady-style eviction: keep experts with the highest estimated near-future reuse, but do so using deployable predictions rather than exact future routing.

The runtime is engineered so that this policy adds little overhead. The GPU cache is slot-major and shared across all MoE layers; experts are copied from pinned CPU memory, or UVA-backed buffers when available, directly into cache slots. We rebuild the prediction-aware eviction index once per decode step and maintain one heap per cached layer, avoiding an `O(C)` scan of all cache slots on every miss. After loading, the MoE kernel consumes the cache through a compact path: instead of repacking active experts into a fresh dense tensor, we pass a direct view of the shared slot buffers together with an `expert_map` that points from expert IDs to physical cache slots. This removes an extra GPU-to-GPU compaction copy from the decode critical path. Overall, the design preserves the simplicity of a synchronous offloader while turning cache management from a future-blind heuristic into a prediction-enhanced, calibrated, and deployable policy for decode-time MoE offloading.
