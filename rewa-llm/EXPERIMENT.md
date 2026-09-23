# Stage-one experiment protocol

## Frozen question

At the same global sparsity over Transformer attention and MLP matrices, does
ReWA-AdamW retain lower TinyStories validation perplexity than dense AdamW and
AdamW with L1 regularization?

This stage evaluates optimization and unstructured sparsifiability. It does not
make claims about hardware speedup, 2:4 sparsity, larger pretrained models, or
scaling behavior.

## Controlled setup

- Model: 6-layer LLaMA-style causal decoder, approximately 14M parameters.
- Data: one frozen TinyStories tokenization and split.
- Eligible parameters: all attention and SwiGLU projection matrices.
- Excluded parameters: token embedding/tied LM head and RMSNorm weights.
- Budget: 20M training tokens per pilot run.
- Randomness: identical initialization seed and batch-sampling seed for each
  method being compared.
- Primary metric: validation perplexity after global pruning to fixed eligible
  sparsities of 30%, 50%, 70%, 80%, 90%, and 95%.

## Pilot configurations

1. Dense AdamW with learning rate in `{3e-4, 6e-4}` and weight decay `0.1`.
2. L1 with coefficient in `{1e-7, 1e-6, 1e-5}`, using the selected dense
   learning rate.
3. ReWA-AdamW with the canonical AdamW configuration `(K, M, epsilon) =
   (9, 2, 0)` and y-space weight decay in `{1e-4, 1e-1, 1}`.

This configuration obeys the method's documented admissibility condition
`0 <= M < K - 1` and matches its canonical AdamW launch recipe. In particular,
`(3, 2)` is excluded because it lies on the boundary rather than inside the
stated regime, and epsilon is exactly zero rather than a generic numerical
stability constant.

The decay grid keeps `1e-4` as the source-recipe anchor and adds `1e-1` and
`1` because this pilot has only 611 optimizer steps. With the frozen learning
rate schedule, `sum_t lr_t` is about `0.2007`, so the corresponding idealized
y-space decay factors are approximately `1.0000`, `0.9801`, and `0.8182`.
The smaller provisional values `1e-3` and `1e-2` were removed after the first
anchor run showed that their cumulative effect would be negligible.

Any nonzero ReWA epsilon is treated as a separate algorithmic configuration,
not as a silent numerical-stability adjustment.

## Evidence recorded

- Training and validation loss by tokens seen.
- Validation perplexity before pruning.
- Eligible and whole-model sparsity after pruning.
- Per-layer attention and MLP sparsity.
- Peak GPU memory, throughput, runtime, non-finite failures, and skipped
  float16 optimizer steps reported by the gradient scaler.
- Exact configuration, seed, data hashes, and Git revision for every run.

## Stage-one decision rule

The evidence supports scale-up only if the same ReWA run remains within 5% of
dense validation perplexity before pruning and beats both Dense and L1 at both
70% and 80% eligible global sparsity. A negative or unstable result is retained
with its configuration and diagnostics.

## High-learning-rate extension

The first search placed every surviving geometry at its `3e-3` learning-rate
boundary. The paper's stagnation analysis gives a mechanism for extending that
boundary: with
`C = (1 - lambda * eta) / (eta * B)`, `C > 1` makes the stagnation region cover
the unit ball, so a base step size that is too small relative to the gradient
bound cannot cross the zero neighborhood. ReWA updates the extended `y`
coordinate, where `x = sign(y) |y|^K`; its base learning rate therefore should
not be inherited unchanged from AdamW in `x` coordinates.

The coordinate map also gives a concrete scale check. Around the model's
roughly `|x| = 0.02` initialization, `dx/dy = K |y|^(K-1)` is about `0.22` for
`K=3` and `0.28` for `K=9`. A first-order y-space step that matches an x-space
AdamW learning rate of `6e-4` is therefore already about `2e-3` to `3e-3`.
The previous optimum at `3e-3` is consistent with this coordinate extension
and motivates searching above it.

The numerical stability probe bracketed the source recipe with learning rates
up to `0.192`. Both K=3/M=0 and K=9/M=2 complete at `0.024`, while `0.048`
becomes non-finite within 32 optimizer steps. The 3M-token screen therefore
uses the refined stable-side grid `{6e-3, 1.2e-2, 2.4e-2, 3.6e-2}`. It
evaluates 70% and 80% sparsity during the
3M- and 10M-token stages, retains the canonical AdamW geometry `(K, M) =
(9, 2)` through the short-run selection, and selects the sparse finalist by
mean 70%/80% validation loss. Dense and L1 receive their own learning-rate
controls before the final method comparison. A factor-of-two refinement is run
around the best stable bracket point.

The high-LR runner uses a zero cosine floor (`min_lr_ratio=0`), matching the
source scheduler and the paper's early-large/late-small mechanism: large early
steps cross the zero neighborhood, while late steps settle rather than remain
at 10% of a very large peak LR.

Large LR changes the cumulative y-space decay because the update contains
`1 - lr_t * weight_decay`. The canonical raw `weight_decay=1e-4` remains in
the geometry screen. Subsequent decay variants are specified at reference
`lr=3e-3` and scaled inversely with candidate LR, preserving the first-order
quantity `weight_decay * sum_t lr_t` when the scheduler shape is fixed.
