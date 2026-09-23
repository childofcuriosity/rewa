# Local high-LR stability probe

The 14M-parameter model ran on a GTX 1650 Ti with float16 autocast, batch size
1, four-way gradient accumulation, 32 optimizer steps, four warmup steps, and
a cosine schedule ending at zero. All runs used the frozen 100k/5k TinyStories
corpus and raw y-space weight decay `1e-4`.

K=3/M=0 and K=9/M=2 both completed at peak LR `0.024` without GradScaler
skips. Peak LR `0.048` became non-finite at iterations 23 and 14 respectively;
K=3/M=0 at `0.192` became non-finite at iteration 6 after one skipped optimizer
step. These observations place the short-run numerical boundary between
`0.036` and `0.048` for K=3/M=0 and between `0.024` and `0.048` for K=9/M=2.

The follow-up 3M-token screen uses `0.006`, `0.012`, `0.024`, and `0.036` and
evaluates 70%/80% pruning. The 32-step validation losses in
`stability-smoke.csv` measure early stability, not final model quality.
