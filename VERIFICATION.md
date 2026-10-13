# CPU verification

Release scope: standard PFM for native image-conditioned PushT. Verification was
performed on macOS arm64 with Python3.10.20, PyTorch2.7.1 and two CPU threads.
The `cpu.yml` workflow repeats the portable tests and wheel checks on Linux.

## Verified locally

- Fresh isolated environment from the declared dependencies, without EGL or
  diffusers. Package build and wheel installation; CLI help and native SDE
  ODE/SDE 20/50-step rollouts and diagnostics from outside the source tree and without `.git`.
- 13 focused tests: joint Gaussian NLL and gradients; bounded variance chain
  rule; stochastic/exact diagonal input derivatives; Gaussian SDE moments;
  ODE/zero-noise equivalence; native model action slicing and both-head/encoder
  training gradients; exact interrupted-versus-uninterrupted model/EMA recovery;
  asynchronous simulator parity with and without shared memory; native videos.
- Native-width `[512,1024,2048]` U-Net plus original image encoder: 262,710,070
  parameters, CPU batch1 forward/backward, finite joint NLL and nonzero gradients
  in the encoder, velocity head and variance head. Input was synthetic; this
  validates tensor interfaces and differentiation, not learned task quality.
- A reduced-width model completed two one-batch training epochs on the original
  local 206-episode, 25,650-frame dataset, with validation, actual native physics
  rollouts, videos and checkpoint saves. Tests otherwise create synthetic
  native-format data explicitly labeled as such. Exact interruption/resume was
  also checked with real local data (one test passed). All four wheel rollout
  videos were decoded and checked for nonempty frame sequences.
- Fixed-seed source-versus-release comparison is bitwise identical for joint
  NLL, variance-head and encoder gradients, ODE/SDE actions, endpoint variance,
  native simulator RGB output and reward. That comparison uses the reduced-width
  native model and source revision `525db84ca8dfbdd67e7fe5649d7f38acdc23d3f7`.

## Re-run

```bash
pytest -q
# Optional real-data interruption/resume test:
PFM_TEST_DATA=/absolute/path/pusht_cchi_v7_replay.zarr pytest -q tests/test_resume.py
uv build --wheel
```

See README for the two-epoch CPU smoke run and actual rollout/video commands.
The resume unit test substitutes a no-op runner to isolate random-state and
weight restoration; real simulator execution is checked separately.

## Evidence limits

No GPU, CUDA, BF16, DDP, full training, paper-metric replication, trained-weight
release or performance benchmark was performed. Exact parity does not establish
posterior calibration. The new CPU smoke run is an implementation test, not a
new scientific result. Gym/torchvision emit upstream deprecation notices; their
versions are deliberately pinned to preserve native interfaces and physics.
