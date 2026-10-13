# Posterior Flow Matching — PushT

Native image-conditioned PushT implementation of Posterior Flow Matching (PFM).

[Project page](https://merry7cherry.github.io/posterior-flow-matching/) · [Paper (PDF)](https://merry7cherry.github.io/posterior-flow-matching/assets/paper.pdf) · [Images / project hub](https://github.com/merry7cherry/posterior-flow-matching) · [Point clouds](https://github.com/merry7cherry/posterior-flow-matching-pointcloud) · [Molecules](https://github.com/merry7cherry/posterior-flow-matching-molecules)

This release contains standard joint Gaussian NLL training, the native trainable
image encoder and action U-Net, ODE/SDE sampling, environment evaluation, video,
and posterior diagnostics. No trained weights or dataset are redistributed.

## Install

Use Python 3.10 and [uv](https://docs.astral.sh/uv/). The pinned legacy Gym/Pymunk
versions preserve the native simulator. On macOS/Linux:

```bash
uv venv --python 3.10
# Linux CPU-only PyTorch wheels (skip this line on macOS):
uv pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install --excludes requirements-exclude.txt -e '.[test]'
source .venv/bin/activate
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy OMP_NUM_THREADS=2
pytest -q
```

The exclusion removes only robomimic's EGL device-discovery package, which is
unused by the native pygame PushT simulator. No robot simulator, W&B login,
cluster account or Git checkout is required at runtime. The public commands also
work from a built wheel (`uv build --wheel`, then install `dist/*.whl` with the
same exclusion). Use a separate environment from the other PFM tasks.

## Data

Download the original [PushT demonstrations](https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip):

```bash
mkdir -p data
curl -fL https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip -o data/pusht.zip
python -c "import hashlib,pathlib; p=pathlib.Path('data/pusht.zip'); assert hashlib.sha256(p.read_bytes()).hexdigest() == '63d52a114a3f010861f0181309d165b7d69133ccae426ece2fc94caed147bdf9'"
python -m zipfile -e data/pusht.zip data
```

The expected directory is `data/pusht/pusht_cchi_v7_replay.zarr`. It has 206
episodes and 25,650 frames. The native seeded mask caps training at 90 episodes;
validation uses the complementary episodes. Normalization follows the upstream
loader and is stored in checkpoints. Pass an absolute data path when invoking
installed commands from another directory. Dataset access is subject to the
publisher's terms; data is not included in this repository.

## Train and resume

```bash
pfm-pusht-train task.dataset.zarr_path=/absolute/path/pusht_cchi_v7_replay.zarr \
  hydra.run.dir=outputs/pfm
```

`pusht.yaml` provides the full network, batch64, AdamW LR1e-4, cosine decay,
500 warmup updates, 3050 epochs, and per-update EMA. It defaults to CPU/FP32 and
one rollout process. Full training is expensive; the CPU release checks do not
run it. A small end-to-end smoke configuration retains the native image encoder,
data handling and action interface while reducing U-Net width, batch size, and
training/rollout steps:

```bash
pfm-pusht-train --config-name=smoke \
  task.dataset.zarr_path=/absolute/path/pusht_cchi_v7_replay.zarr \
  hydra.run.dir=outputs/smoke
```

Ordinary interruption recovery uses the same configuration and output directory:

```bash
pfm-pusht-train --config-name=smoke \
  task.dataset.zarr_path=/absolute/path/pusht_cchi_v7_replay.zarr \
  hydra.run.dir=outputs/smoke training.resume=true
```

Checkpoints include model, EMA, optimizer, scheduler, epoch/update counters and
Python/NumPy/PyTorch random states. Recovery resumes after the last fully saved
epoch; an interrupted partial epoch is repeated. It preserves the original
training horizon and rejects changed model/optimizer/schedule settings. Writes
are atomic. Use a fresh output directory for a new run. Local scalar logs are
`logs.json.txt`; complete checkpoints are `checkpoints/latest.ckpt` and
`checkpoints/final.ckpt`. Only load checkpoints you trust: the inherited format
uses dill. No historical staged-training workflow is included.

## Sample, evaluate and visualize

One saved model supplies both samplers. Native scoring is mean per-episode
maximum normalized block-overlap reward, not binary success rate:

```bash
pfm-pusht-evaluate --checkpoint outputs/pfm/checkpoints/final.ckpt \
  --output outputs/ode20 --solver ode --steps 20
pfm-pusht-evaluate --checkpoint outputs/pfm/checkpoints/final.ckpt \
  --output outputs/sde50 --solver sde --steps 50 --diffusion-scale 0.1
```

Use `--steps 50` or `--steps 20` with either solver. Any positive integration
step count is supported. Evaluation defaults to 50 held-out simulator starts,
CPU, a single environment process, and at most 300 environment steps. Specify
`--n-envs` to increase parallelism. Every run writes `evaluation.json` plus one
native rollout video under `media/`. For a quick visualization of your smoke
checkpoint, add `--n-test 1 --max-steps 8 --steps 2`; this is an execution check,
not meaningful task performance.

Add `--data /absolute/path/pusht_cchi_v7_replay.zarr --diagnostic-batches 2` to
write `diagnostics.json`. It compares predicted coordinate velocity variance
with squared velocity residual on held-out reference interpolations at
t=0.1,0.5,0.9,0.99 and reports Gaussian NLL without its additive constant.
The endpoint variance is `(1-t)^2 Gamma_v`; these are normalized **action**
readouts, not object-position confidence regions or a calibration guarantee.

## Method and scope

- Two 96x96 RGB/agent-position frames; native crop84, ResNet image encoder with
  GroupNorm and continuous-time conditional U-Net.
- Predict a `[B,16,2]` action block and execute `action_pred[:,1:9]`.
- Train both velocity mean and variance with joint coordinate Gaussian NLL.
  Sigmoid log-variance bounds are `[-3,3]`, initialized at zero.
- Independent unit Gaussian source, time in `[0.001,0.999]`, float time embedding
  scaled by 1000, and global gradient-norm clipping at 1.
- Euler ODE or corrected Itô Euler–Maruyama SDE, using
  `lambda(t)=lambda0*(1-t)`. The SDE drift includes covariance divergence and
  covariance-weighted score; four Rademacher VJP probes estimate the diagonal
  variance derivative. Sampling stops at 0.99 and uses the endpoint mean readout.
- EMA weights are used when enabled. No A/B variants, diffusion baseline,
  non-PushT task, physical-robot deployment, or historical experiment pipeline.

The inherited `diffusion_policy` module and network parameter names preserve
standard-PFM state loading. Former variant checkpoints are explicitly rejected.
Existing compatible standard-PFM checkpoints can be evaluated; this release's
new trainer checkpoints are the supported interruption-recovery format.

[VERIFICATION.md](VERIFICATION.md) records CPU evidence and its limits. Neither
default commands nor the reduced smoke model constitute a reproduction of the
paper's trained weights, CUDA behavior, speed or reported task metrics.

## Citation and license

Please cite *Posterior Flow Matching* (Ma et al., 2026) using `CITATION.cff`, and
[Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) for the upstream
policy architecture, demonstration pipeline and simulator. The paper link
currently provides the project-hosted PDF; no unverified arXiv ID is inserted.
PFM additions are MIT; inherited components also carry Apache-2.0 notices. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
