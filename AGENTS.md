# PFM PushT release

Keep only standard joint-NLL PFM for native image-conditioned PushT. Preserve two
96x96 RGB/agent-position frames, crop84, trained encoder, horizon16, and executed
action slice 1:9. Keep the current U-Net, normalization, Gaussian NLL, sigmoid
log-variance bounds, corrected SDE, EMA and checkpoint semantics.

Use CPU validation by default (`pytest -q`). Keep source/data/outputs separate;
never commit datasets or checkpoints. Keep LICENSE and third-party notices.
Update VERIFICATION.md when the validated release behavior or entrypoints change.
GPU, full training, and paper metric reproduction require separate evidence.
