# Neutrino solution (step 3): MLP read-out on GA-selected streaming features, blended with the GRU baseline

* **Features:** `model.json` is the `neutrino-wunder` GA champion
  (`crates/neutrino-wunder` in the neutrino repository): its `layout` defines
  the causal streaming transform (raw features, fast-EMA residuals, 42-row
  lagged differences; 336 values per row). The GA (`evoforge` through
  `neutrino-optimizer::run_search`) chose EMA spans, feature groups, lag,
  sample-weight power and ridge strength directly on the competition metric.
* **Read-out:** `mlp.onnx`, a 336 → 96 → 32 → 2 ReLU MLP trained by
  `scripts/train_mlp.py` on `neutrino-wunder features` exports (all 1,000
  training sequences of the subset, every 8th row) and exported by
  `scripts/export_mlp_onnx.py`. `mlp_train.json` records the run.
* **Blend:** `blend.json` mixes the MLP with the organisers' stateful GRU
  baseline (`baseline.onnx`, public starter pack): `pred = 0.775·mlp +
  0.225·gru` (t0) and `0.75·mlp + 0.25·gru` (t1), weights chosen on the
  first half of the validation sequences.
* **Inference:** `solution.py`, NumPy + ONNX Runtime, one thread, two
  pre-bound sessions, deterministic. ≈ 50–55 µs/row on the development
  machine (≈ 33–36 min for the 39.4 M-row test set).
* **Validation (organisers' scorer, all 1,873 sequences):** 0.6732 WP;
  0.6740 on the half whose sequences were not used for the blend weights.
  MLP alone 0.6497, linear read-out alone 0.6331, GRU alone 0.6171.
