# Neutrino linear read-out (neutrino-wunder-linear-v1)

* **Trainer:** `neutrino-wunder train` in the `neutrino` repository
  (`crates/neutrino-wunder`). It uses the `evoforge` genetic algorithm through
  `neutrino-optimizer::run_search` to pick the feature hyper-parameters
  (EMA spans, feature groups, sample-weight power, ridge strength) directly on
  the competition metric, and fits the linear weights in closed form
  (weighted ridge regression) for every candidate.
* **Inference:** `solution.py` + `model.json`. Pure NumPy, one thread,
  deterministic. See the docstring in `solution.py` for the exact recurrence.
* **Blend:** `blend.json` mixes the linear prediction per target with the
  organisers' stateful GRU baseline (`baseline.onnx`, from the public starter
  pack): `pred = (1 - w) * linear + w * gru`, w = 0.325 (t0) / 0.350 (t1).
  Full validation WP 0.6641 vs 0.6161 (linear) and 0.6171 (GRU).
* **Metrics:** stored in `model.json` under `metrics` (validation Global WP,
  including the part of the validation set the GA never saw).
