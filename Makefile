# Wunder Fund "Alpha Connectome" hackathon workspace.
PY ?= .venv/bin/python
VALID ?= datasets/valid.parquet
SEQ ?= 50

.PHONY: help setup fetch-small fetch-valid fetch-train-subset fetch-train list-archive \
        score-baseline score score-full train parity package check docker-scorer test-rust clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

setup: ## create .venv, install requirements, fetch starter pack docs + baseline
	scripts/setup_env.sh

fetch-small: ## starter pack docs/baseline/utils only (a few MB)
	$(PY) scripts/fetch_starterpack.py --what small

fetch-valid: ## datasets/valid.parquet + valid_mask.parquet (5.6 GB)
	$(PY) scripts/fetch_starterpack.py --what valid

fetch-train-subset: ## first N training sequences (streams 28 GB, keeps ~3 GB) N=$(N)
	$(PY) scripts/fetch_starterpack.py --what train-subset --train-sequences $(or $(N),1000)

fetch-train: ## the full 28 GB datasets/train.parquet
	$(PY) scripts/fetch_starterpack.py --what train

list-archive: ## list members of the official archive without downloading it
	$(PY) scripts/fetch_starterpack.py --list

score-baseline: ## score the organisers' GRU baseline on SEQ validation sequences
	$(PY) scripts/score_solution.py --solution starterpack/baseline --validation $(VALID) --sequences $(SEQ)

score: ## score solution/ on SEQ validation sequences (with timing)
	$(PY) scripts/score_solution.py --solution solution --validation $(VALID) --sequences $(SEQ)

score-full: ## score solution/ on the complete validation set
	$(PY) scripts/score_solution.py --solution solution --validation $(VALID)

train: ## build neutrino-wunder and train solution/model.json (see scripts/train_neutrino.sh)
	scripts/train_neutrino.sh

parity: ## Rust trainer vs Python solution.py predictions must agree
	$(PY) scripts/parity_test.py --validation $(VALID)

package: ## zip solution/ into submission.zip and run pre-flight checks
	scripts/package_submission.sh solution submission.zip --validation $(VALID)

check: ## pre-flight checks on an existing submission.zip
	$(PY) scripts/check_submission.py submission.zip --validation $(VALID)

docker-scorer: ## build the organisers' scoring image locally
	docker build -f docker/Dockerfile.scorer -t wunder-scorer .

test-rust: ## unit tests of the neutrino-wunder crate
	cargo test --manifest-path ../neutrino/Cargo.toml -p neutrino-wunder -p neutrino-optimizer

clean:
	rm -rf submission.zip datasets/parity_rust.csv solution/__pycache__
