#!/usr/bin/env bash
# All four cold-entity DAVIS splits, run sequentially (they share one GPU).
# Each writes run.log / diagnostics.json / train.csv / test.csv into its own dir.
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
COMMON=(--input input_data/DAVIS/full.csv
        --drug-col ligand --target-col protein
        --target-seq-csv input_data/DAVIS_seqs.csv
        --max-iters 20000 --test-ratio 0.2)

$PY run_sae_split_cold.py "${COMMON[@]}" \
    --split-level cold_drug   --save-dir dta_output/DAVIS_cold_drug

$PY run_sae_split_cold.py "${COMMON[@]}" \
    --split-level cold_target --save-dir dta_output/DAVIS_cold_target

# direct: 20% of each axis -> ~4% of pairs are fully cold
$PY run_sae_split_cold.py "${COMMON[@]}" \
    --split-level cold_both --entity-ratio-mode direct \
    --save-dir dta_output/DAVIS_cold_both

# sqrt: ~45% of each axis -> ~20% of pairs are fully cold, comparable in size
# to the pair-level split, at the cost of train coverage
$PY run_sae_split_cold.py "${COMMON[@]}" \
    --split-level cold_both --entity-ratio-mode sqrt \
    --save-dir dta_output/DAVIS_cold_both_sqrt

echo "ALL DONE"
