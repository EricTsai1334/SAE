#!/usr/bin/env python3
"""
Pair-aware extension of Amshoreline/SAE for drug-target-affinity datasets
like DAVIS/KIBA/BindingDB, where similarity has to account for BOTH the
ligand and the protein, not just the ligand.

Why not just reuse run_sae_split.py
------------------------------------
SAE_split.py's main() hard-codes: column 0 of your csv = SMILES, similarity
= Tanimoto over that column only. That's fine for a single-target dataset
(one EGFR, one BACE1, ...). DAVIS has 68 drugs x 442 kinases: a test pair
(drugA, kinase1) can look "novel" on drug identity alone while kinase1's
whole family is saturated in training, or vice versa. That's not the hard,
realistic generalization gap SAE is trying to create.

The core split routine, balance_split(table, S, alpha, ...), is actually
agnostic to how S was built - main() just happens to build it SMILES-only.
So this script builds a better S and calls balance_split() directly,
skipping main() entirely.

How S is built
---------------
1. DAVIS-style datasets have far fewer UNIQUE drugs/targets than pairs
   (68 drugs, 442 proteins vs 30056 pairs). So we compute two small
   similarity matrices once:
     - drug-drug   (n_drugs x n_drugs)   via rdkit Tanimoto (SAE's get_sim)
     - target-target (n_targets x n_targets) via k-mer Jaccard by default,
       or an alignment-based identity if you have Biopython installed
       (--target-sim-method identity)
2. Expand to the full pair-level NxN matrix by indexing into those two
   small matrices with each pair's drug-index/target-index (cheap fancy
   indexing, not O(N^2) individual similarity computations).
3. Combine per pair with --combine: min (default - both drug AND target
   must be close for the pair to count as close, matches the usual
   cold-drug/cold-target/cold-both intuition in the DTA literature),
   product, or mean.

Note on memory: S is materialized as an NxN array (N = number of pairs),
same as vanilla SAE. For full DAVIS (~30k pairs) that's ~30056^2 floats,
~3.6GB in float32. Fine on a machine with a few GB of RAM/VRAM; if that's
tight, subsample pairs first or split per-target and concatenate (see
bottom of this file for that fallback).

Usage
-----
    python run_sae_split_pairs.py \
        --sae-repo /path/to/SAE \
        --input davis.csv \
        --drug-col smiles --target-col target_sequence --label-col affinity \
        --save-dir outputs/davis_balance \
        --test-ratio 0.2 --combine min \
        --max-iters 200   # smoke test; drop for the real 20000-iter run
"""
import argparse
import hashlib
import itertools
import os
import sys

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Target (protein) similarity
# --------------------------------------------------------------------------
def kmer_set(seq, k=3):
    seq = str(seq)
    return {seq[i:i + k] for i in range(len(seq) - k + 1)}


def kmer_jaccard_matrix(seqs, k=3):
    """Fast, dependency-free proxy for sequence similarity.

    Vectorized as a sparse k-mer presence matrix M (n_seqs x n_kmers): the
    pairwise intersection is M @ M.T and the union follows from the row
    sums, so the whole matrix costs one sparse product instead of O(n^2)
    Python set operations.
    """
    import scipy.sparse as sp

    vocab, rows, cols = {}, [], []
    for i, s in enumerate(seqs):
        for km in kmer_set(s, k):
            rows.append(i)
            cols.append(vocab.setdefault(km, len(vocab)))
    M = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(seqs), len(vocab)),
    )
    inter = np.asarray((M @ M.T).todense(), dtype=np.float64)
    sizes = np.asarray(M.sum(axis=1)).ravel()
    union = sizes[:, None] + sizes[None, :] - inter
    S = inter / np.maximum(union, 1.0)
    np.fill_diagonal(S, 1.0)
    return S


def identity_matrix(seqs):
    """Alignment-based percent identity. Needs Biopython (Bio.Align)."""
    from Bio import Align
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    aligner.substitution_matrix = Align.substitution_matrices.load("BLOSUM62")
    n = len(seqs)
    S = np.eye(n, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            aln = aligner.align(seqs[i], seqs[j])[0]
            matches = sum(a == b for a, b in zip(str(aln[0]), str(aln[1])) if a != "-" and b != "-")
            sim = matches / max(len(seqs[i]), len(seqs[j]))
            S[i, j] = S[j, i] = sim
    return S


def build_target_sim(unique_targets, method="kmer", k=3):
    if method == "kmer":
        return kmer_jaccard_matrix(unique_targets, k=k)
    elif method == "identity":
        return identity_matrix(unique_targets)
    else:
        raise ValueError(method)


# --------------------------------------------------------------------------
# Drug (ligand) similarity - reuse SAE's own rdkit/Tanimoto implementation
# --------------------------------------------------------------------------
def build_drug_sim(unique_smiles, sae_repo):
    if sae_repo not in sys.path:
        sys.path.insert(0, sae_repo)
    from utils import get_sim  # repo's utils.py (Morgan fp + Tanimoto)
    return get_sim(list(unique_smiles), list(unique_smiles))


# --------------------------------------------------------------------------
# Pair-level expansion
# --------------------------------------------------------------------------
COMBINERS = {
    "min": np.minimum,
    "product": lambda a, b: a * b,
    "mean": lambda a, b: (a + b) / 2,
    "drug_only": lambda a, b: a,
    "target_only": lambda a, b: b,
}


def build_pair_similarity(df, drug_col, target_col, sae_repo,
                          target_sim_method="kmer", combine="min",
                          target_seqs=None, kmer_k=3, block=2048,
                          return_parts=False):
    """
    Returns an (N, N) float64 similarity matrix over the rows of df, in df's
    current row order.

    `target_seqs` is an optional {target_id: sequence} mapping, for datasets
    whose target column holds an identifier (DAVIS gene symbol, KIBA UniProt
    accession) rather than the sequence itself.

    dtype is float64 on purpose: SAE's objective evaluates
    exp(scale_factor * S) with scale_factor=100, and exp(100) overflows
    float32 (max ~3.4e38) whenever a pair reaches similarity 1.0.

    The N x N expansion is done in row blocks so that peak memory stays at
    S plus one (block x N) temporary, rather than three full N x N arrays.
    """
    unique_drugs = df[drug_col].drop_duplicates().tolist()
    unique_targets = df[target_col].drop_duplicates().tolist()
    drug_to_idx = {d: i for i, d in enumerate(unique_drugs)}
    target_to_idx = {t: i for i, t in enumerate(unique_targets)}

    print(f"[sim] {len(unique_drugs)} unique drugs, {len(unique_targets)} unique targets, "
          f"{len(df)} pairs")

    drug_S = np.asarray(build_drug_sim(unique_drugs, sae_repo), dtype=np.float64)
    seqs = [target_seqs[t] for t in unique_targets] if target_seqs else unique_targets
    target_S = np.asarray(
        build_target_sim(seqs, method=target_sim_method, k=kmer_k), dtype=np.float64
    )

    drug_idx = df[drug_col].map(drug_to_idx).to_numpy()
    target_idx = df[target_col].map(target_to_idx).to_numpy()

    try:
        combiner = COMBINERS[combine]
    except KeyError:
        raise ValueError(combine)

    n = len(df)
    S = np.empty((n, n), dtype=np.float64)
    for start in range(0, n, block):
        stop = min(start + block, n)
        d_blk = drug_S[drug_idx[start:stop]][:, drug_idx]
        t_blk = target_S[target_idx[start:stop]][:, target_idx]
        S[start:stop] = combiner(d_blk, t_blk)

    np.fill_diagonal(S, 0)  # SAE_split.py excludes self-similarity the same way
    if return_parts:
        return S, dict(drug_S=drug_S, target_S=target_S,
                       drug_idx=drug_idx, target_idx=target_idx,
                       unique_drugs=unique_drugs, unique_targets=unique_targets)
    return S


# --------------------------------------------------------------------------
# Driver: call balance_split directly, bypassing SAE_split.py's main()
# --------------------------------------------------------------------------
DEFAULT_HYPERPARAMS = dict(
    sigmoid=True, base_lr=1e-2, optim_kind="ExtraAdam", sched_kind="CosineAnnealing",
    init_kind="custom", lamb=2.03091762e-03, sigma=0.1, scale_factor=100,
    seed=233, init_scale=5, max_iters=20000,
)


def run_split(df, S, test_ratio, save_dir, bins, hyperparam_overrides=None):
    hp = dict(DEFAULT_HYPERPARAMS)
    if hyperparam_overrides:
        hp.update(hyperparam_overrides)
    hp["bins"] = bins

    from SAE_split import balance_split  # imported with sae_repo on sys.path / cwd

    os.makedirs(save_dir, exist_ok=True)
    table = df.reset_index(drop=True)
    train_table, test_table, loss_list, test_count, W, real_R, real_R_hist, hist_score = \
        balance_split(table, S, test_ratio, **hp)

    dirname = "pair_split_" + hashlib.md5(str(hp).encode()).hexdigest()[:8]
    out_dir = os.path.join(save_dir, dirname)
    os.makedirs(out_dir, exist_ok=True)
    train_table.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    test_table.to_csv(os.path.join(out_dir, "test.csv"), index=False)
    np.save(os.path.join(out_dir, "W.npy"), W)
    np.save(os.path.join(out_dir, "real_R.npy"), real_R)
    print(f"[done] real_R hist: {real_R_hist} (score {hist_score:.3f}) -> {out_dir}")
    return out_dir


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sae-repo", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--drug-col", required=True)
    p.add_argument("--target-col", required=True, help="protein sequence column (or unique target id)")
    p.add_argument("--target-seq-csv", default=None,
                   help="csv with columns (protein, sequence), for datasets whose target "
                        "column holds an id rather than the sequence (DAVIS/KIBA)")
    p.add_argument("--kmer-k", type=int, default=3)
    p.add_argument("--label-col", default=None, help="kept in output for reference, not used in similarity")
    p.add_argument("--save-dir", required=True)
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--combine", choices=["min", "product", "mean", "drug_only", "target_only"], default="min")
    p.add_argument("--target-sim-method", choices=["kmer", "identity"], default="kmer")
    p.add_argument("--balance-bins", default="0,0.333333,0.666667,1.0")
    p.add_argument("--max-iters", type=int, default=None)
    p.add_argument("--dry-run", action="store_true", help="build S and stop before optimizing")
    args = p.parse_args()

    sae_repo = os.path.abspath(args.sae_repo)
    df = pd.read_csv(args.input)
    for c in (args.drug_col, args.target_col):
        assert c in df.columns, f"{c!r} not in columns: {list(df.columns)}"

    # validate/canonicalize ligand SMILES, drop unparsable
    if sae_repo not in sys.path:
        sys.path.insert(0, sae_repo)
    from rdkit import Chem
    n_before = len(df)
    df = df[df[args.drug_col].apply(lambda s: Chem.MolFromSmiles(str(s)) is not None)].reset_index(drop=True)
    print(f"[prep] {n_before} rows in, {n_before - len(df)} dropped (unparsable SMILES), {len(df)} remain")

    target_seqs = None
    if args.target_seq_csv:
        seq_df = pd.read_csv(args.target_seq_csv)
        target_seqs = dict(zip(seq_df["protein"], seq_df["sequence"]))
        n_before = len(df)
        df = df[df[args.target_col].isin(target_seqs)].reset_index(drop=True)
        print(f"[prep] {n_before - len(df)} pairs dropped (no target sequence), {len(df)} remain")

    S = build_pair_similarity(
        df, args.drug_col, args.target_col, sae_repo,
        target_sim_method=args.target_sim_method, combine=args.combine,
        target_seqs=target_seqs, kmer_k=args.kmer_k,
    )
    print(f"[sim] pair similarity matrix: {S.shape}, "
          f"{S.nbytes / 1e6:.1f} MB, mean={S.mean():.3f}")

    if args.dry_run:
        print("[dry-run] stopping before optimization")
        return

    bins = [float(x) for x in args.balance_bins.split(",")]
    hp_overrides = {}
    if args.max_iters is not None:
        hp_overrides["max_iters"] = args.max_iters

    os.chdir(sae_repo)  # balance_split lives in SAE_split.py, needs repo on path
    run_split(df, S, args.test_ratio, os.path.abspath(args.save_dir), bins, hp_overrides)


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------
# Fallback if the full NxN pair matrix is too big for memory: split each
# target's drugs independently with vanilla SAE (drug-only similarity is
# fine *within* one target), then concatenate:
#
#   for target, group in df.groupby(target_col):
#       run vanilla run_sae_split.py logic on `group` (drug_col only)
#       concatenate all the per-target train/test splits
#
# This loses the "test target's whole family is saturated in train" signal
# that --combine min/product captures, but avoids ever materializing an
# N x N matrix bigger than any single target's drug count.
# --------------------------------------------------------------------------
