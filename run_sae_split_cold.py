#!/usr/bin/env python3
"""
Cold-entity extension of run_sae_split_pairs.py / run_dta_sae_split.py.

The problem this fixes
-----------------------
On a dense drug x target grid (DAVIS-style), splitting at the PAIR level
caps how "hard" any test set can look, no matter which pairs you pick or
which algorithm picks them: for almost every test pair, the training set
still contains the same drug (with a different target) or the same target
(with a different drug), and since drugs cluster into scaffolds and targets
cluster into families, that's usually enough to keep the max-similarity-to-
train high.

This is not hypothetical here - it is exactly what the pair-level DAVIS run
in dta_output/DAVIS_min_iters20000_seed233/ produced:

    coldness: cold_drug 0/6012, cold_target 0/6012, warm_both 6012/6012

Every one of the 6012 test pairs had BOTH its ligand and its target already
in training. SAE moved the similarity histogram in the right direction
(hist_score -0.028 vs. -0.078 for random) but the downstream RandomForest
did not get harder at all - R2 0.693 on the SAE split vs. 0.650 on random,
i.e. the "hard" split scored *higher*. A pair-level split cannot manufacture
novelty that the pair grid does not contain.

The fix
-------
Run SAE's balance_split on the SMALL entity-level similarity matrix
(n_drugs x n_drugs, or n_targets x n_targets - 2-3 orders of magnitude
smaller than the pair matrix, so also much cheaper) to decide which DRUGS
or TARGETS are held out, then expand that entity-level decision to pairs:

  --split-level cold_drug   -> hold out whole drugs (all their pairs -> test)
  --split-level cold_target -> hold out whole targets (all their pairs -> test)
  --split-level cold_both   -> hold out a set of drugs AND a set of targets
                                (independently, via two balance_split calls);
                                test = pairs where BOTH endpoints are held out,
                                train = pairs where NEITHER is held out,
                                "mixed" pairs (one endpoint held out, one not)
                                are written separately - they're a real, distinct
                                difficulty tier (semi-cold), not garbage, but
                                mixing them into train or test silently would
                                misrepresent both.

Note on --test-ratio under cold_both
-------------------------------------
The ratio applies to the ENTITY axis being held out, so cold_both yields a
test set of roughly test_ratio^2 of the pairs (0.2 -> 4%) and a train set of
(1 - test_ratio)^2 (0.2 -> 64%), with the remaining ~32% landing in mixed.
Pass --entity-ratio-mode sqrt to solve the other way round: hold out
sqrt(test_ratio) of each axis so that the *pair-level* test set comes out at
test_ratio. That costs train coverage (0.2 -> 55% held out per axis, ~30%
of pairs trainable), which is the honest price of a fully-cold test set.

Usage
-----
    python run_sae_split_cold.py \
        --input input_data/DAVIS/full.csv \
        --drug-col ligand --target-col protein \
        --target-seq-csv input_data/DAVIS_seqs.csv \
        --save-dir dta_output/DAVIS_cold_drug \
        --split-level cold_drug --test-ratio 0.2

Differences from the reference draft this was adapted from
-----------------------------------------------------------
1. --target-seq-csv. DAVIS/KIBA's target column holds an identifier (gene
   symbol / UniProt accession), not a sequence. Feeding those strings to
   build_target_sim() computes a k-mer Jaccard over "AAK1" vs "ABL1(E255K)-
   phosphorylated" - a similarity between *names*. Sequences are mapped in
   first, exactly as run_dta_sae_split.py does.
2. Paths are made absolute before os.chdir(sae_repo), which otherwise
   re-roots any relative --save-dir / --input.
3. --split-level pair is a real choice (the reference docstring promised it
   but argparse rejected it).
4. Writes run.log + diagnostics.json + the held-out entity lists, matching
   what run_dta_sae_split.py leaves behind, so the result is inspectable
   without re-running.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from run_sae_split_pairs import build_drug_sim, build_target_sim, DEFAULT_HYPERPARAMS

REPO = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------
# small helpers (same conventions as run_dta_sae_split.py)
# --------------------------------------------------------------------------
class Tee:
    """Mirror stdout into a log file so the full SAE trace is kept on disk."""

    def __init__(self, path):
        self.fh = open(path, 'w')
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.fh.write(s)
        self.fh.flush()

    def flush(self):
        self.stdout.flush()
        self.fh.flush()

    def close(self):
        self.fh.close()


def jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(f'not JSON serializable: {type(o)}')


def stage(i, title):
    print(f'\n{"=" * 88}\n[{i}] {title}\n{"=" * 88}')


def coldness(train_df, test_df, drug_col, target_col):
    """How much of the test set is unseen at the drug / target / pair level."""
    tr_d, tr_t = set(train_df[drug_col]), set(train_df[target_col])
    cold_d = ~test_df[drug_col].isin(tr_d)
    cold_t = ~test_df[target_col].isin(tr_t)
    return {
        'n_test': int(len(test_df)),
        'cold_drug': int(cold_d.sum()), 'cold_drug_frac': float(cold_d.mean()),
        'cold_target': int(cold_t.sum()), 'cold_target_frac': float(cold_t.mean()),
        'cold_both': int((cold_d & cold_t).sum()),
        'cold_both_frac': float((cold_d & cold_t).mean()),
        'warm_both': int((~cold_d & ~cold_t).sum()),
        'warm_both_frac': float((~cold_d & ~cold_t).mean()),
    }


# --------------------------------------------------------------------------
# entity-level similarity + split
# --------------------------------------------------------------------------
def entity_table_and_sim(df, entity_col, entity_kind, target_seqs=None,
                         target_sim_method='kmer', kmer_k=3):
    """
    Returns (entity_df, S) where entity_df has one row per UNIQUE entity
    (column 'entity'), and S is its similarity matrix - i.e. exactly the
    (table, S) shape balance_split() expects, just at entity granularity.

    For targets, `target_seqs` maps the identifier in `entity_col` to the
    actual protein sequence; without it the k-mer similarity would be
    computed over the identifier strings.
    """
    unique_entities = df[entity_col].drop_duplicates().reset_index(drop=True)
    entity_df = pd.DataFrame({'entity': unique_entities})
    if entity_kind == 'drug':
        S = np.asarray(build_drug_sim(unique_entities.tolist(), REPO), dtype=np.float64)
    else:
        if target_seqs is not None:
            seqs = [target_seqs[t] for t in unique_entities]
        else:
            seqs = unique_entities.tolist()
        S = np.asarray(
            build_target_sim(seqs, method=target_sim_method, k=kmer_k), dtype=np.float64
        )
    np.fill_diagonal(S, 0)
    # float64 on purpose: SAE evaluates exp(scale_factor * S) with
    # scale_factor=100, and exp(100) overflows float32.
    return entity_df, S


def split_entities(entity_df, S, test_ratio, bins, hyperparam_overrides=None):
    """Run SAE's balance_split at entity granularity.

    Returns (train_entities, test_entities, extras).
    """
    from SAE_split import balance_split

    hp = dict(DEFAULT_HYPERPARAMS)
    if hyperparam_overrides:
        hp.update(hyperparam_overrides)
    hp['bins'] = bins

    t0 = time.time()
    train_tbl, test_tbl, loss_list, test_count, W, real_R, real_R_hist, hist_score = \
        balance_split(entity_df.reset_index(drop=True), S, test_ratio, **hp)
    extras = {
        'seconds': time.time() - t0,
        'n_entities': int(len(entity_df)),
        'n_train_entities': int(len(train_tbl)),
        'n_test_entities': int(len(test_tbl)),
        'entity_real_R_hist': [int(x) for x in real_R_hist],
        'entity_hist_score': float(hist_score),
        'main_loss': float(loss_list[0]), 'regular_loss': float(loss_list[1]),
        'sum_W': float(np.sum(W)),
    }
    return set(train_tbl['entity']), set(test_tbl['entity']), extras, W, real_R


# --------------------------------------------------------------------------
# entity decision -> pair-level split
# --------------------------------------------------------------------------
def expand_cold_drug(df, drug_col, train_drugs, test_drugs):
    train_df = df[df[drug_col].isin(train_drugs)].reset_index(drop=True)
    test_df = df[df[drug_col].isin(test_drugs)].reset_index(drop=True)
    return train_df, test_df, None


def expand_cold_target(df, target_col, train_targets, test_targets):
    train_df = df[df[target_col].isin(train_targets)].reset_index(drop=True)
    test_df = df[df[target_col].isin(test_targets)].reset_index(drop=True)
    return train_df, test_df, None


def expand_cold_both(df, drug_col, target_col, train_drugs, test_drugs,
                     train_targets, test_targets):
    drug_in_train = df[drug_col].isin(train_drugs)
    drug_in_test = df[drug_col].isin(test_drugs)
    target_in_train = df[target_col].isin(train_targets)
    target_in_test = df[target_col].isin(test_targets)

    train_df = df[drug_in_train & target_in_train].reset_index(drop=True)
    test_df = df[drug_in_test & target_in_test].reset_index(drop=True)
    mixed_df = df[(drug_in_train & target_in_test)
                  | (drug_in_test & target_in_train)].reset_index(drop=True)
    return train_df, test_df, mixed_df


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sae-repo', default=REPO)
    p.add_argument('--input', required=True)
    p.add_argument('--drug-col', required=True)
    p.add_argument('--target-col', required=True)
    p.add_argument('--target-seq-csv', default=None,
                   help='csv with columns (protein, sequence), for datasets whose '
                        'target column holds an id rather than the sequence')
    p.add_argument('--save-dir', required=True)
    p.add_argument('--split-level',
                   choices=['cold_drug', 'cold_target', 'cold_both', 'pair'],
                   default='cold_both')
    p.add_argument('--test-ratio', type=float, default=0.2,
                   help='test ratio applied to the ENTITY axis being held out')
    p.add_argument('--entity-ratio-mode', choices=['direct', 'sqrt'], default='direct',
                   help="cold_both only: 'sqrt' holds out sqrt(test_ratio) of each "
                        'axis so the pair-level test set lands near test_ratio')
    p.add_argument('--balance-bins', default='0,0.333333,0.666667,1.0')
    p.add_argument('--target-sim-method', choices=['kmer', 'identity'], default='kmer')
    p.add_argument('--kmer-k', type=int, default=3)
    p.add_argument('--combine', default='min',
                   help='pair split-level only, forwarded to build_pair_similarity')
    p.add_argument('--max-iters', type=int, default=None)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    # Resolve every path BEFORE the chdir below, which would otherwise re-root
    # relative --input / --save-dir / --target-seq-csv against the repo.
    sae_repo = os.path.abspath(args.sae_repo)
    input_path = os.path.abspath(args.input)
    save_dir = os.path.abspath(args.save_dir)
    seq_csv = os.path.abspath(args.target_seq_csv) if args.target_seq_csv else None
    if sae_repo not in sys.path:
        sys.path.insert(0, sae_repo)

    os.makedirs(save_dir, exist_ok=True)
    tee = Tee(os.path.join(save_dir, 'run.log'))
    sys.stdout = tee
    t_start = time.time()
    diag = {'args': vars(args), 'save_dir': save_dir}

    try:
        # ---- 1. load ------------------------------------------------------
        stage(1, 'LOAD')
        df = pd.read_csv(input_path)
        for c in (args.drug_col, args.target_col):
            assert c in df.columns, f'{c!r} not in columns: {list(df.columns)}'
        info = {'n_pairs_raw': len(df),
                'n_drugs_raw': int(df[args.drug_col].nunique()),
                'n_targets_raw': int(df[args.target_col].nunique())}
        print(f'    raw: {len(df):,} pairs, {info["n_drugs_raw"]:,} drugs, '
              f'{info["n_targets_raw"]:,} targets')

        target_seqs = None
        if seq_csv:
            seq_df = pd.read_csv(seq_csv)
            target_seqs = dict(zip(seq_df['protein'], seq_df['sequence']))
            keep = df[args.target_col].isin(target_seqs)
            info['n_dropped_no_sequence'] = int((~keep).sum())
            df = df[keep]
            print(f'    dropped {info["n_dropped_no_sequence"]:,} pairs with no target sequence')

        from rdkit import Chem, RDLogger
        RDLogger.DisableLog('rdApp.*')
        ok = df[args.drug_col].map(lambda s: Chem.MolFromSmiles(str(s)) is not None)
        info['n_dropped_bad_smiles'] = int((~ok).sum())
        df = df[ok].reset_index(drop=True)
        print(f'    dropped {info["n_dropped_bad_smiles"]:,} pairs with unparsable SMILES')

        info.update(n_pairs=len(df),
                    n_drugs=int(df[args.drug_col].nunique()),
                    n_targets=int(df[args.target_col].nunique()))
        print(f'    final: {info["n_pairs"]:,} pairs, {info["n_drugs"]:,} drugs, '
              f'{info["n_targets"]:,} targets')
        diag['load'] = info

        bins = [float(x) for x in args.balance_bins.split(',')]
        hp_overrides = {}
        if args.max_iters is not None:
            hp_overrides['max_iters'] = args.max_iters
        if args.seed is not None:
            hp_overrides['seed'] = args.seed

        # ---- 2. entity ratio ----------------------------------------------
        entity_ratio = args.test_ratio
        if args.split_level == 'cold_both' and args.entity_ratio_mode == 'sqrt':
            entity_ratio = float(np.sqrt(args.test_ratio))
            print(f'\n    entity-ratio-mode=sqrt: holding out {entity_ratio:.4f} of each '
                  f'axis so the pair-level test set lands near {args.test_ratio:.2f}')
        diag['entity_ratio'] = entity_ratio

        need_drug_split = args.split_level in ('cold_drug', 'cold_both')
        need_target_split = args.split_level in ('cold_target', 'cold_both')
        train_drugs = test_drugs = train_targets = test_targets = None
        diag['entity_split'] = {}

        # ---- 3. drug axis --------------------------------------------------
        if need_drug_split:
            stage(2, 'DRUG-ENTITY SPLIT')
            drug_entities, drug_S = entity_table_and_sim(df, args.drug_col, 'drug')
            off = drug_S[~np.eye(len(drug_S), dtype=bool)]
            print(f'    {len(drug_entities)} unique drugs, similarity matrix {drug_S.shape}, '
                  f'off-diagonal mean={off.mean():.4f} max={off.max():.4f}')
            if not args.dry_run:
                os.chdir(sae_repo)
                train_drugs, test_drugs, extras, W_d, realR_d = split_entities(
                    drug_entities, drug_S, entity_ratio, bins, hp_overrides)
                print(f'    held out {len(test_drugs)}/{len(drug_entities)} drugs '
                      f'in {extras["seconds"]:.1f}s, entity hist {extras["entity_real_R_hist"]}')
                diag['entity_split']['drug'] = extras
                np.save(os.path.join(save_dir, 'drug_W.npy'), W_d)
                np.save(os.path.join(save_dir, 'drug_real_R.npy'), realR_d)
            np.save(os.path.join(save_dir, 'drug_sim.npy'), drug_S.astype(np.float32))

        # ---- 4. target axis ------------------------------------------------
        if need_target_split:
            stage(3, 'TARGET-ENTITY SPLIT')
            target_entities, target_S = entity_table_and_sim(
                df, args.target_col, 'target', target_seqs=target_seqs,
                target_sim_method=args.target_sim_method, kmer_k=args.kmer_k)
            off = target_S[~np.eye(len(target_S), dtype=bool)]
            print(f'    {len(target_entities)} unique targets, similarity matrix '
                  f'{target_S.shape}, off-diagonal mean={off.mean():.4f} max={off.max():.4f}')
            if not args.dry_run:
                os.chdir(sae_repo)
                train_targets, test_targets, extras, W_t, realR_t = split_entities(
                    target_entities, target_S, entity_ratio, bins, hp_overrides)
                print(f'    held out {len(test_targets)}/{len(target_entities)} targets '
                      f'in {extras["seconds"]:.1f}s, entity hist {extras["entity_real_R_hist"]}')
                diag['entity_split']['target'] = extras
                np.save(os.path.join(save_dir, 'target_W.npy'), W_t)
                np.save(os.path.join(save_dir, 'target_real_R.npy'), realR_t)
            np.save(os.path.join(save_dir, 'target_sim.npy'), target_S.astype(np.float32))

        if args.dry_run:
            print('\n[dry-run] stopping before expansion')
            return

        # ---- 5. expand to pairs ---------------------------------------------
        stage(4, 'EXPAND TO PAIRS')
        if args.split_level == 'cold_drug':
            train_df, test_df, mixed_df = expand_cold_drug(
                df, args.drug_col, train_drugs, test_drugs)
        elif args.split_level == 'cold_target':
            train_df, test_df, mixed_df = expand_cold_target(
                df, args.target_col, train_targets, test_targets)
        elif args.split_level == 'cold_both':
            train_df, test_df, mixed_df = expand_cold_both(
                df, args.drug_col, args.target_col,
                train_drugs, test_drugs, train_targets, test_targets)
        else:  # 'pair' - the original pair-level behaviour, for comparison
            from run_sae_split_pairs import build_pair_similarity
            S = build_pair_similarity(
                df, args.drug_col, args.target_col, sae_repo,
                target_sim_method=args.target_sim_method, combine=args.combine,
                target_seqs=target_seqs, kmer_k=args.kmer_k)
            print(f'    pair similarity {S.shape}, {S.nbytes / 1e9:.2f} GB')
            os.chdir(sae_repo)
            from SAE_split import balance_split
            hp = dict(DEFAULT_HYPERPARAMS)
            hp.update(hp_overrides)
            hp['bins'] = bins
            train_df, test_df, *_ = balance_split(
                df.reset_index(drop=True), S, args.test_ratio, **hp)
            train_df, test_df = train_df.reset_index(drop=True), test_df.reset_index(drop=True)
            mixed_df = None

        # ---- 6. report -------------------------------------------------------
        stage(5, 'REPORT')
        n = len(df)
        result = {
            'n_train': len(train_df), 'n_test': len(test_df),
            'n_mixed': int(len(mixed_df)) if mixed_df is not None else 0,
            'train_frac': len(train_df) / n, 'test_frac': len(test_df) / n,
            'mixed_frac': (len(mixed_df) / n) if mixed_df is not None else 0.0,
        }
        result['coldness'] = coldness(train_df, test_df, args.drug_col, args.target_col)
        if 'label' in df.columns:
            result['label_train_mean'] = float(train_df['label'].mean())
            result['label_test_mean'] = float(test_df['label'].mean())
        print(f'    train {result["n_train"]:,} ({result["train_frac"]:.3f})  '
              f'test {result["n_test"]:,} ({result["test_frac"]:.3f})  '
              f'mixed {result["n_mixed"]:,} ({result["mixed_frac"]:.3f})')
        c = result['coldness']
        print(f'    coldness: cold_drug {c["cold_drug"]}/{c["n_test"]} ({c["cold_drug_frac"]:.3f})  '
              f'cold_target {c["cold_target"]}/{c["n_test"]} ({c["cold_target_frac"]:.3f})  '
              f'cold_both {c["cold_both"]}/{c["n_test"]} ({c["cold_both_frac"]:.3f})  '
              f'warm_both {c["warm_both"]}/{c["n_test"]} ({c["warm_both_frac"]:.3f})')
        diag['result'] = result

        # ---- 7. write ---------------------------------------------------------
        train_df.to_csv(os.path.join(save_dir, 'train.csv'), index=False)
        test_df.to_csv(os.path.join(save_dir, 'test.csv'), index=False)
        if mixed_df is not None:
            mixed_df.to_csv(os.path.join(save_dir, 'mixed.csv'), index=False)
        held_out = {}
        if test_drugs is not None:
            held_out['test_drugs'] = sorted(test_drugs)
        if test_targets is not None:
            held_out['test_targets'] = sorted(test_targets)
        if held_out:
            with open(os.path.join(save_dir, 'held_out_entities.json'), 'w') as fh:
                json.dump(held_out, fh, indent=2, default=jsonable)

        diag['total_seconds'] = time.time() - t_start
        with open(os.path.join(save_dir, 'diagnostics.json'), 'w') as fh:
            json.dump(diag, fh, indent=2, default=jsonable)
        print(f'\n[done] {time.time() - t_start:.1f}s -> {save_dir}')
    finally:
        sys.stdout = tee.stdout
        tee.close()


if __name__ == '__main__':
    main()
