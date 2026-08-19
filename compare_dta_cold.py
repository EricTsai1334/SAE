#!/usr/bin/env python3
"""
Compare the pair-level SAE split against the cold-entity splits on DAVIS.

Why a separate script from compare_dta_splits.py
-------------------------------------------------
compare_dta_splits.py reconstitutes its pair universe as concat(train, test)
of a single run. That is exact for a pair-level split, where every pair lands
on one side or the other, but a cold_both split deliberately leaves the
"mixed" (semi-cold) pairs out of both, so its train + test is a strict subset
of the dataset. To compare splits against each other they all have to be
indexed into the SAME universe - the full DAVIS pair table - which is what
this script does.

It also adds the control that makes the downstream numbers interpretable:
each split gets a SIZE-MATCHED random baseline (same n_train / n_test, rows
drawn at random). cold_both trains on ~63% of the pairs and tests on ~4%, so
comparing its R2 against a random split that trains on 80% would confound
"the split is harder" with "the training set is smaller". The size-matched
control isolates the split's structure.

Usage:
    python compare_dta_cold.py --dataset DAVIS
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from run_sae_split_pairs import COMBINERS
from run_dta_sae_split import DATASETS, jsonable
from compare_dta_splits import (BINS, build_blocks, test_to_train_max_sim,
                                build_features, evaluate, summarize_split)

REPO = os.path.dirname(os.path.abspath(__file__))

# split name -> (directory under dta_output, human label)
DEFAULT_RUNS = [
    ('pair_SAE',        'DAVIS_min_iters20000_seed233'),
    ('cold_drug',       'DAVIS_cold_drug'),
    ('cold_target',     'DAVIS_cold_target'),
    ('cold_both',       'DAVIS_cold_both'),
    ('cold_both_sqrt',  'DAVIS_cold_both_sqrt'),
]


def rows_for(df_keys, split_dir, fname):
    """Map the (ligand, protein) pairs in split_dir/fname to row indices of the
    full table. Returns None when the file is absent."""
    path = os.path.join(split_dir, fname)
    if not os.path.isfile(path):
        return None
    sub = pd.read_csv(path)
    idx = [df_keys[k] for k in zip(sub.ligand, sub.protein) if k in df_keys]
    return np.array(sorted(set(idx)))


def size_matched_random(n, n_train, n_test, seed):
    """Random split with exactly n_train / n_test rows drawn from n."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return np.sort(perm[n_test:n_test + n_train]), np.sort(perm[:n_test])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', default='DAVIS', choices=sorted(DATASETS))
    p.add_argument('--output-root', default='dta_output')
    p.add_argument('--combine', default='min')
    p.add_argument('--kmer-k', type=int, default=3)
    p.add_argument('--random-seeds', default='233,234,235')
    p.add_argument('--n-estimators', type=int, default=200)
    p.add_argument('--skip-model', action='store_true')
    p.add_argument('--out', default='dta_output/cold_comparison')
    args = p.parse_args()

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    root = os.path.join(REPO, args.output_root)

    # ---- the shared universe: every pair in the dataset ---------------------
    df = pd.read_csv(os.path.join(REPO, DATASETS[args.dataset]['csv']))
    seq_df = pd.read_csv(os.path.join(REPO, DATASETS[args.dataset]['seqs']))
    target_seqs = dict(zip(seq_df.protein, seq_df.sequence))
    df = df[df.protein.isin(target_seqs)].reset_index(drop=True)
    n = len(df)
    df_keys = {k: i for i, k in enumerate(zip(df.ligand, df.protein))}
    print(f'[setup] universe: {n:,} pairs, {df.ligand.nunique():,} ligands, '
          f'{df.protein.nunique():,} targets')

    t = time.time()
    blocks = build_blocks(df, target_seqs, kmer_k=args.kmer_k)
    print(f'[setup] ligand {blocks["drug_S"].shape} / target {blocks["target_S"].shape} '
          f'similarity in {time.time() - t:.1f}s')

    # ---- assemble the splits ------------------------------------------------
    splits, meta = {}, {}
    for name, dirname in DEFAULT_RUNS:
        d = os.path.join(root, dirname)
        tr = rows_for(df_keys, d, 'train.csv')
        te = rows_for(df_keys, d, 'test.csv')
        if tr is None or te is None:
            print(f'[setup] SKIP {name}: no train/test in {d}')
            continue
        splits[name] = (tr, te)
        mx = rows_for(df_keys, d, 'mixed.csv')
        meta[name] = {'dir': dirname, 'n_mixed': 0 if mx is None else len(mx)}
        print(f'[setup] {name:16s} train {len(tr):6,} test {len(te):5,} '
              f'mixed {meta[name]["n_mixed"]:6,}  ({dirname})')

    # size-matched random control for every real split
    seeds = [int(s) for s in args.random_seeds.split(',')]
    for name in list(splits):
        tr, te = splits[name]
        for seed in seeds:
            splits[f'{name}__rand{seed}'] = size_matched_random(n, len(tr), len(te), seed)

    # DeepDTA benchmark fold, for reference
    fold_dir = os.path.join(REPO, 'input_data', args.dataset, 'folds')
    if os.path.isdir(fold_dir):
        with open(os.path.join(fold_dir, 'train_fold_setting1.txt')) as fh:
            tr_folds = json.load(fh)
        with open(os.path.join(fold_dir, 'test_fold_setting1.txt')) as fh:
            te_fold = json.load(fh)
        full = pd.read_csv(os.path.join(REPO, DATASETS[args.dataset]['csv']))
        full_keys = list(zip(full.ligand, full.protein))
        tr_idx = sorted({df_keys[full_keys[i]] for fold in tr_folds for i in fold
                         if full_keys[i] in df_keys})
        te_idx = sorted({df_keys[full_keys[i]] for i in te_fold if full_keys[i] in df_keys})
        if tr_idx and te_idx:
            splits['benchmark_fold'] = (np.array(tr_idx), np.array(te_idx))
            print(f'[setup] benchmark_fold  train {len(tr_idx):6,} test {len(te_idx):5,}')

    # ---- A. split geometry --------------------------------------------------
    print(f'\n{"=" * 100}\nA. SPLIT GEOMETRY (test -> train maximum similarity)\n{"=" * 100}')
    hdr = (f'{"split":24s} {"ntrain":>7s} {"ntest":>6s} {"mean":>6s} {"med":>6s} '
           f'{"bin0":>6s} {"bin1":>6s} {"bin2":>6s} {"score":>7s} '
           f'{"coldL":>6s} {"coldT":>6s} {"cold2":>6s}')
    print(hdr + '\n' + '-' * len(hdr))
    results, real_Rs = {}, {}
    for name, (tr, te) in splits.items():
        s, rr = summarize_split(name, df, blocks, tr, te, args.combine)
        s.update(meta.get(name, {}))
        results[name], real_Rs[name] = s, rr
        print(f'{name:24s} {s["n_train"]:7,} {s["n_test"]:6,} '
              f'{s["real_R_mean"]:6.3f} {s["real_R_median"]:6.3f} '
              f'{s["bin_frac"][0]:6.3f} {s["bin_frac"][1]:6.3f} {s["bin_frac"][2]:6.3f} '
              f'{s["hist_score"]:7.4f} {s["cold_drug_frac"]:6.3f} '
              f'{s["cold_target_frac"]:6.3f} {s["cold_both_frac"]:6.3f}')
    print('\nbin0/1/2 = fraction of test pairs whose nearest training pair falls in')
    print('[0,1/3), [1/3,2/3), [2/3,1]. A balanced split targets 0.333 each; hist')
    print('score is -mean|achieved - target|, so 0 is perfect. coldL/coldT/cold2 =')
    print('fraction of test pairs whose ligand / target / both are absent from train.')

    # ---- B. downstream model ------------------------------------------------
    if not args.skip_model:
        print(f'\n{"=" * 100}\nB. DOWNSTREAM AFFINITY MODEL (RandomForest)\n{"=" * 100}')
        X, y = build_features(df, blocks)
        print(f'    features: {X.shape[1]} dims '
              f'(1024 Morgan bits + {X.shape[1] - 1024} target SVD dims)\n')
        hdr = (f'{"split":24s} {"rmse":>7s} {"mae":>7s} {"r2":>8s} '
               f'{"pearson":>8s} {"spearman":>9s}')
        print(hdr + '\n' + '-' * len(hdr))
        for name, (tr, te) in splits.items():
            m = evaluate(X, y, tr, te, n_estimators=args.n_estimators)
            results[name]['model'] = m
            print(f'{name:24s} {m["rmse"]:7.4f} {m["mae"]:7.4f} {m["r2"]:8.4f} '
                  f'{m["pearson_r"]:8.4f} {m["spearman_r"]:9.4f}')

    # ---- C. each split vs. its own size-matched random control --------------
    print(f'\n{"=" * 100}\nC. EACH SPLIT vs. ITS SIZE-MATCHED RANDOM CONTROL\n{"=" * 100}')
    print('(same n_train / n_test, rows drawn at random - so any gap is the split')
    print(f' structure, not the training-set size; random = mean +/- std over {len(seeds)} seeds)\n')
    deltas = {}
    keys = ['real_R_mean', 'cold_both_frac'] + \
           ([] if args.skip_model else ['rmse', 'r2', 'spearman_r'])
    hdr = f'{"split":18s} {"metric":14s} {"actual":>9s} {"random ctrl":>18s} {"delta":>9s}'
    print(hdr + '\n' + '-' * len(hdr))
    for name, _ in DEFAULT_RUNS:
        if name not in results:
            continue
        ctrls = [results[f'{name}__rand{s}'] for s in seeds if f'{name}__rand{s}' in results]
        if not ctrls:
            continue
        deltas[name] = {}
        for key in keys:
            get = (lambda r: r[key]) if key in results[name] else (lambda r: r['model'][key])
            actual = get(results[name])
            mu = float(np.mean([get(c) for c in ctrls]))
            sd = float(np.std([get(c) for c in ctrls]))
            deltas[name][key] = {'actual': actual, 'random_mean': mu,
                                 'random_std': sd, 'delta': actual - mu}
            print(f'{name:18s} {key:14s} {actual:9.4f} {mu:11.4f} +/-{sd:5.4f} '
                  f'{actual - mu:+9.4f}')
        print()

    payload = {'dataset': args.dataset, 'n_pairs': n, 'runs': dict(DEFAULT_RUNS),
               'random_seeds': seeds, 'splits': results, 'vs_size_matched_random': deltas}
    with open(os.path.join(out_dir, 'cold_comparison.json'), 'w') as fh:
        json.dump(payload, fh, indent=2, default=jsonable)

    # ---- figure -------------------------------------------------------------
    named = [nm for nm, _ in DEFAULT_RUNS if nm in real_Rs]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    edges = np.linspace(0, 1, 41)
    for nm in named:
        ax[0].hist(real_Rs[nm], bins=edges, histtype='step', lw=2, label=nm, density=True)
    ax[0].hist(real_Rs[f'{named[0]}__rand{seeds[0]}'], bins=edges, histtype='step',
               lw=2, ls='--', color='0.4', label='random (size-matched)', density=True)
    ax[0].set_xlabel('test -> train maximum similarity')
    ax[0].set_ylabel('density')
    ax[0].set_title(f'{args.dataset}: how close is each test pair to training?')
    ax[0].legend(fontsize=8)

    width = 0.35
    xs = np.arange(len(named))
    ax[1].bar(xs - width / 2, [results[nm]['cold_drug_frac'] for nm in named],
              width, label='cold ligand')
    ax[1].bar(xs + width / 2, [results[nm]['cold_target_frac'] for nm in named],
              width, label='cold target')
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels(named, rotation=20, ha='right', fontsize=8)
    ax[1].set_ylabel('fraction of test pairs')
    ax[1].set_title('novelty actually delivered')
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_cold_comparison.png'), dpi=150)
    print(f'[done] -> {out_dir}')


if __name__ == '__main__':
    main()
