#!/usr/bin/env python3
"""
Compare an SAE pair-aware split against random splits (and, for DAVIS, the
DeepDTA benchmark fold) on the same set of pairs.

Two kinds of evidence:

  A. Split geometry - no model involved. For every split we measure the
     test->train maximum similarity of each test pair (SAE's "real_R"), its
     distribution over the balance bins, and how much of the test set is
     cold-ligand / cold-target / cold-both. This is what SAE actually
     optimizes, so it is the direct check on whether the split differs.

  B. Downstream difficulty - a RandomForest affinity model trained on each
     split. Features are Morgan fingerprints for the ligand concatenated
     with an SVD embedding of the target k-mer similarity matrix, so the
     model can generalize to targets it has not seen (a plain one-hot target
     encoding cannot, which would make cold-target splits trivially
     unlearnable and the comparison meaningless).

The test->train similarity is computed in blocks straight from the small
ligand-ligand and target-target matrices, so the N x N pair matrix is never
materialized.

Usage:
    python compare_dta_splits.py --dataset DAVIS --sae-dir dta_output/DAVIS_...
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

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

from run_sae_split_pairs import COMBINERS, build_drug_sim, build_target_sim
from run_dta_sae_split import DATASETS, coldness, jsonable

REPO = os.path.dirname(os.path.abspath(__file__))
BINS = [0.0, 1 / 3, 2 / 3, 1.0]


# --------------------------------------------------------------------------
# similarity, without ever building the N x N matrix
# --------------------------------------------------------------------------
def build_blocks(df, target_seqs, kmer_k=3):
    drugs = df.ligand.drop_duplicates().tolist()
    targets = df.protein.drop_duplicates().tolist()
    drug_S = np.asarray(build_drug_sim(drugs, REPO), dtype=np.float32)
    target_S = np.asarray(
        build_target_sim([target_seqs[t] for t in targets], method='kmer', k=kmer_k),
        dtype=np.float32)
    d_idx = df.ligand.map({d: i for i, d in enumerate(drugs)}).to_numpy()
    t_idx = df.protein.map({t: i for i, t in enumerate(targets)}).to_numpy()
    return dict(drug_S=drug_S, target_S=target_S, drug_idx=d_idx, target_idx=t_idx,
                drugs=drugs, targets=targets)


def test_to_train_max_sim(blocks, test_rows, train_rows, combine='min', block=1024):
    """max_j in train  S[i, j]  for every test row i, computed blockwise."""
    combiner = COMBINERS[combine]
    dS, tS = blocks['drug_S'], blocks['target_S']
    d_tr, t_tr = blocks['drug_idx'][train_rows], blocks['target_idx'][train_rows]
    out = np.empty(len(test_rows), dtype=np.float32)
    for start in range(0, len(test_rows), block):
        stop = min(start + block, len(test_rows))
        rows = test_rows[start:stop]
        d_blk = dS[blocks['drug_idx'][rows]][:, d_tr]
        t_blk = tS[blocks['target_idx'][rows]][:, t_tr]
        out[start:stop] = combiner(d_blk, t_blk).max(axis=1)
    return out


# --------------------------------------------------------------------------
# features for the downstream model
# --------------------------------------------------------------------------
def build_features(df, blocks, n_target_dims=64, n_bits=1024):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog('rdApp.*')

    fps = {}
    for smi in blocks['drugs']:
        fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), 2, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=np.float32)
        arr[list(fp.GetOnBits())] = 1.0
        fps[smi] = arr
    drug_X = np.stack([fps[d] for d in blocks['drugs']])

    # Unsupervised target embedding: truncated SVD of the k-mer similarity
    # matrix. Uses no labels and is defined for every target, so a model
    # trained without a target can still place it.
    tS = blocks['target_S'].astype(np.float64)
    k = min(n_target_dims, tS.shape[0] - 1)
    U, sv, _ = np.linalg.svd(tS - tS.mean(axis=0, keepdims=True), full_matrices=False)
    target_X = (U[:, :k] * sv[:k]).astype(np.float32)
    target_X /= (np.abs(target_X).max() or 1.0)

    X = np.hstack([drug_X[blocks['drug_idx']], target_X[blocks['target_idx']]])
    y = df.label.to_numpy(dtype=np.float64)
    return X, y


def evaluate(X, y, train_rows, test_rows, n_estimators=200, seed=42):
    model = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1,
                                  random_state=seed, min_samples_leaf=2)
    model.fit(X[train_rows], y[train_rows])
    pred = model.predict(X[test_rows])
    truth = y[test_rows]
    return {
        'rmse': float(mean_squared_error(truth, pred) ** 0.5),
        'mae': float(mean_absolute_error(truth, pred)),
        'r2': float(r2_score(truth, pred)),
        'pearson_r': float(pearsonr(truth, pred)[0]),
        'spearman_r': float(spearmanr(truth, pred)[0]),
    }


# --------------------------------------------------------------------------
def summarize_split(name, df, blocks, train_rows, test_rows, combine):
    real_R = test_to_train_max_sim(blocks, test_rows, train_rows, combine=combine)
    counts, _ = np.histogram(real_R, bins=BINS[:-1] + [1.0 + 1e-9])
    frac = counts / len(test_rows)
    target = np.full(len(counts), 1 / len(counts))
    out = {
        'name': name,
        'n_train': len(train_rows), 'n_test': len(test_rows),
        'real_R_mean': float(real_R.mean()),
        'real_R_median': float(np.median(real_R)),
        'real_R_p10': float(np.percentile(real_R, 10)),
        'real_R_p90': float(np.percentile(real_R, 90)),
        'bin_counts': counts.tolist(),
        'bin_frac': frac.tolist(),
        'hist_score': float(-np.mean(np.abs(frac - target))),
        'label_train_mean': float(df.label.values[train_rows].mean()),
        'label_test_mean': float(df.label.values[test_rows].mean()),
    }
    out.update(coldness(df.iloc[train_rows], df.iloc[test_rows]))
    return out, real_R


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', required=True, choices=sorted(DATASETS))
    p.add_argument('--sae-dir', required=True,
                   help='output directory of run_dta_sae_split.py')
    p.add_argument('--combine', default='min')
    p.add_argument('--kmer-k', type=int, default=3)
    p.add_argument('--random-seeds', default='233,234,235,236,237')
    p.add_argument('--n-estimators', type=int, default=200)
    p.add_argument('--skip-model', action='store_true',
                   help='split geometry only, no RandomForest')
    p.add_argument('--out', default=None)
    args = p.parse_args()

    out_dir = args.out or os.path.join(args.sae_dir, 'comparison')
    os.makedirs(out_dir, exist_ok=True)

    # The universe of pairs is exactly what the SAE run used, so every split
    # here is over an identical row set.
    sae_train = pd.read_csv(os.path.join(args.sae_dir, 'train.csv'))
    sae_test = pd.read_csv(os.path.join(args.sae_dir, 'test.csv'))
    df = pd.concat([sae_train, sae_test], ignore_index=True)
    n = len(df)
    sae_train_rows = np.arange(len(sae_train))
    sae_test_rows = np.arange(len(sae_train), n)
    test_ratio = len(sae_test) / n
    print(f'[setup] {n:,} pairs, {df.ligand.nunique():,} ligands, '
          f'{df.protein.nunique():,} targets, test ratio {test_ratio:.4f}')

    seq_df = pd.read_csv(os.path.join(REPO, DATASETS[args.dataset]['seqs']))
    target_seqs = dict(zip(seq_df.protein, seq_df.sequence))

    t = time.time()
    blocks = build_blocks(df, target_seqs, kmer_k=args.kmer_k)
    print(f'[setup] ligand {blocks["drug_S"].shape} and target {blocks["target_S"].shape} '
          f'similarity matrices in {time.time() - t:.1f}s')

    # ---- assemble the splits ------------------------------------------------
    splits = {'SAE': (sae_train_rows, sae_test_rows)}

    for seed in [int(s) for s in args.random_seeds.split(',')]:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        n_test = len(sae_test)
        splits[f'random_seed{seed}'] = (np.sort(perm[n_test:]), np.sort(perm[:n_test]))

    fold_dir = os.path.join(REPO, 'input_data', args.dataset, 'folds')
    full = pd.read_csv(os.path.join(REPO, DATASETS[args.dataset]['csv']))
    if os.path.isdir(fold_dir) and len(full) == n:
        key_to_row = {k: i for i, k in enumerate(zip(df.ligand, df.protein))}
        full_keys = list(zip(full.ligand, full.protein))
        with open(os.path.join(fold_dir, 'train_fold_setting1.txt')) as fh:
            tr_folds = json.load(fh)
        with open(os.path.join(fold_dir, 'test_fold_setting1.txt')) as fh:
            te_fold = json.load(fh)
        tr_idx = [i for fold in tr_folds for i in fold]
        mapped_tr = [key_to_row[full_keys[i]] for i in tr_idx if full_keys[i] in key_to_row]
        mapped_te = [key_to_row[full_keys[i]] for i in te_fold if full_keys[i] in key_to_row]
        if mapped_tr and mapped_te:
            splits['benchmark_fold'] = (np.array(sorted(set(mapped_tr))),
                                        np.array(sorted(set(mapped_te))))
            print(f'[setup] DeepDTA benchmark fold: {len(mapped_tr):,} train / '
                  f'{len(mapped_te):,} test')
    else:
        print('[setup] benchmark folds skipped (row set does not match full.csv)')

    # ---- A. split geometry --------------------------------------------------
    print(f'\n{"=" * 92}\nA. SPLIT GEOMETRY (test -> train maximum similarity)\n{"=" * 92}')
    results, real_Rs = {}, {}
    hdr = (f'{"split":22s} {"mean":>6s} {"med":>6s} {"p10":>6s} {"p90":>6s} '
           f'{"bin0":>6s} {"bin1":>6s} {"bin2":>6s} {"score":>7s} '
           f'{"coldL":>6s} {"coldT":>6s}')
    print(hdr + '\n' + '-' * len(hdr))
    for name, (tr, te) in splits.items():
        summary, real_R = summarize_split(name, df, blocks, tr, te, args.combine)
        results[name], real_Rs[name] = summary, real_R
        print(f'{name:22s} {summary["real_R_mean"]:6.3f} {summary["real_R_median"]:6.3f} '
              f'{summary["real_R_p10"]:6.3f} {summary["real_R_p90"]:6.3f} '
              f'{summary["bin_frac"][0]:6.3f} {summary["bin_frac"][1]:6.3f} '
              f'{summary["bin_frac"][2]:6.3f} {summary["hist_score"]:7.4f} '
              f'{summary["cold_drug_frac"]:6.3f} {summary["cold_target_frac"]:6.3f}')
    print('\nbin0/1/2 = fraction of test pairs whose nearest training pair falls in')
    print('[0,1/3), [1/3,2/3), [2/3,1]. A balanced SAE split targets 0.333 each;')
    print('hist score is -mean|achieved - target|, so 0 is perfect.')

    # ---- overlap between the SAE test set and the random ones ---------------
    sae_set = set(sae_test_rows.tolist())
    for name, (_, te) in splits.items():
        if name == 'SAE':
            continue
        ov = len(sae_set & set(te.tolist())) / len(sae_test_rows)
        results[name]['overlap_with_sae_test'] = ov
    print('\noverlap of each test set with the SAE test set (0.2 = chance):')
    for name in splits:
        if name != 'SAE':
            print(f'    {name:22s} {results[name]["overlap_with_sae_test"]:.4f}')

    # ---- B. downstream model ------------------------------------------------
    if not args.skip_model:
        print(f'\n{"=" * 92}\nB. DOWNSTREAM AFFINITY MODEL (RandomForest)\n{"=" * 92}')
        X, y = build_features(df, blocks)
        print(f'    features: {X.shape[1]} dims '
              f'(1024 Morgan bits + {X.shape[1] - 1024} target SVD dims)')
        hdr = (f'{"split":22s} {"rmse":>7s} {"mae":>7s} {"r2":>7s} '
               f'{"pearson":>8s} {"spearman":>9s}')
        print(hdr + '\n' + '-' * len(hdr))
        for name, (tr, te) in splits.items():
            t = time.time()
            m = evaluate(X, y, tr, te, n_estimators=args.n_estimators)
            results[name]['model'] = m
            print(f'{name:22s} {m["rmse"]:7.4f} {m["mae"]:7.4f} {m["r2"]:7.4f} '
                  f'{m["pearson_r"]:8.4f} {m["spearman_r"]:9.4f}   ({time.time() - t:.0f}s)')

    # ---- aggregate + save ---------------------------------------------------
    rand = [r for k, r in results.items() if k.startswith('random_seed')]
    agg = {}
    for key in ['real_R_mean', 'hist_score', 'cold_drug_frac', 'cold_target_frac']:
        agg[key] = {'mean': float(np.mean([r[key] for r in rand])),
                    'std': float(np.std([r[key] for r in rand]))}
    if not args.skip_model:
        for key in ['rmse', 'mae', 'r2', 'pearson_r', 'spearman_r']:
            agg[key] = {'mean': float(np.mean([r['model'][key] for r in rand])),
                        'std': float(np.std([r['model'][key] for r in rand]))}

    print(f'\n{"=" * 92}\nSAE vs. RANDOM (mean +/- std over '
          f'{len(rand)} random seeds)\n{"=" * 92}')
    print(f'{"metric":18s} {"SAE":>10s} {"random":>20s} {"delta":>10s}')
    for key, val in agg.items():
        sae_val = results['SAE'][key] if key in results['SAE'] else results['SAE']['model'][key]
        print(f'{key:18s} {sae_val:10.4f} {val["mean"]:11.4f} +/-{val["std"]:6.4f} '
              f'{sae_val - val["mean"]:10.4f}')

    payload = {'dataset': args.dataset, 'sae_dir': args.sae_dir,
               'n_pairs': n, 'test_ratio': test_ratio,
               'splits': results, 'random_aggregate': agg}
    with open(os.path.join(out_dir, 'comparison.json'), 'w') as fh:
        json.dump(payload, fh, indent=2, default=jsonable)

    # ---- figure -------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    edges = np.linspace(0, 1, 41)
    for name, rr in real_Rs.items():
        if name.startswith('random_seed') and not name.endswith('233'):
            continue
        label = 'random (seed 233)' if name.startswith('random_seed') else name
        ax[0].hist(rr, bins=edges, histtype='step', lw=2, label=label, density=True)
    ax[0].set_xlabel('test -> train maximum similarity')
    ax[0].set_ylabel('density')
    ax[0].set_title(f'{args.dataset}: how close is each test pair to training?')
    ax[0].legend(fontsize=8)

    names = list(splits)
    width = 0.8 / len(names)
    for i, name in enumerate(names):
        ax[1].bar(np.arange(3) + i * width, results[name]['bin_frac'], width,
                  label=name if not name.startswith('random_seed')
                  else (f'random x{len(rand)}' if name.endswith('233') else None))
    ax[1].axhline(1 / 3, c='k', ls='--', lw=1, label='balanced target')
    ax[1].set_xticks(np.arange(3) + 0.4 - width / 2)
    ax[1].set_xticklabels(['[0, 1/3)', '[1/3, 2/3)', '[2/3, 1]'])
    ax[1].set_ylabel('fraction of test pairs')
    ax[1].set_title('similarity-bin distribution vs. balanced target')
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_split_comparison.png'), dpi=130)
    plt.close(fig)

    print(f'\n[done] -> {out_dir}')


if __name__ == '__main__':
    main()
