#!/usr/bin/env python3
"""
Pair-aware SAE split for DAVIS / KIBA, instrumented so every stage of the
algorithm is inspectable.

What it does, in order (each stage prints stats and lands in diagnostics.json):

  1. load       - read full.csv, attach target sequences, drop unusable rows,
                  optionally subsample
  2. drug sim   - Morgan/Tanimoto over the unique ligands            (n_d x n_d)
  3. target sim - k-mer Jaccard over the unique target sequences     (n_t x n_t)
  4. pair sim   - expand + combine into the pair-level S             (N x N)
  5. optimize   - SAE_split.balance_split, with a per-closure trace of the
                  losses, the constraint sum(W) = alpha*N, and how fast W
                  binarizes from its fractional initialization
  6. report     - final split, achieved similarity histogram vs. the target
                  distribution, and cold-drug / cold-target / cold-both counts

Example:
    python run_dta_sae_split.py --dataset DAVIS --max-iters 20000
    python run_dta_sae_split.py --dataset KIBA --subsample 30000 --max-iters 20000
"""
import argparse
import contextlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from run_sae_split_pairs import build_pair_similarity, DEFAULT_HYPERPARAMS

REPO = os.path.dirname(os.path.abspath(__file__))

DATASETS = {
    'DAVIS': dict(csv='input_data/DAVIS/full.csv', seqs='input_data/DAVIS_seqs.csv'),
    'KIBA': dict(csv='input_data/KIBA/full.csv', seqs='input_data/KIBA_seqs.csv'),
}


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
class Tee:
    """Mirror stdout into a log file so the full SAE trace is kept on disk."""

    def __init__(self, path):
        self.fh = open(path, 'w')
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.fh.write(s)
        # these runs are long and watched from another shell; unbuffered
        # output costs nothing next to a 20000-iteration optimization
        self.stdout.flush()
        self.fh.flush()

    def flush(self):
        self.stdout.flush()
        self.fh.flush()

    def close(self):
        self.fh.close()


def describe(a, name, percentiles=(1, 25, 50, 75, 90, 99)):
    """Summary stats for a matrix/vector, printed and returned as a dict."""
    a = np.asarray(a).ravel()
    d = {
        'n': int(a.size), 'mean': float(a.mean()), 'std': float(a.std()),
        'min': float(a.min()), 'max': float(a.max()),
    }
    d.update({f'p{p}': float(np.percentile(a, p)) for p in percentiles})
    pct = ' '.join(f'p{p}={d[f"p{p}"]:.3f}' for p in percentiles)
    print(f'    {name:26s} n={d["n"]:>12,} mean={d["mean"]:.4f} sd={d["std"]:.4f} '
          f'min={d["min"]:.3f} {pct} max={d["max"]:.3f}')
    return d


def jsonable(o):
    """json.dump default: numpy scalars/arrays are all over these results."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f'not JSON serializable: {type(o)}')


def offdiag(M):
    n = M.shape[0]
    return M[~np.eye(n, dtype=bool)]


def stage(i, title):
    print(f'\n{"=" * 88}\n[{i}] {title}\n{"=" * 88}')


# --------------------------------------------------------------------------
# stage 5 instrumentation: hook SAE's inner objective
# --------------------------------------------------------------------------
@contextlib.contextmanager
def traced_closure(record_every=25):
    """Record loss / constraint / W-binarization on every Nth closure call.

    balance_split only prints a summary every 500 iterations, and the
    extragradient optimizers evaluate the closure more than once per
    iteration, so hooking the closure gives a much finer picture of what the
    optimizer is actually doing than the printed log does.
    """
    import torch
    import SAE_split

    original = SAE_split.MaximumBalance.closure
    trace, counter, first_test = [], [0], []

    def closure(self, T):
        state = original(self, T)
        counter[0] += 1
        if (counter[0] - 1) % record_every == 0:
            with torch.no_grad():
                W, R = self.W.detach(), self.R.detach()
                n = W.shape[0]
                # which pairs would be the test set if we rounded right now?
                n_test = int(round(self.alpha * n))
                test_idx = torch.topk(W, n_test).indices
                if not first_test:
                    first_test.append(torch.zeros(n, dtype=torch.bool, device=W.device))
                    first_test[0][test_idx] = True
                overlap = float(first_test[0][test_idx].sum()) / n_test
                trace.append(dict(
                    call=counter[0],
                    main_loss=self.main_loss,
                    regular_loss=self.regular_loss,
                    total_loss=self.total_loss,
                    sum_W=float(W.sum()),
                    target_sum_W=float(self.alpha * n),
                    # how binarized is W? SAE starts it fractional and the
                    # entropy term (lamb * BCE(W, W)) pushes it to {0, 1}
                    frac_W_near0=float((W < 0.01).sum()) / n,
                    frac_W_near1=float((W > 0.99).sum()) / n,
                    frac_W_fractional=float(((W >= 0.01) & (W <= 0.99)).sum()) / n,
                    # 1.0 = the rounded split is still the initial random one
                    overlap_with_init_test=overlap,
                    # R = soft max-similarity of each pair to the train set
                    R_mean=float(R.mean()),
                    R_p50=float(R.median()),
                    R_min=float(R.min()),
                    R_max=float(R.max()),
                ))
        return state

    SAE_split.MaximumBalance.closure = closure
    try:
        yield trace
    finally:
        SAE_split.MaximumBalance.closure = original


# --------------------------------------------------------------------------
# stage 1
# --------------------------------------------------------------------------
def load_dataset(name, subsample=None, seed=233):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')

    cfg = DATASETS[name]
    df = pd.read_csv(os.path.join(REPO, cfg['csv']))
    seq_df = pd.read_csv(os.path.join(REPO, cfg['seqs']))
    target_seqs = dict(zip(seq_df['protein'], seq_df['sequence']))
    info = {'n_pairs_raw': len(df),
            'n_ligands_raw': int(df.ligand.nunique()),
            'n_targets_raw': int(df.protein.nunique())}
    print(f'    raw: {len(df):,} pairs, {info["n_ligands_raw"]:,} ligands, '
          f'{info["n_targets_raw"]:,} targets')

    keep = df.protein.isin(target_seqs)
    info['n_dropped_no_sequence'] = int((~keep).sum())
    df = df[keep]
    print(f'    dropped {info["n_dropped_no_sequence"]:,} pairs with no target sequence')

    ok = df.ligand.map(lambda s: Chem.MolFromSmiles(str(s)) is not None)
    info['n_dropped_bad_smiles'] = int((~ok).sum())
    df = df[ok].reset_index(drop=True)
    print(f'    dropped {info["n_dropped_bad_smiles"]:,} pairs with unparsable SMILES')

    if subsample and subsample < len(df):
        df = df.sample(n=subsample, random_state=seed).reset_index(drop=True)
        info['subsampled_to'] = int(subsample)
        print(f'    SUBSAMPLED to {len(df):,} pairs (seed {seed})')

    info.update(n_pairs=len(df),
                n_ligands=int(df.ligand.nunique()),
                n_targets=int(df.protein.nunique()),
                label_mean=float(df.label.mean()), label_std=float(df.label.std()),
                label_min=float(df.label.min()), label_max=float(df.label.max()))
    print(f'    final: {info["n_pairs"]:,} pairs, {info["n_ligands"]:,} ligands, '
          f'{info["n_targets"]:,} targets')
    print(f'    label: mean={info["label_mean"]:.3f} sd={info["label_std"]:.3f} '
          f'range=[{info["label_min"]:.2f}, {info["label_max"]:.2f}]')
    return df, target_seqs, info


# --------------------------------------------------------------------------
# stage 6
# --------------------------------------------------------------------------
def coldness(train_df, test_df):
    """How much of the test set is unseen at the drug / target / pair level."""
    tr_d, tr_t = set(train_df.ligand), set(train_df.protein)
    cold_d = ~test_df.ligand.isin(tr_d)
    cold_t = ~test_df.protein.isin(tr_t)
    n = len(test_df)
    return {
        'n_test': int(n),
        'cold_drug': int(cold_d.sum()), 'cold_drug_frac': float(cold_d.mean()),
        'cold_target': int(cold_t.sum()), 'cold_target_frac': float(cold_t.mean()),
        'cold_both': int((cold_d & cold_t).sum()),
        'cold_both_frac': float((cold_d & cold_t).mean()),
        'warm_both': int((~cold_d & ~cold_t).sum()),
        'warm_both_frac': float((~cold_d & ~cold_t).mean()),
    }


def hist_over_bins(values, bins):
    counts, _ = np.histogram(values, bins=bins)
    return counts.astype(int).tolist()


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------
def plot_similarity_stages(drug_off, target_off, S_sample, row_max, out):
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
    ax[0, 0].hist(drug_off, bins=50, color='#4C72B0')
    ax[0, 0].set_title('Stage 2: ligand-ligand Tanimoto (off-diagonal)')
    ax[0, 1].hist(target_off, bins=50, color='#DD8452')
    ax[0, 1].set_title('Stage 3: target-target k-mer Jaccard (off-diagonal)')
    ax[1, 0].hist(S_sample, bins=50, color='#55A868')
    ax[1, 0].set_title('Stage 4: pair-pair S (random off-diagonal sample)')
    ax[1, 1].hist(row_max, bins=50, color='#C44E52')
    ax[1, 1].set_title('Stage 4: per-pair max similarity to any other pair')
    for a in ax.ravel():
        a.set_yscale('log')
        a.set_xlabel('similarity')
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_optimization(trace_df, out):
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
    ax[0, 0].plot(trace_df.call, trace_df.main_loss, label='main (histogram match)')
    ax[0, 0].plot(trace_df.call, trace_df.regular_loss, label='regular (entropy)')
    ax[0, 0].plot(trace_df.call, trace_df.total_loss, label='total', ls='--', c='k', lw=1)
    ax[0, 0].set_yscale('log')
    ax[0, 0].set_title('Stage 5: objective')
    ax[0, 0].legend(fontsize=8)

    ax[0, 1].plot(trace_df.call, trace_df.sum_W, label='sum(W)')
    ax[0, 1].axhline(trace_df.target_sum_W.iloc[0], c='r', ls='--',
                     label=f'target = alpha*N = {trace_df.target_sum_W.iloc[0]:.0f}')
    ax[0, 1].set_title('Stage 5: test-size equality constraint')
    ax[0, 1].legend(fontsize=8)

    ax[1, 0].plot(trace_df.call, trace_df.frac_W_near0, label='W < 0.01 (train)')
    ax[1, 0].plot(trace_df.call, trace_df.frac_W_near1, label='W > 0.99 (test)')
    ax[1, 0].plot(trace_df.call, trace_df.frac_W_fractional, label='fractional')
    ax[1, 0].set_title('Stage 5: binarization of W')
    ax[1, 0].set_ylim(0, 1)
    ax[1, 0].legend(fontsize=8)

    ax[1, 1].plot(trace_df.call, trace_df.R_mean, label='R mean')
    ax[1, 1].plot(trace_df.call, trace_df.R_p50, label='R median')
    ax[1, 1].fill_between(trace_df.call, trace_df.R_min, trace_df.R_max,
                          alpha=0.2, label='R min-max')
    ax2 = ax[1, 1].twinx()
    ax2.plot(trace_df.call, trace_df.overlap_with_init_test, c='k', ls=':',
             label='overlap with init test set')
    ax2.set_ylim(0, 1.02)
    ax2.set_ylabel('overlap with init split')
    ax[1, 1].set_title('Stage 5: soft max-similarity-to-train R')
    ax[1, 1].legend(fontsize=8, loc='center left')
    ax2.legend(fontsize=8, loc='lower right')
    for a in ax.ravel():
        a.set_xlabel('closure call')
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_result(W, real_R, bins_arr, real_R_hist, n_test, out):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].hist(W, bins=50, color='#4C72B0')
    ax[0].set_yscale('log')
    ax[0].set_title('final W (0 = train, 1 = test)')

    ax[1].hist(real_R, bins=np.linspace(0, 1, 41), color='#C44E52')
    ax[1].set_title('test-to-train max similarity (real_R)')
    ax[1].set_xlabel('similarity')

    centers = [(a + b) / 2 for a, b in zip(bins_arr, bins_arr[1:])]
    width = min(b - a for a, b in zip(bins_arr, bins_arr[1:])) * 0.4
    achieved = np.array(real_R_hist) / n_test
    target = np.full(len(achieved), 1 / len(achieved))
    ax[2].bar(np.array(centers) - width / 2, achieved, width, label='achieved')
    ax[2].bar(np.array(centers) + width / 2, target, width, label='target')
    ax[2].set_title('similarity distribution vs. target')
    ax[2].set_xlabel('similarity bin')
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', required=True, choices=sorted(DATASETS))
    p.add_argument('--save-dir', default=None)
    p.add_argument('--test-ratio', type=float, default=0.2)
    p.add_argument('--combine', default='min',
                   choices=['min', 'product', 'mean', 'drug_only', 'target_only'])
    p.add_argument('--kmer-k', type=int, default=3)
    p.add_argument('--subsample', type=int, default=None,
                   help='cap the number of pairs; the N x N float64 matrix is '
                        '8*N^2 bytes, so full KIBA (118k) would need 112 GB')
    p.add_argument('--max-iters', type=int, default=20000)
    p.add_argument('--seed', type=int, default=233)
    p.add_argument('--lamb', type=float, default=None,
                   help='weight of the entropy term that binarizes W; raising it '
                        'closes the gap between the relaxed and the rounded split')
    p.add_argument('--init-kind', default=None, choices=['normal', 'uniform', 'custom'])
    p.add_argument('--init-scale', type=float, default=None)
    p.add_argument('--base-lr', type=float, default=None)
    p.add_argument('--bins', default='0,0.333333,0.666667,1.0')
    p.add_argument('--record-every', type=int, default=25)
    p.add_argument('--log-freq', type=int, default=500,
                   help="how often balance_split prints its full report; each one "
                        "recomputes the exact test->train similarities, so it is "
                        "informative but not free")
    p.add_argument('--dry-run', action='store_true',
                   help='stop after stage 4 (build the matrices, skip optimization)')
    args = p.parse_args()

    tag = args.dataset + (f'_sub{args.subsample}' if args.subsample else '')
    tag += f'_{args.combine}_iters{args.max_iters}_seed{args.seed}'
    for name in ('lamb', 'init_kind', 'init_scale', 'base_lr'):
        override = getattr(args, name)
        if override is not None:
            tag += f'_{name}-{override}'
    save_dir = args.save_dir or os.path.join(REPO, 'dta_output', tag)
    os.makedirs(save_dir, exist_ok=True)

    tee = Tee(os.path.join(save_dir, 'run.log'))
    sys.stdout = tee
    diag = {'args': vars(args), 'save_dir': save_dir}
    t0 = time.time()

    try:
        stage(1, f'LOAD {args.dataset}')
        df, target_seqs, load_info = load_dataset(
            args.dataset, subsample=args.subsample, seed=args.seed)
        diag['load'] = load_info
        N = len(df)

        gb = 8 * N * N / 1e9
        print(f'    pair matrix will be {N} x {N} float64 = {gb:.1f} GB')
        if gb > 60:
            raise SystemExit(f'refusing to allocate {gb:.0f} GB; pass --subsample')

        stage(2, 'SIMILARITY MATRICES (ligand, target, pair)')
        t = time.time()
        S, parts = build_pair_similarity(
            df, 'ligand', 'protein', REPO, target_sim_method='kmer',
            combine=args.combine, target_seqs=target_seqs, kmer_k=args.kmer_k,
            return_parts=True)
        print(f'    built in {time.time() - t:.1f}s')

        drug_S, target_S = parts['drug_S'], parts['target_S']
        drug_off, target_off = offdiag(drug_S), offdiag(target_S)
        sim = {}
        print(f'\n  stage 2 - ligand-ligand Tanimoto, {drug_S.shape}')
        sim['drug'] = describe(drug_off, 'ligand sim (off-diag)')
        sim['drug']['frac_gt_0.7'] = float((drug_off > 0.7).mean())
        print(f'    fraction of ligand pairs with Tanimoto > 0.7: '
              f'{sim["drug"]["frac_gt_0.7"]:.4f}')

        print(f'\n  stage 3 - target-target k-mer(k={args.kmer_k}) Jaccard, {target_S.shape}')
        sim['target'] = describe(target_off, 'target sim (off-diag)')
        sim['target']['frac_gt_0.9'] = float((target_off > 0.9).mean())
        print(f'    fraction of target pairs with Jaccard > 0.9 (near-identical, '
              f'e.g. mutants of one gene): {sim["target"]["frac_gt_0.9"]:.4f}')

        print(f'\n  stage 4 - pair-pair S = {args.combine}(ligand, target), {S.shape}, '
              f'{S.nbytes / 1e9:.1f} GB float64')
        rng = np.random.default_rng(args.seed)
        idx = rng.integers(0, N, size=(2, 2_000_000))
        idx = idx[:, idx[0] != idx[1]]
        S_sample = S[idx[0], idx[1]]
        sim['pair_sample'] = describe(S_sample, 'pair sim (sampled off-diag)')
        row_max = S.max(axis=1)
        sim['pair_row_max'] = describe(row_max, 'per-pair nearest neighbour')
        print(f'    note: row-max is the max-similarity each pair would have if every '
              f'other pair were in train,\n          i.e. the R that SAE has to push '
              f'down for the high-difficulty test bins.')
        diag['similarity'] = sim

        np.save(os.path.join(save_dir, 'drug_sim.npy'), drug_S)
        np.save(os.path.join(save_dir, 'target_sim.npy'), target_S)
        plot_similarity_stages(drug_off, target_off, S_sample, row_max,
                               os.path.join(save_dir, 'fig1_similarity_stages.png'))

        if args.dry_run:
            print('\n[dry-run] stopping before optimization')
            return

        stage(5, 'OPTIMIZE (SAE_split.balance_split)')
        bins = [float(x) for x in args.bins.split(',')]
        hp = dict(DEFAULT_HYPERPARAMS)
        hp.update(bins=bins, max_iters=args.max_iters, seed=args.seed,
                  log_freq=args.log_freq)
        for name in ('lamb', 'init_kind', 'init_scale', 'base_lr'):
            override = getattr(args, name)
            if override is not None:
                hp[name] = override
        print(f'    hyper-parameters: {hp}')
        print(f'    alpha (test ratio) = {args.test_ratio} -> '
              f'{int(round(args.test_ratio * N)):,} test / '
              f'{N - int(round(args.test_ratio * N)):,} train pairs')

        sys.path.insert(0, REPO)
        from SAE_split import balance_split

        t = time.time()
        with traced_closure(record_every=args.record_every) as trace:
            (train_df, test_df, loss_list, test_count, W, real_R,
             real_R_hist, hist_score) = balance_split(df, S, args.test_ratio, **hp)
        opt_seconds = time.time() - t

        trace_df = pd.DataFrame(trace)
        trace_df.to_csv(os.path.join(save_dir, 'optimization_trace.csv'), index=False)
        plot_optimization(trace_df, os.path.join(save_dir, 'fig2_optimization.png'))
        diag['optimization'] = {
            'seconds': opt_seconds,
            'closure_calls': int(trace_df.call.max()),
            'main_loss_first': float(trace_df.main_loss.iloc[0]),
            'main_loss_last': float(trace_df.main_loss.iloc[-1]),
            'regular_loss_first': float(trace_df.regular_loss.iloc[0]),
            'regular_loss_last': float(trace_df.regular_loss.iloc[-1]),
            'sum_W_last': float(trace_df.sum_W.iloc[-1]),
            'target_sum_W': float(trace_df.target_sum_W.iloc[0]),
            'frac_W_fractional_last': float(trace_df.frac_W_fractional.iloc[-1]),
            'overlap_with_init_test_last': float(trace_df.overlap_with_init_test.iloc[-1]),
            'best_hist_score': float(hist_score),
        }
        print(f'\n    optimization took {opt_seconds / 60:.1f} min, '
              f'{int(trace_df.call.max()):,} closure calls')
        print(f'    main loss {trace_df.main_loss.iloc[0]:.3e} -> '
              f'{trace_df.main_loss.iloc[-1]:.3e}')
        print(f'    sum(W) {trace_df.sum_W.iloc[-1]:.1f} vs target '
              f'{trace_df.target_sum_W.iloc[0]:.1f}')
        print(f'    W still fractional at the end: '
              f'{trace_df.frac_W_fractional.iloc[-1] * 100:.2f}% of pairs')

        stage(6, 'RESULT')
        bins_arr = bins if bins[-1] >= 1.0 else bins + [1.0]
        n_test = len(test_df)
        res = {
            'n_train': len(train_df), 'n_test': n_test,
            'test_ratio_achieved': n_test / N,
            'real_R_hist': list(real_R_hist),
            'real_R_frac': [c / n_test for c in real_R_hist],
            'hist_score': float(hist_score),
        }
        res['real_R'] = describe(real_R, 'test->train max sim')
        res['coldness'] = coldness(train_df, test_df)
        res['label_train_mean'] = float(train_df.label.mean())
        res['label_test_mean'] = float(test_df.label.mean())
        print(f'    train {len(train_df):,} / test {n_test:,} '
              f'(achieved ratio {n_test / N:.4f})')
        print(f'    similarity bins {bins_arr}')
        print(f'    achieved counts  {list(real_R_hist)}')
        print(f'    achieved fracs   {[round(f, 3) for f in res["real_R_frac"]]} '
              f'(target {[round(1 / len(real_R_hist), 3)] * len(real_R_hist)})')
        print(f'    hist score (0 = perfect) {hist_score:.4f}')
        c = res['coldness']
        print(f'    cold ligand {c["cold_drug_frac"]:.3f} | cold target '
              f'{c["cold_target_frac"]:.3f} | cold both {c["cold_both_frac"]:.3f} | '
              f'warm both {c["warm_both_frac"]:.3f}')
        print(f'    label mean train {res["label_train_mean"]:.3f} / '
              f'test {res["label_test_mean"]:.3f}')
        diag['result'] = res

        train_df.to_csv(os.path.join(save_dir, 'train.csv'), index=False)
        test_df.to_csv(os.path.join(save_dir, 'test.csv'), index=False)
        np.save(os.path.join(save_dir, 'W.npy'), W)
        np.save(os.path.join(save_dir, 'real_R.npy'), real_R)
        plot_result(W, real_R, bins_arr, real_R_hist, n_test,
                    os.path.join(save_dir, 'fig3_result.png'))

        diag['total_seconds'] = time.time() - t0
        print(f'\n[done] {time.time() - t0:.0f}s total -> {save_dir}')
    finally:
        with open(os.path.join(save_dir, 'diagnostics.json'), 'w') as fh:
            json.dump(diag, fh, indent=2, default=jsonable)
        sys.stdout = tee.stdout
        tee.close()


if __name__ == '__main__':
    main()
