#!/usr/bin/env python3
"""
Prepare DAVIS / KIBA for the pair-aware SAE split.

The distributed `full.csv` files only carry a protein *name* (DAVIS: gene
symbol with mutation/phospho annotations, e.g. "ABL1(E255K)-phosphorylated";
KIBA: UniProt accession, e.g. "O00141"). Pair-aware SAE needs an actual
target similarity, so we recover sequences from the AlphaFold PDB models
that ship alongside each dataset:

    input_data/DAVIS/pdb/<protein>.pdb
    input_data/KIBA/pdb/AF-<uniprot>-F1-model_v4.pdb

Sequences come from the SEQRES records when present (AlphaFold DB models),
and otherwise from the CA atoms of the model (the DAVIS mutant/domain
entries are ESMFold-style predictions carrying ATOM records only). Either
way, mutants/truncations of the same gene keep their near-identical
sequences and end up with similarity ~1 - exactly the leakage a pair-aware
split has to see.

Usage:
    python dta_prep.py                # writes input_data/{DAVIS,KIBA}_seqs.csv
"""
import json
import os
import re
import sys

import pandas as pd

THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'SEC': 'U', 'PYL': 'O', 'MSE': 'M', 'UNK': 'X',
}

DATASETS = {
    'DAVIS': dict(root='input_data/DAVIS', csv='input_data/DAVIS/full.csv'),
    'KIBA': dict(root='input_data/KIBA', csv='input_data/KIBA/full.csv'),
}


def seqres_from_pdb(path, chain=None):
    """Read the SEQRES records of `path` and return the one-letter sequence.

    Only the first chain is used - these are AlphaFold monomer models, so
    there is exactly one.
    """
    residues, seen_chain = [], None
    with open(path) as fh:
        for line in fh:
            if not line.startswith('SEQRES'):
                continue
            ch = line[11]
            if seen_chain is None:
                seen_chain = ch
            if ch != seen_chain:
                break
            residues.extend(line[19:].split())
    return ''.join(THREE_TO_ONE.get(r, 'X') for r in residues)


def seq_from_atoms(path):
    """Fallback: read the sequence off the CA atoms of the first chain."""
    residues, seen_chain = [], None
    with open(path) as fh:
        for line in fh:
            if not line.startswith(('ATOM  ', 'HETATM')):
                continue
            if line[12:16].strip() != 'CA':
                continue
            ch = line[21]
            if seen_chain is None:
                seen_chain = ch
            if ch != seen_chain:
                break
            residues.append(line[17:20].strip())
    return ''.join(THREE_TO_ONE.get(r, 'X') for r in residues)


def read_sequence(path):
    """SEQRES if the file has it, CA atoms otherwise. Returns (seq, source)."""
    seq = seqres_from_pdb(path)
    if seq:
        return seq, 'SEQRES'
    return seq_from_atoms(path), 'ATOM'


def pdb_index(root):
    """Map protein key -> pdb path, handling both naming conventions."""
    pdb_dir = os.path.join(root, 'pdb')
    index = {}
    for fname in os.listdir(pdb_dir):
        if not fname.endswith('.pdb'):
            continue
        m = re.match(r'AF-([A-Z0-9]+)-F\d+-model.*\.pdb$', fname)
        key = m.group(1) if m else fname[:-4]
        index[key] = os.path.join(pdb_dir, fname)
    return index


def build(name, root, csv, out_path):
    df = pd.read_csv(csv)
    proteins = df['protein'].drop_duplicates().tolist()
    index = pdb_index(root)

    rows, missing = [], []
    for p in proteins:
        path = index.get(p)
        if path is None:
            missing.append(p)
            continue
        seq, source = read_sequence(path)
        if not seq:
            missing.append(p)
            continue
        rows.append({'protein': p, 'sequence': seq, 'length': len(seq),
                     'seq_source': source})

    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)

    n_pairs_lost = int(df['protein'].isin(missing).sum())
    print(f'[{name}] {len(df)} pairs, {len(proteins)} unique proteins')
    print(f'[{name}] sequences recovered for {len(out)}/{len(proteins)} '
          f'({out.length.min()}-{out.length.max()} aa, median {int(out.length.median())}); '
          f'source: {out.seq_source.value_counts().to_dict()}')
    if missing:
        print(f'[{name}] NO structure for {len(missing)}: {missing} '
              f'-> {n_pairs_lost} pairs ({100 * n_pairs_lost / len(df):.2f}%) will be dropped')
    print(f'[{name}] wrote {out_path}')
    return out


def main():
    os.makedirs('input_data', exist_ok=True)
    summary = {}
    for name, cfg in DATASETS.items():
        out_path = f'input_data/{name}_seqs.csv'
        out = build(name, cfg['root'], cfg['csv'], out_path)
        summary[name] = {'n_proteins_with_seq': len(out), 'path': out_path}
        print()
    with open('input_data/seq_extraction_summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2)


if __name__ == '__main__':
    sys.exit(main())
