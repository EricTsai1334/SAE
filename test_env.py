import sys
from importlib.metadata import version
import torch
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
from omegaconf import OmegaConf

def test_environment():
    print("=" * 50)
    print("Environment Sanity Check")
    print("=" * 50)
    print(f"Python Version    : {sys.version.split()[0]}")
    print(f"PyTorch Version   : {torch.__version__}")
    print(f"CUDA Available    : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device Name  : {torch.cuda.get_device_name(0)}")
    print(f"RDKit Version     : {rdkit.__version__}")
    print(f"Pandas Version    : {pd.__version__}")
    print(f"NumPy Version     : {np.__version__}")
    print(f"Seaborn Version   : {sns.__version__}")
    print(f"Matplotlib Version: {mpl.__version__}")
    print(f"OmegaConf Version : {version('omegaconf')}")
    
    # Test RDKit molecular parsing
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, "RDKit failed to parse SMILES string."
    mw = Descriptors.MolWt(mol)
    print(f"\nRDKit Test Passed : Aspirin MW = {mw:.2f}")

    # Test PyTorch Tensor creation
    x = torch.ones((2, 2))
    assert x.shape == (2, 2), "PyTorch tensor creation failed."
    print("PyTorch Test Passed: Basic tensor operations working.")
    print("=" * 50)
    print("All checks passed successfully!")

if __name__ == "__main__":
    test_environment()