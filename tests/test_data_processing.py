import pytest
import pandas as pd
import numpy as np
import sys
import os

# Ajouter le dossier src au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_processing import optimize_memory, handle_missing_values


# ─── Tests optimize_memory ───────────────────────────────────────────

def test_optimize_memory_reduces_float():
    """Vérifie que float64 est converti en float32"""
    df = pd.DataFrame({'a': np.array([1.0, 2.0, 3.0], dtype='float64')})
    df_opt = optimize_memory(df.copy())
    assert df_opt['a'].dtype == np.float32

def test_optimize_memory_reduces_int():
    """Vérifie que int64 est converti en int8 ou int32"""
    df = pd.DataFrame({'a': np.array([1, 2, 3], dtype='int64')})
    df_opt = optimize_memory(df.copy())
    assert df_opt['a'].dtype in [np.int8, np.int32]

def test_optimize_memory_same_values():
    """Vérifie que les valeurs ne changent pas après optimisation"""
    df = pd.DataFrame({'a': np.array([1.0, 2.0, 3.0], dtype='float64')})
    df_opt = optimize_memory(df.copy())
    np.testing.assert_array_almost_equal(df['a'].values, df_opt['a'].values)


# ─── Tests handle_missing_values ─────────────────────────────────────

def test_handle_missing_values_no_nan():
    """Vérifie qu'il n'y a plus de NaN après traitement"""
    df = pd.DataFrame({
        'age': [1.0, np.nan, 3.0],
        'weight': [np.nan, 2.0, 3.0]
    })
    df_clean = handle_missing_values(df.copy())
    assert df_clean.isnull().sum().sum() == 0

def test_handle_missing_values_median():
    """Vérifie que les NaN numériques sont remplacés par la médiane"""
    df = pd.DataFrame({'a': [1.0, 2.0, np.nan, 4.0]})
    df_clean = handle_missing_values(df.copy())
    assert df_clean['a'].iloc[2] == pytest.approx(2.0)

def test_handle_missing_categorical():
    """Vérifie que les NaN catégoriels sont remplacés par le mode"""
    df = pd.DataFrame({'cat': ['A', 'A', None, 'B']})
    df_clean = handle_missing_values(df.copy())
    assert df_clean['cat'].isnull().sum() == 0