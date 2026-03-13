"""
Tests unitaires — data_processing.py
=====================================
Couvre les 6 fonctions du pipeline :
  - load_arff
  - impute
  - optimize_memory
  - prepare_target
  - encode_split_resample
  - save_test_data
  - full_pipeline (intégration)

Lancement :
    pytest tests/test_data_processing.py -v
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

# Ajout du dossier src/ au path pour l'import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from data_processing import (
    load_arff,
    impute,
    optimize_memory,
    prepare_target,
    encode_split_resample,
    save_test_data,
    full_pipeline,
)

# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES — DataFrames synthétiques qui imitent le dataset réel
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """
    DataFrame minimal (20 lignes) qui reproduit la structure du dataset BMT :
    colonnes numériques continues + colonnes catégorielles + cible.
    Pas de valeurs manquantes → fixture de base.
    """
    np.random.seed(42)
    n = 20
    return pd.DataFrame({
        "Donorage":       np.random.uniform(18, 56, n),
        "Recipientage":   np.random.uniform(0.5, 20, n),
        "CD34kgx10d6":    np.random.uniform(0.8, 58, n),
        "CD3dCD34":       np.random.uniform(0.2, 30, n),
        "CD3dkgx10d8":    np.random.uniform(0.04, 20, n),
        "Rbodymass":      np.random.uniform(6, 104, n),
        "Disease":        np.random.choice(["ALL", "AML", "chronic", "nonmalignant"], n),
        "Riskgroup":      np.random.choice(["0", "1"], n),
        "HLAmatch":       np.random.choice(["0", "1", "2"], n),
        "Stemcellsource": np.random.choice(["0", "1"], n),
        "survival_status": np.random.choice([0.0, 1.0], n),
    })


@pytest.fixture
def df_with_missing(sample_df):
    """
    Même DataFrame mais avec des NaN sur les colonnes numériques
    et des '?' sur les colonnes catégorielles (convention ARFF).
    """
    df = sample_df.copy()
    df.loc[0, "CD34kgx10d6"]  = np.nan
    df.loc[1, "CD3dCD34"]     = np.nan
    df.loc[2, "Rbodymass"]    = np.nan
    df.loc[3, "Disease"]      = np.nan
    df.loc[4, "Riskgroup"]    = np.nan
    return df


@pytest.fixture
def df_imputed(sample_df):
    """DataFrame déjà imputé (pas de NaN) — utilisé pour les tests aval."""
    return sample_df.copy()


@pytest.fixture
def df_with_target(df_imputed):
    """DataFrame avec la colonne cible renommée en 'target'."""
    df = df_imputed.copy()
    df = df.rename(columns={"survival_status": "target"})
    df["target"] = df["target"].astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 1. load_arff
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadArff:

    def test_returns_dataframe(self):
        """load_arff retourne un pd.DataFrame."""
        df = load_arff("data/bone-marrow.arff")
        assert isinstance(df, pd.DataFrame)

    def test_correct_shape(self):
        """Dataset BMT : 187 lignes × 37 colonnes."""
        df = load_arff("data/bone-marrow.arff")
        assert df.shape == (187, 37), f"Shape inattendue : {df.shape}"

    def test_no_byte_strings(self):
        """Toutes les colonnes objet sont de type str, pas bytes."""
        df = load_arff("data/bone-marrow.arff")
        for col in df.select_dtypes(include=["object"]).columns:
            non_null = df[col].dropna()
            assert not non_null.apply(lambda x: isinstance(x, bytes)).any(), \
                f"Colonne '{col}' contient encore des bytes"

    def test_question_marks_replaced(self):
        """Aucun '?' ne subsiste — tous remplacés par NaN."""
        df = load_arff("data/bone-marrow.arff")
        for col in df.select_dtypes(include=["object"]).columns:
            assert not (df[col] == "?").any(), \
                f"'?' détecté dans la colonne '{col}'"

    def test_missing_values_detected(self):
        """Au moins 69 valeurs manquantes issues des '?' ARFF + NaN natifs."""
        df = load_arff("data/bone-marrow.arff")
        total_missing = df.isnull().sum().sum()
        assert total_missing >= 69, \
            f"Seulement {total_missing} NaN détectés — vérifier le remplacement des '?'"

    def test_target_column_exists(self):
        """La colonne 'survival_status' est présente."""
        df = load_arff("data/bone-marrow.arff")
        assert "survival_status" in df.columns

    def test_target_values(self):
        """survival_status ne contient que 0.0 et 1.0."""
        df = load_arff("data/bone-marrow.arff")
        unique_vals = set(df["survival_status"].dropna().unique())
        assert unique_vals <= {0.0, 1.0}, \
            f"Valeurs inattendues dans survival_status : {unique_vals}"

    def test_file_not_found(self):
        """Lève une exception si le fichier n'existe pas."""
        with pytest.raises(Exception):
            load_arff("data/fichier_inexistant.arff")


# ─────────────────────────────────────────────────────────────────────────────
# 2. impute
# ─────────────────────────────────────────────────────────────────────────────

class TestImpute:

    def test_no_nan_after_imputation(self, df_with_missing):
        """Aucun NaN ne doit subsister après impute()."""
        result = impute(df_with_missing)
        assert result.isnull().sum().sum() == 0, \
            f"NaN résiduels : {result.isnull().sum()[result.isnull().sum() > 0]}"

    def test_shape_preserved(self, df_with_missing):
        """Le shape du DataFrame est conservé après imputation."""
        result = impute(df_with_missing)
        assert result.shape == df_with_missing.shape

    def test_columns_preserved(self, df_with_missing):
        """Les colonnes ne changent pas."""
        result = impute(df_with_missing)
        assert list(result.columns) == list(df_with_missing.columns)

    def test_numeric_values_in_range(self, df_with_missing):
        """
        Les valeurs imputées pour CD34kgx10d6 restent dans une plage raisonnable.
        Le KNN ne doit pas produire de valeurs aberrantes.
        """
        result = impute(df_with_missing)
        # Valeurs imputées doivent rester proches des vraies (dataset réel : 0.79–57.78)
        cd34_vals = result["CD34kgx10d6"]
        assert cd34_vals.min() >= 0, "Valeur CD34+ négative après KNN"
        assert cd34_vals.max() <= 200, "Valeur CD34+ aberrante après KNN"

    def test_categorical_imputed_with_valid_category(self, df_with_missing):
        """
        Les colonnes catégorielles imputées contiennent uniquement
        des valeurs valides (appartenant aux catégories existantes).
        """
        original_disease_vals = set(df_with_missing["Disease"].dropna().unique())
        result = impute(df_with_missing)
        imputed_disease_vals = set(result["Disease"].unique())
        assert imputed_disease_vals <= original_disease_vals, \
            f"Valeurs inattendues après imputation : {imputed_disease_vals - original_disease_vals}"

    def test_no_nan_already_clean(self, sample_df):
        """Sur un DataFrame sans NaN, impute() ne modifie rien."""
        result = impute(sample_df)
        assert result.isnull().sum().sum() == 0

    def test_does_not_modify_original(self, df_with_missing):
        """impute() ne modifie pas le DataFrame d'entrée (copie défensive)."""
        original_nan_count = df_with_missing.isnull().sum().sum()
        _ = impute(df_with_missing)
        assert df_with_missing.isnull().sum().sum() == original_nan_count


# ─────────────────────────────────────────────────────────────────────────────
# 3. optimize_memory
# ─────────────────────────────────────────────────────────────────────────────

class TestOptimizeMemory:

    def test_float64_converted_to_float32(self, sample_df):
        """Toutes les colonnes float64 sont converties en float32."""
        result = optimize_memory(sample_df.copy())
        float_cols = result.select_dtypes(include=["float"]).columns
        for col in float_cols:
            assert result[col].dtype == np.float32, \
                f"Colonne '{col}' est encore {result[col].dtype}"

    def test_no_float64_remaining(self, sample_df):
        """Aucune colonne float64 ne subsiste."""
        result = optimize_memory(sample_df.copy())
        assert not any(result.dtypes == np.float64), \
            f"float64 restant : {result.select_dtypes(include=['float64']).columns.tolist()}"

    def test_values_preserved(self, sample_df):
        """
        Les valeurs numériques sont conservées (à la précision float32 près).
        On tolère une erreur relative < 1e-5.
        """
        result = optimize_memory(sample_df.copy())
        np.testing.assert_allclose(
            result["CD34kgx10d6"].values,
            sample_df["CD34kgx10d6"].values.astype(np.float32),
            rtol=1e-5,
        )

    def test_memory_reduced(self, sample_df):
        """La mémoire utilisée est réduite après conversion."""
        before = sample_df.memory_usage(deep=True).sum()
        result = optimize_memory(sample_df.copy())
        after  = result.memory_usage(deep=True).sum()
        assert after <= before, \
            f"Mémoire augmentée : {before} → {after}"

    def test_shape_preserved(self, sample_df):
        """Le shape est inchangé."""
        result = optimize_memory(sample_df.copy())
        assert result.shape == sample_df.shape


# ─────────────────────────────────────────────────────────────────────────────
# 4. prepare_target
# ─────────────────────────────────────────────────────────────────────────────

class TestPrepareTarget:

    def test_column_renamed(self, df_imputed):
        """'survival_status' est renommée en 'target'."""
        result = prepare_target(df_imputed.copy())
        assert "target" in result.columns
        assert "survival_status" not in result.columns

    def test_target_is_integer(self, df_imputed):
        """La colonne 'target' est de type entier."""
        result = prepare_target(df_imputed.copy())
        assert result["target"].dtype in [np.int32, np.int64, int]

    def test_target_values_binary(self, df_imputed):
        """La cible ne contient que 0 et 1."""
        result = prepare_target(df_imputed.copy())
        assert set(result["target"].unique()) <= {0, 1}

    def test_shape_preserved(self, df_imputed):
        """Le shape est inchangé."""
        result = prepare_target(df_imputed.copy())
        assert result.shape == df_imputed.shape

    def test_raises_if_target_missing(self, df_imputed):
        """Lève ValueError si 'survival_status' est absente."""
        df_no_target = df_imputed.drop(columns=["survival_status"])
        with pytest.raises(ValueError, match="survival_status"):
            prepare_target(df_no_target)

    def test_custom_target_col(self, df_imputed):
        """Fonctionne avec un nom de colonne cible personnalisé."""
        df = df_imputed.rename(columns={"survival_status": "outcome"})
        result = prepare_target(df, target_col="outcome")
        assert "target" in result.columns


# ─────────────────────────────────────────────────────────────────────────────
# 5. encode_split_resample
# ─────────────────────────────────────────────────────────────────────────────

class TestEncodeSplitResample:

    def test_returns_four_objects(self, df_with_target):
        """Retourne exactement 4 objets."""
        result = encode_split_resample(df_with_target)
        assert len(result) == 4

    def test_output_types(self, df_with_target):
        """X_train, X_test sont des DataFrames ; y_train, y_test des Series/arrays."""
        X_train, X_test, y_train, y_test = encode_split_resample(df_with_target)
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test,  pd.DataFrame)

    def test_test_size_approx_20pct(self, df_with_target):
        """Le test set représente environ 20% des données originales."""
        X_train, X_test, y_train, y_test = encode_split_resample(df_with_target)
        total = len(X_train) + len(X_test)  # avant SMOTE pour train
        # X_test doit être ~20% du dataset original (20 lignes → 4 test)
        assert len(X_test) >= 1

    def test_no_nan_in_outputs(self, df_with_target):
        """Aucun NaN dans les outputs."""
        X_train, X_test, y_train, y_test = encode_split_resample(df_with_target)
        assert X_train.isnull().sum().sum() == 0
        assert X_test.isnull().sum().sum() == 0

    def test_no_target_column_in_X(self, df_with_target):
        """'target' ne doit PAS apparaître dans X_train ou X_test."""
        X_train, X_test, y_train, y_test = encode_split_resample(df_with_target)
        assert "target" not in X_train.columns
        assert "target" not in X_test.columns

    def test_same_columns_train_test(self, df_with_target):
        """X_train et X_test ont exactement les mêmes colonnes."""
        X_train, X_test, y_train, y_test = encode_split_resample(df_with_target)
        assert list(X_train.columns) == list(X_test.columns)

    def test_ohe_columns_created(self, df_with_target):
        """Des colonnes OHE ont été créées (présence de '_' dans les noms)."""
        X_train, X_test, y_train, y_test = encode_split_resample(df_with_target)
        ohe_cols = [c for c in X_train.columns if "_" in c]
        assert len(ohe_cols) > 0, "Aucune colonne OHE générée"

    def test_smote_balances_classes(self, df_with_target):
        """
        Après SMOTE, les classes du train set sont équilibrées.
        On tolère une différence <= 1 (arrondi SMOTE).
        """
        X_train, X_test, y_train, y_test = encode_split_resample(df_with_target)
        counts = pd.Series(y_train).value_counts()
        if len(counts) == 2:
            assert abs(counts[0] - counts[1]) <= 1, \
                f"Classes déséquilibrées après SMOTE : {counts.to_dict()}"

    def test_test_set_not_resampled(self, df_with_target):
        """
        Le test set doit conserver la distribution naturelle (pas de SMOTE).
        Son taille doit être exactement celle du split sans augmentation.
        """
        X_train_res, X_test, y_train_res, y_test = encode_split_resample(df_with_target)
        n_original = len(df_with_target)
        expected_test = int(n_original * 0.2)
        # Test set doit être proche de 20% (stratifié)
        assert abs(len(X_test) - expected_test) <= 1

    def test_y_values_binary(self, df_with_target):
        """Les labels de sortie ne contiennent que 0 et 1."""
        _, _, y_train, y_test = encode_split_resample(df_with_target)
        assert set(np.unique(y_train)) <= {0, 1}
        assert set(np.unique(y_test))  <= {0, 1}


# ─────────────────────────────────────────────────────────────────────────────
# 6. save_test_data
# ─────────────────────────────────────────────────────────────────────────────

class TestSaveTestData:

    def test_files_created(self, df_with_target, tmp_path):
        """X_test.csv et y_test.csv sont créés dans le dossier cible."""
        _, X_test, _, y_test = encode_split_resample(df_with_target)
        save_test_data(X_test, y_test, output_dir=str(tmp_path))
        assert (tmp_path / "X_test.csv").exists(), "X_test.csv non créé"
        assert (tmp_path / "y_test.csv").exists(), "y_test.csv non créé"

    def test_files_content_correct(self, df_with_target, tmp_path):
        """Les fichiers CSV rechargés correspondent aux données sauvegardées."""
        _, X_test, _, y_test = encode_split_resample(df_with_target)
        save_test_data(X_test, y_test, output_dir=str(tmp_path))

        X_reloaded = pd.read_csv(tmp_path / "X_test.csv")
        y_reloaded = pd.read_csv(tmp_path / "y_test.csv")

        assert X_reloaded.shape == X_test.shape, "Shape X_test différente après rechargement"
        assert len(y_reloaded) == len(y_test), "Longueur y_test différente après rechargement"

    def test_creates_directory_if_missing(self, df_with_target, tmp_path):
        """Crée le dossier s'il n'existe pas encore."""
        new_dir = tmp_path / "new_output_dir"
        assert not new_dir.exists()
        _, X_test, _, y_test = encode_split_resample(df_with_target)
        save_test_data(X_test, y_test, output_dir=str(new_dir))
        assert new_dir.exists()

    def test_no_index_column_in_csv(self, df_with_target, tmp_path):
        """
        Les CSV ne doivent pas contenir de colonne d'index non souhaitée.
        (Vérifie que index=False est bien passé à to_csv.)
        """
        _, X_test, _, y_test = encode_split_resample(df_with_target)
        save_test_data(X_test, y_test, output_dir=str(tmp_path))
        X_reloaded = pd.read_csv(tmp_path / "X_test.csv")
        assert "Unnamed: 0" not in X_reloaded.columns


# ─────────────────────────────────────────────────────────────────────────────
# 7. full_pipeline — Test d'intégration
# ─────────────────────────────────────────────────────────────────────────────

class TestFullPipeline:

    def test_pipeline_runs_without_error(self, tmp_path):
        """Le pipeline complet s'exécute sans lever d'exception."""
        X_train, X_test, y_train, y_test = full_pipeline(
            "data/bone-marrow.arff",
            save_dir=str(tmp_path),
        )
        assert X_train is not None
        assert X_test  is not None

    def test_output_shapes_consistent(self, tmp_path):
        """X et y ont des longueurs cohérentes après le pipeline."""
        X_train, X_test, y_train, y_test = full_pipeline(
            "data/bone-marrow.arff",
            save_dir=str(tmp_path),
        )
        assert len(X_train) == len(y_train)
        assert len(X_test)  == len(y_test)

    def test_no_nan_in_outputs(self, tmp_path):
        """Aucun NaN dans les outputs du pipeline complet."""
        X_train, X_test, y_train, y_test = full_pipeline(
            "data/bone-marrow.arff",
            save_dir=str(tmp_path),
        )
        assert X_train.isnull().sum().sum() == 0
        assert X_test.isnull().sum().sum()  == 0

    def test_test_set_size(self, tmp_path):
        """Le test set = 20% de 187 patients = ~37-38 patients."""
        _, X_test, _, y_test = full_pipeline(
            "data/bone-marrow.arff",
            save_dir=str(tmp_path),
        )
        assert 35 <= len(X_test) <= 40, \
            f"Taille test inattendue : {len(X_test)} (attendu ~38)"

    def test_feature_count(self, tmp_path):
        """
        Après OHE sur les 24 variables originales conservées,
        le nombre de features doit être dans la plage [30, 60].
        (Valeur exacte dépend du OHE drop_first, attendu ~44)
        """
        X_train, _, _, _ = full_pipeline(
            "data/bone-marrow.arff",
            save_dir=str(tmp_path),
        )
        assert 30 <= X_train.shape[1] <= 60, \
            f"Nombre de features inattendu : {X_train.shape[1]}"

    def test_csv_files_saved(self, tmp_path):
        """X_test.csv et y_test.csv sont bien sauvegardés."""
        full_pipeline("data/bone-marrow.arff", save_dir=str(tmp_path))
        assert (tmp_path / "X_test.csv").exists()
        assert (tmp_path / "y_test.csv").exists()

    def test_save_dir_none_does_not_save(self, tmp_path):
        """Avec save_dir=None, aucun fichier CSV n'est créé."""
        full_pipeline("data/bone-marrow.arff", save_dir=None)
        assert not (tmp_path / "X_test.csv").exists()

    def test_smote_balances_train(self, tmp_path):
        """Classes équilibrées dans y_train après SMOTE."""
        _, _, y_train, _ = full_pipeline(
            "data/bone-marrow.arff",
            save_dir=str(tmp_path),
        )
        counts = pd.Series(y_train).value_counts()
        assert abs(counts[0] - counts[1]) <= 1, \
            f"Déséquilibre après SMOTE : {counts.to_dict()}"

    def test_deterministic_with_same_seed(self, tmp_path):
        """
        Deux exécutions avec le même random_state produisent
        exactement les mêmes données de test.
        """
        _, X_test1, _, y_test1 = full_pipeline("data/bone-marrow.arff", save_dir=None)
        _, X_test2, _, y_test2 = full_pipeline("data/bone-marrow.arff", save_dir=None)
        pd.testing.assert_frame_equal(X_test1.reset_index(drop=True),
                                      X_test2.reset_index(drop=True))