import pandas as pd
import numpy as np
from scipy.io import arff

def load_and_clean_data(file_path):
    """
    Charge le fichier .arff et remplace les '?' par des valeurs nulles.
    """
    # Chargement du fichier ARFF [cite: 17]
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)
    
    # Remplacement des '?' par NaN (en gérant le format bytes habituel des ARFF) 
    df = df.replace(b'?', np.nan)
    
    # Conversion en numérique pour les colonnes qui le permettent
    df = df.apply(pd.to_numeric, errors='ignore')
    
    return df

def optimize_memory(df):
    """
    Optimise l'usage mémoire en ajustant les types de données (ex: float64 vers float32).
    Exigence du projet 4. [cite: 49]
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            # Optimisation des entiers [cite: 49]
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            
            # Optimisation des flottants [cite: 49]
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                    
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Mémoire avant: {start_mem:.2f} MB - Après: {end_mem:.2f} MB') [cite: 50]
    return df

def handle_missing_values(df):
    """
    Stratégie de traitement des valeurs manquantes identifiées lors de l'EDA. 
    """
    # Exemple : Remplissage par la médiane pour les colonnes numériques
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    
    # Remplissage par le mode pour les colonnes catégorielles
    for col in df.select_dtypes(include=[object]).columns:
        df[col] = df[col].fillna(df[col].mode()[0])
        
    return df