<<<<<<< HEAD
# Medical Decision Support Application
### Predicting Success of Pediatric Bone Marrow Transplants

## Description
This project aims to build a machine learning system that predicts the success of pediatric bone marrow transplants using clinical data.

The goal is to support physicians in medical decision-making by providing predictive insights and interpretable explanations.

The system includes:
- Data preprocessing 
- Exploratory Data Analysis
- Machine learning model training
- Model evaluation
- Explainability using SHAP
- A web interface for predictions

## Dataset
Dataset used in this project:

https://archive.ics.uci.edu/dataset/565/bone+marrow+transplant+children

The dataset contains clinical information about children who underwent bone marrow transplantation.

Target variable:
- Survival status after transplant

## Exploratory Data Analysis
Exploratory Data Analysis was conducted in:

notebooks/eda.ipynb

The analysis includes:
- Missing values detection and handling
- Outlier analysis
- Class imbalance analysis
- Correlation analysis between features

The dataset is moderately imbalanced (approximately 60% survived and 40% not survived).

## Machine Learning Models
The following machine learning models were trained and evaluated:

- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)

Models were evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

The best performing model was selected and saved for deployment.

## Model Explainability
To improve transparency and interpretability, SHAP (SHapley Additive Explanations) was used.

SHAP helps identify which clinical features most influence the model’s predictions and allows clinicians to better understand the reasoning behind predictions.

## Project Structure
bone-marrow-ml/

data/
    bone-marrow.arff

notebooks/
    eda.ipynb

src/
    data_processing.py
    train_modele.py
    evaluate_model.py

app/
    app.py

tests/

README.md
requirements.txt

## Installation
Install project dependencies:

pip install -r requirements.txt

## Train the Model
Run the training script:

python src/train_modele.py

## Run the Application
Start the Streamlit interface:

streamlit run app/app.py

The interface allows users to input patient data and obtain transplant success predictions.

## Run Tests
Execute automated tests:

pytest tests/

## Reproducibility
To reproduce the project:

1. Install dependencies
pip install -r requirements.txt

2. Train the model
python src/train_modele.py

3. Run the application
streamlit run app/app.py

## Author
Student Project – Coding Week  
Medical Decision Support Application
=======
# 🩸 HémoVision — Prédiction de Succès de Greffe de Moelle Osseuse Pédiatrique

> **Coding Week · 09–15 Mars 2026**  
> Jay · Léandre Zadi · Adama Sana · Ilias Janati

---

## 📌 Présentation

HémoVision est une application d'aide à la décision médicale qui prédit la **survie d'un enfant après une greffe allogénique de cellules souches hématopoïétiques**. Le modèle est entraîné sur 187 patients pédiatriques (LAL, LAM, aplasie médullaire…) et s'appuie sur des variables exclusivement **pré-opératoires** — aucune information post-greffe n'est utilisée.

L'interface Streamlit permet à un praticien de saisir le dossier patient et d'obtenir instantanément une prédiction accompagnée d'une explication SHAP.

---

## 🗂️ Structure du Projet

```
bone_marrow_project/
│
├── app/
│   └── app.py                   # Interface Streamlit (HémoVision)
│
├── data/
│   ├── bone-marrow.arff         # Dataset source (Silesian University)
│   ├── X_test.csv               # Features de test (générées par train_model.py)
│   ├── y_test.csv               # Labels de test
│   └── model_comparison.csv     # Résultats comparatifs des 3 modèles
│
├── models/
│   ├── xgboost_model.pkl        # Pipeline XGBoost (StandardScaler + XGBClassifier)
│   ├── randomforest_model.pkl   # Pipeline Random Forest
│   ├── svm_model.pkl            # Pipeline SVM (probability=True)
│   └── features_list.pkl        # Liste des 44 features finales
│
├── notebooks/
│   └── eda.ipynb                # Analyse exploratoire des données
│
├── src/
│   ├── data_processing.py       # Pipeline de prétraitement complet
│   ├── train_model.py           # Entraînement et sauvegarde des modèles
│   └── evaluate_model.py        # Évaluation, métriques et visualisations
│
├── tests/                       # Tests unitaires
├── Dockerfile                   # Image Docker
├── requirements.txt             # Dépendances Python
└── README.md                    # Ce fichier
```

---

## ⚙️ Installation

### Prérequis

- Python 3.10+

### Via pip

```bash
pip install -r requirements.txt
```

### Via Docker

```bash
docker build -t hemovision .
docker run -p 8501:8501 hemovision
```

---

## 🚀 Utilisation

### 1. Placer le dataset

```
data/bone-marrow.arff
```

### 2. Entraîner les modèles

```bash
python src/train_model.py
```

Génère les fichiers `.pkl` dans `models/` et les données de test dans `data/`.

### 3. Évaluer les modèles

```bash
python src/evaluate_model.py
```

Génère dans `outputs/` : matrices de confusion, courbes ROC, feature importances, heatmap des métriques.

### 4. Lancer l'interface

```bash
streamlit run app/app.py
```

---

## 🔬 Pipeline ML

```
data/bone-marrow.arff
        │
        ▼
load_arff()            ← Décodage bytes + remplacement '?' → NaN (convention ARFF)
        │
        ▼
impute()               ← KNNImputer (numérique) · Mode (catégoriel)
        │
        ▼
optimize_memory()      ← float64→float32  [après imputation]
        │
        ▼
prepare_target()       ← survival_status → target  (0=survie · 1=décès)
        │
        ▼
encode_split_resample()
        ├── One-Hot Encoding     (drop_first=True)
        ├── Train/Test split     (80/20 · stratifié)
        └── SMOTE                (train uniquement — pas de data leakage)
        │
        ▼
drop_invalid_cols()
        ├── 8 colonnes post-greffe  (leakage)
        └── 5 colonnes redondantes  (multicolinéarité)
        │
        ▼
Pipeline sklearn  [StandardScaler + Estimateur]
        ├── RandomForestClassifier  (n_estimators=100)
        ├── XGBClassifier           (eval_metric=logloss)
        └── SVC                     (probability=True)
```

---

## 🚫 Colonnes Exclues

### Variables post-greffe — Data Leakage

Ces informations ne sont **pas disponibles** au moment de la décision clinique.

| Colonne | Raison |
|---|---|
| `IIIV` | GvHD aiguë stade II/III/IV — observable après greffe |
| `Relapse` | Rechute de la maladie — après greffe |
| `aGvHDIIIIV` | GvHD aiguë stade III/IV — après greffe |
| `extcGvHD` | GvHD chronique extensive — après greffe |
| `ANCrecovery` | Temps récupération neutrophiles — après greffe |
| `PLTrecovery` | Temps récupération plaquettes — après greffe |
| `time_to_aGvHD_III_IV` | Délai avant GvHD III/IV — après greffe |
| `survival_time` | Durée de survie — leakage évident |

### Variables redondantes — Multicolinéarité

| Supprimée | Conservée | Relation |
|---|---|---|
| `Donorage35` | `Donorage` | Binarisation seuil 35 ans |
| `Recipientage10` | `Recipientage` | Binarisation seuil 10 ans |
| `Recipientageint` | `Recipientage` | Découpage en intervalles |
| `HLAmismatch` | `HLAmatch` | Binarisation du score HLA |
| `Diseasegroup` | `Disease` | Binarisation maligne/non-maligne |

---

## 📊 Résultats

Les métriques sont générées par `src/evaluate_model.py` et sauvegardées dans `data/model_comparison.csv`.

| Modèle | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| XGBoost | — | — | — | — | — |
| Random Forest | — | — | — | — | — |
| SVM | — | — | — | — | — |

> Remplir après exécution de `evaluate_model.py`

---

## 🧬 Explicabilité SHAP

Chaque prédiction XGBoost est accompagnée d'un graphique **SHAP waterfall** qui décompose la prédiction en contributions individuelles de chaque variable. Cela permet au praticien de comprendre **pourquoi** le modèle prédit un risque élevé ou faible pour un patient donné.

---

## 📚 Source des Données

**Marek Sikora, Lukasz Wrobel**  
Institute of Computer Science, Silesian University of Technology, Gliwice, Pologne

Dataset : [UCI ML Repository — Bone Marrow Transplant: Children](https://archive.ics.uci.edu/ml/datasets/Bone+Marrow+Transplant%3A+Children)

---

## ⚠️ Avertissement

Ce projet est un **prototype académique** développé dans le cadre de la Coding Week. Il n'est **pas certifié** pour un usage clinique réel. Toute décision médicale doit reposer sur l'évaluation d'un professionnel de santé qualifié.
>>>>>>> 66debb900212697f87d49a69e294f90032729b57
