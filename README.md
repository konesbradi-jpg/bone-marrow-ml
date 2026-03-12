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