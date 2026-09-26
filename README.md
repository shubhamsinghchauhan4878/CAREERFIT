# CareerFit — XGBoost Streamlit App

This project converts the supplied CareerFit Colab workflow into a deployable
Streamlit application.

## Model

The notebook workflow uses:

- CGPA as a numeric feature
- Education and Academic Stream with OneHotEncoder
- Skills, Interests, Favorite Subjects and Personality with a custom
  MultiLabelBinarizerTransformer
- LabelEncoder for the target
- XGBClassifier for multi-class prediction
- RandomizedSearchCV for model tuning
- joblib for saving the fitted pipeline and target encoder

Career Goal is excluded from model training, matching the notebook.

## Folder structure

```text
CareerFit_XGBoost_Streamlit/
├── app.py
├── train_model.py
├── custom_transformer.py
├── requirements.txt
├── data/
│   └── Career_Dataset.csv
└── models/
    └── (created after training)
```

## Run locally

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

The training step creates:

```text
models/career_recommendation_xgboost.pkl
models/target_encoder.pkl
models/metrics.csv
```

## Deploy on Streamlit Cloud

Push the project to GitHub, including the trained model files if the repository
size permits.

Set the main file to:

```text
app.py
```

Streamlit will install the packages from `requirements.txt`.

## Important

The supplied notebook's dataset is treated as the source of the model schema.
Do not replace the XGBoost model with TF-IDF/Logistic Regression if the goal is
to reproduce this notebook.
