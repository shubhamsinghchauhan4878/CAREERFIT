
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from xgboost import XGBClassifier

from custom_transformer import MultiLabelBinarizerTransformer

DATA_PATH = "data/Career_Dataset.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print("Original shape:", df.shape)

# Same cleaning logic as the Colab notebook.
df_clean = df.copy()

text_columns = df_clean.select_dtypes(include="object").columns.tolist()
for col in text_columns:
    df_clean[col] = df_clean[col].astype(str).str.strip()

multi_value_cols = [
    "Skills", "Interests", "Favorite Subjects", "Personality"
]
for col in multi_value_cols:
    df_clean[col] = df_clean[col].apply(
        lambda x: ", ".join(
            [tok.strip() for tok in x.split(",") if tok.strip()]
        )
    )

df_model = df_clean.drop(
    columns=["Index", "Student Name", "Career Goal"],
    errors="ignore"
)

target = "Target Career"
X = df_model.drop(columns=[target])
y_text = df_model[target]

X_train, X_test, y_train_text, y_test_text = train_test_split(
    X,
    y_text,
    test_size=0.20,
    random_state=42,
    stratify=y_text,
)

numeric_features = ["CGPA"]
categorical_features = ["Education", "Academic Stream"]
multilabel_features = [
    "Skills", "Interests", "Favorite Subjects", "Personality"
]

transformers = [
    ("numeric", "passthrough", numeric_features),
    (
        "categorical",
        OneHotEncoder(handle_unknown="ignore"),
        categorical_features
    ),
]

for col in multilabel_features:
    transformers.append(
        (
            f"multilabel_{col.replace(' ', '_')}",
            MultiLabelBinarizerTransformer(),
            [col],
        )
    )

preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop"
)

print("Fitting preprocessing...")
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train_text)
y_test = label_encoder.transform(y_test_text)
n_classes = len(label_encoder.classes_)

print("Processed training shape:", X_train_processed.shape)
print("Classes:", list(label_encoder.classes_))

# Baseline model, matching the notebook.
baseline_model = XGBClassifier(
    objective="multi:softprob",
    num_class=n_classes,
    eval_metric="mlogloss",
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    n_jobs=-1,
)

print("Training baseline XGBoost...")
baseline_model.fit(X_train_processed, y_train)
baseline_pred = baseline_model.predict(X_test_processed)
print(
    "Baseline accuracy:",
    f"{accuracy_score(y_test, baseline_pred)*100:.2f}%"
)

# Same tuning strategy as the notebook.
param_distributions = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "min_child_weight": [1, 3, 5],
    "gamma": [0, 0.1, 0.2],
}

base_model = XGBClassifier(
    objective="multi:softprob",
    num_class=n_classes,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
)

random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_distributions,
    n_iter=10,
    scoring="f1_weighted",
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1,
)

print("Running RandomizedSearchCV...")
random_search.fit(X_train_processed, y_train)

best_model = random_search.best_estimator_
print("Best parameters:", random_search.best_params_)
print("Best CV F1:", random_search.best_score_)

y_pred = best_model.predict(X_test_processed)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1w = f1_score(y_test, y_pred, average="weighted", zero_division=0)
f1m = f1_score(y_test, y_pred, average="macro", zero_division=0)

print(f"Final Accuracy: {acc:.4f}")
print(f"Final Precision: {prec:.4f}")
print(f"Final Recall: {rec:.4f}")
print(f"Final F1 weighted: {f1w:.4f}")
print(f"Final F1 macro: {f1m:.4f}")
print(classification_report(
    y_test, y_pred,
    target_names=label_encoder.classes_,
    zero_division=0
))

cm = confusion_matrix(y_test, y_pred)
np.save(os.path.join(MODEL_DIR, "confusion_matrix.npy"), cm)

# Pipeline contains the fitted preprocessor and fitted XGBoost model.
final_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", best_model),
])

joblib.dump(
    final_pipeline,
    os.path.join(MODEL_DIR, "career_recommendation_xgboost.pkl")
)
joblib.dump(
    label_encoder,
    os.path.join(MODEL_DIR, "target_encoder.pkl")
)

# Save evaluation metrics for display/reference.
metrics = pd.DataFrame([{
    "accuracy": acc,
    "precision_weighted": prec,
    "recall_weighted": rec,
    "f1_weighted": f1w,
    "f1_macro": f1m,
}])
metrics.to_csv(
    os.path.join(MODEL_DIR, "metrics.csv"),
    index=False
)

print("\nSaved:")
print("models/career_recommendation_xgboost.pkl")
print("models/target_encoder.pkl")
print("models/metrics.csv")
