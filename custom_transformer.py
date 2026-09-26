
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer

class MultiLabelBinarizerTransformer(BaseEstimator, TransformerMixin):
    """Convert one comma-separated text column into binary indicator features."""

    def __init__(self):
        self.mlb = MultiLabelBinarizer()

    def _split_column(self, X):
        values = np.asarray(X).ravel()
        return [
            [t.strip() for t in str(v).split(",") if t.strip()]
            for v in values
        ]

    def fit(self, X, y=None):
        self.mlb.fit(self._split_column(X))
        return self

    def transform(self, X):
        return self.mlb.transform(self._split_column(X))

    def get_feature_names_out(self, input_features=None):
        prefix = (
            input_features[0]
            if input_features is not None and len(input_features)
            else "tag"
        )
        return np.array(
            [f"{prefix}__{tag}" for tag in self.mlb.classes_],
            dtype=object
        )
