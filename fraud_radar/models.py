"""The four models, behind one interface.

Each is wrapped in a scikit-learn Pipeline that carries its own preprocessing.
That is deliberate: the API in Phase 2 loads one artifact and calls it. It
cannot forget to scale a feature or encode a category, because the scaler and
the encoder travel inside the saved object.
"""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

RANDOM_STATE = 42


def make_preprocessor() -> ColumnTransformer:
    """Impute, scale, and one-hot encode.

    The history features are genuinely missing on a card's first ever
    transaction — there is no previous one to compare against. Median
    imputation says 'assume typical' rather than dropping those rows.
    """
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )


class AutoencoderScorer:
    """An autoencoder built on scikit-learn's MLPRegressor.

    The network is trained to reproduce its own input. Trained only on
    legitimate transactions, it gets good at reconstructing normal behaviour and
    stays bad at reconstructing anything unusual — so the reconstruction error
    itself becomes the anomaly score. No labels are used in training.

    MLPRegressor rather than PyTorch keeps the serving image small; the concept
    and the scoring rule are identical, and this class is the only thing that
    would need replacing to swap in a torch model later.
    """

    def __init__(self, hidden=(24, 8, 24), max_iter=40):
        self.net = MLPRegressor(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            max_iter=max_iter,
            batch_size=1024,
            random_state=RANDOM_STATE,
            early_stopping=True,
            n_iter_no_change=5,
        )

    def fit(self, X, y=None):
        self.net.fit(X, X)
        return self

    def score_samples(self, X):
        """Higher = more anomalous, matching the convention used elsewhere."""
        reconstructed = self.net.predict(X)
        return ((X - reconstructed) ** 2).mean(axis=1)


def build_models() -> dict:
    """The four candidates.

    Two unsupervised models that learn 'normal' without labels, and two
    supervised baselines they have to justify themselves against.
    """
    return {
        "isolation_forest": {
            "estimator": IsolationForest(
                n_estimators=200,
                contamination=0.01,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "supervised": False,
        },
        "autoencoder": {
            "estimator": AutoencoderScorer(),
            "supervised": False,
        },
        "logistic_regression": {
            "estimator": LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            "supervised": True,
        },
        "gradient_boosting": {
            "estimator": HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.1,
                random_state=RANDOM_STATE,
            ),
            "supervised": True,
        },
    }
