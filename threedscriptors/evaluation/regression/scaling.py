from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MaxAbsScaler, Normalizer, StandardScaler


class NoOpTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


class AutoScaler(BaseEstimator, TransformerMixin):
    """
    strategy: "none" | "standard" | "l2"
      - standard: StandardScaler (dense) / MaxAbsScaler (sparse)
      - l2: Normalizer(norm="l2")
    """

    def __init__(self, strategy="standard", with_mean=True):
        self.strategy = strategy
        self.with_mean = with_mean
        self._t = None

    def fit(self, X, y=None):
        if self.strategy == "none":
            self._t = NoOpTransformer()
        elif self.strategy == "l2":
            self._t = Normalizer(norm="l2")
            self._t.fit(X, y)
        else:
            self._t = (
                MaxAbsScaler()
                if sparse.issparse(X)
                else StandardScaler(with_mean=self.with_mean)
            )
            self._t.fit(X, y)
        return self

    def transform(self, X):
        return X if isinstance(self._t, NoOpTransformer) else self._t.transform(X)
