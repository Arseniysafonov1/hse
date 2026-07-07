from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import roc_auc_score

from tqdm.auto import tqdm

from sklearn.base import ClassifierMixin

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder


class BoostingClassifier(ClassifierMixin):

    def __init__(
        self,
        base_model_class = DecisionTreeRegressor,
        base_model_params: dict | None = None,
        n_estimators: int = 20,
        learning_rate: float = 0.05,
        random_state: int | None = None,
        verbose: bool = True,
        early_stopping_rounds: int | None = 0,
        eval_metric: str | None = None,
        cat_features: Iterable | None = None
    ):
        super().__init__()

        self.base_model_class = base_model_class
        self.base_model_params = {} if base_model_params is None else base_model_params

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

        self.models = []
        self.gammas = []

        self.random_state = random_state  # не забудьте вставить его везде, где у вас возникает рандом
        self.verbose = verbose

        self.history = defaultdict(list)  # {"train_roc_auc": [], "train_loss": [], ...}

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.grad_fn = lambda y, z: -y * self.sigmoid(-y * z)

        self.curr_model_count = 0
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric

        self.cat_features = cat_features
        self.cat_info = dict()

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.models.append(self.base_model_class(random_state=self.random_state, **self.base_model_params))
        self.models[-1].fit(X, y)

    def _cat_fit(self, X: np.array, y: np.ndarray) -> None:
        if self.cat_features is None:
            return
        y = (y == 1)
        self._train_mean = y.mean()
        for feature in self.cat_features:
            le = LabelEncoder()
            cats = le.fit_transform(X[:, feature])
            class_cnt = len(le.classes_)
            self.cat_info[feature] = (np.bincount(cats, weights=y, minlength=class_cnt) / np.bincount(cats, minlength=class_cnt), le)

    def _cat_transform(self, X: np.array) -> np.ndarray:
        if self.cat_features is None:
            return X
        X = X.copy()
        df_len = X.shape[0]
        for feature in self.cat_features:
            valid_mask = np.isin(X[:, feature], self.cat_info[feature][1].classes_)
            labels = -np.ones(df_len, dtype=int)
            labels[valid_mask] = self.cat_info[feature][1].transform(X[valid_mask, feature])
            X[:, feature] = np.where(valid_mask, self.cat_info[feature][0][labels], self._train_mean)
        return X

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, eval_set: tuple[np.ndarray] | None = None, use_best_model: bool = False) -> None:
        train_predictions = np.zeros(X_train.shape[0])
        self.classes_ = np.unique(y_train)  # не рекомендуется убирать, нужно для калибровки
        estimator_range = range(self.n_estimators)
        if self.verbose:
            estimator_range = tqdm(estimator_range)

        use_early_stop = self.eval_metric is not None and self.early_stopping_rounds > 0 and eval_set is not None

        X_eval, y_eval = None, None

        self._cat_fit(X_train, y_train)
        X_train = self._cat_transform(X_train)

        if eval_set is not None:
            X_eval, y_eval = eval_set
            X_eval = self._cat_transform(X_eval)
            eval_quality = lambda y_pred: roc_auc_score(y_eval == 1, y_pred) if self.eval_metric == 'roc_auc' else -self.loss_fn(y_eval, y_pred)
            prev_quality_val = eval_quality(np.zeros(len(y_eval)))
            bad_rounds = 0
        
        for _ in estimator_range:
            y_pred = self.predict(X_train)
            grad = self.grad_fn(y_train, y_pred)
            self.partial_fit(X_train, -grad)
            new_model_pred = self.models[-1].predict(X_train)
            self.gammas.append(self._find_optimal_gamma(y_train, y_pred, new_model_pred))
            y_pred_new = y_pred + self.learning_rate * self.gammas[-1] * new_model_pred
            self.history['train_roc_auc'].append(roc_auc_score(y_train == 1, y_pred_new))
            self.history['train_loss'].append(self.loss_fn(y_train, y_pred_new))
            self.curr_model_count += 1
            if use_early_stop:
                curr_quality_val = eval_quality(self.predict(X_eval))
    
                if curr_quality_val < prev_quality_val:
                    bad_rounds += 1
                else:
                    bad_rounds = 0
    
                prev_quality_val = curr_quality_val
    
                if bad_rounds >= self.early_stopping_rounds:
                    break

        if use_early_stop and use_best_model:
            del self.models[-bad_rounds:]
            del self.gammas[-bad_rounds:]

        # чтобы было удобнее смотреть
        for key in self.history:
            self.history[key] = np.array(self.history[key])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        prob = self.sigmoid(self.predict(X))
        return np.column_stack([1 - prob, prob])

    def predict(self, X : np.ndarray) -> np.ndarray:
        X = self._cat_transform(X)
        res = np.zeros(X.shape[0])
        for i in range(self.curr_model_count):
            res += self.learning_rate * self.gammas[i] * self.models[i].predict(X)
        return res

    def _find_optimal_gamma(
        self,
        y: np.ndarray,
        old_predictions: np.ndarray, 
        new_predictions: np.ndarray
    ) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [
            self.loss_fn(y, old_predictions + gamma * new_predictions)
            for gamma in gammas
        ]
        return gammas[np.argmin(losses)]

    def plot_history(self, keys: str | Iterable[str]):
        if isinstance(keys, str):
            keys = [keys]
        fig, axes = plt.subplots(1, len(keys))
        x = np.arange(len(self.history[keys[0]]))
        for key, ax in zip(keys, axes):
            ax.plot(x, self.history[key])
            ax.set_xlabel('Round')
            ax.set_ylabel(key)
            ax.set_title(key + 'metric dynamics')
        plt.show()

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return roc_auc_score(y == 1, self.predict_proba(X)[:, 1])
