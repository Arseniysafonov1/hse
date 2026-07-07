from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC
from category_encoders import OneHotEncoder
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
import numpy as np



def gini(y_true, y_score):
    return 2 * roc_auc_score(y_true, y_score) - 1.0

class PipeLine:
    def __init__(self, loss='hinge', penalty='l2', alpha=0.0001, max_iter=1000, n_splits=5, one_hot_encode_features=[], scaled_cols=[], missing_cols=[]):
        self.loss = loss
        self.model = SGDClassifier(loss=loss, penalty=penalty, alpha=alpha, max_iter=max_iter)
        self.encoder = OneHotEncoder(cols=one_hot_encode_features)
        self.tscv = TimeSeriesSplit(n_splits=n_splits,)
        self.scaled_cols = scaled_cols
        self.missing_cols = missing_cols

    def transform(self, X):
        X = self.encoder.transform(X)
        for i in self.missing_cols:
            X[i + '_missing'] = X[i].isna().astype(np.int64)
            X[i] = X[i].fillna(0)
        for i in self.scaled_cols:
            X[i] = (X[i] + 1) ** (-1)
        return X

    def fit(self, X, y):
        print('Transform started')
        self.encoder.fit(X)
        X = self.transform(X)
        print('Transform ended')
        print('Start of train:')
        for i, (train_ids, test_ids) in enumerate(self.tscv.split(X)):
            print('Fold', i, ':')
            
            X_train, X_test = X.iloc[train_ids], X.iloc[test_ids]
            y_train, y_test = y.iloc[train_ids], y.iloc[test_ids]
            
            self.model.fit(X_train, y_train)

            if self.loss == 'log_loss':
                gini_score = gini(y_test, self.model.predict_proba(X_test)[:, 1])
                print('Gini:', gini_score)
            acc_score = accuracy_score(y_test, self.model.predict(X_test))
            print('Accuracy:', acc_score)

    def predict(self, X):
        X = self.transform(X)
        return self.model.predict(X)

    def predict_proba(self, X):
        X = self.transform(X)
        return self.model.predict_proba(X)