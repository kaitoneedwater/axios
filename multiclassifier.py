from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from catboost import CatBoostClassifier
import xgboost as xgb
from tabpfn import TabPFNClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV
import numpy as np

class WeightedEns(BaseEstimator):
    def __init__(self):
        self.classifiers = [
            xgb.XGBClassifier(learning_rate=0.1, n_estimators=100, max_depth=3, tree_method='gpu_hist'),
            CatBoostClassifier(learning_rate=0.1, depth=5, iterations=100, task_type='GPU'),
            MLPClassifier(hidden_layer_sizes=(100,), activation='relu'),
            TabPFNClassifier(N_ensemble_configurations=128, device='cuda:0')
        ]
        self.imp = SimpleImputer(missing_values=np.nan, strategy='median')

    def fit(self, X, y):
        cls, y = np.unique(y, return_inverse=True)
        self.classes_ = cls
        X = self.imp.fit_transform(X)
        for cl in self.classifiers:
            cl.fit(X, y)

        # Hyperparameter tuning
        param_grid = {
            'xgboost__learning_rate': [0.01, 0.1, 0.2],
            'xgboost__max_depth': [3, 5, 7],
            'catboost__learning_rate': [0.01, 0.1, 0.2],
            'catboost__depth': [5, 7, 9],
            'mlpclassifier__hidden_layer_sizes': [(100,), (100, 50), (50, 50)],
            'tabpfn__N_ensemble_configurations': [64, 128, 256]
        }

        self.classifiers[0] = GridSearchCV(self.classifiers[0], param_grid={'learning_rate': param_grid['xgboost__learning_rate'], 'max_depth': param_grid['xgboost__max_depth']}, cv=3)
        self.classifiers[0].fit(X, y)
        self.classifiers[1] = GridSearchCV(self.classifiers[1], param_grid={'learning_rate': param_grid['catboost__learning_rate'], 'depth': param_grid['catboost__depth']}, cv=3)
        self.classifiers[1].fit(X, y)
        self.classifiers[2] = GridSearchCV(self.classifiers[2], param_grid={'hidden_layer_sizes': param_grid['mlpclassifier__hidden_layer_sizes']}, cv=3)
        self.classifiers[2].fit(X, y)
        self.classifiers[3] = GridSearchCV(self.classifiers[3], param_grid={'N_ensemble_configurations': param_grid['tabpfn__N_ensemble_configurations']}, cv=3)
        self.classifiers[3].fit(X, y)

    def predict_proba(self, X):
        X = self.imp.transform(X)
        ps = np.stack([cl.predict_proba(X) for cl in self.classifiers])
        p = np.mean(ps, axis=0)
        class_0_est_instances = p[:, 0].sum()
        others_est_instances = p[:, 1:].sum()

        new_p = p * np.array([[1 / (class_0_est_instances if i == 0 else others_est_instances) for i in range(p.shape[1])]])
        return new_p / np.sum(new_p, axis=1, keepdims=1)