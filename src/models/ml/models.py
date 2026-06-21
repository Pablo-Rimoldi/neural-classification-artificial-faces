"""Leakage-safe ML pipelines and hyperparameter grids.

Ported verbatim from notebook cell 54 ("Stage 1 - 8 models"). Pipelines all
share the same leakage-safe skeleton (per-subject z-score -> scaler ->
variance filter -> univariate selector -> classifier) and every grid sweeps
the `scaler` and `selector__k` choices alongside the classifier's own
hyperparameters.
"""
import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.metrics import make_scorer, accuracy_score, matthews_corrcoef
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from src.models.ml.transforms import IntraSubjectZScore
from src import config

CALIB_CV = 3


def safe_mcc(y_true, y_pred):
    if len(np.unique(y_pred)) <= 1:
        return 0.0
    return matthews_corrcoef(y_true, y_pred)


acc_scorer = make_scorer(accuracy_score)


def _pipe(clf):
    return Pipeline([
        ('subjz',    IntraSubjectZScore()),
        ('scaler',   StandardScaler()),
        ('variance', VarianceThreshold(threshold=0.0)),
        ('selector', SelectKBest(score_func=f_classif)),
        ('clf',      clf),
    ])


def _cal(estimator, method='sigmoid'):
    """Wraps a margin classifier in CalibratedClassifierCV."""
    return CalibratedClassifierCV(estimator, method=method, cv=CALIB_CV)


def get_models_and_grids():
    """Stage 1 - 8 models; every grid includes `scaler` and `selector__k`."""
    return [
        ('LDA', _pipe(LinearDiscriminantAnalysis(solver='lsqr')),
         {'clf__shrinkage': ['auto', 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
          'scaler': config.SCALER_CHOICES,
          'selector__k': config.K_CANDIDATES}, 40),


        ('LinearSVC', _pipe(LinearSVC(dual='auto', max_iter=5000, random_state=config.RANDOM_STATE)),
         {'clf__C': loguniform(1e-2, 3e1),
          'clf__class_weight': [None, 'balanced'],
          'scaler': config.SCALER_CHOICES,
          'selector__k': config.K_CANDIDATES}, 45),


        ('LinearSVC_Cal',
         _pipe(_cal(LinearSVC(dual='auto', max_iter=5000, random_state=config.RANDOM_STATE))),
         {'clf__estimator__C': loguniform(5e-2, 2e1),
          'clf__estimator__class_weight': [None, 'balanced'],
          'scaler': config.SCALER_CHOICES,
          'selector__k': config.K_CANDIDATES}, 45),

#        ('SVC_RBF', _pipe(SVC(kernel='rbf', cache_size=500, probability=False,
#                              random_state=config.RANDOM_STATE)),
#         {'clf__C': loguniform(1e-1, 1e2),
#          'clf__gamma': loguniform(1e-4, 1e0),
#          'clf__class_weight': [None, 'balanced'],
#          'scaler': config.SCALER_CHOICES,
#          'selector__k': config.K_CANDIDATES}, 60),

        ('LogReg_L1', _pipe(LogisticRegression(solver='saga', penalty='l1',
                                               max_iter=5000, random_state=config.RANDOM_STATE)),
         {'clf__C': loguniform(1e-2, 1e1),
          'clf__class_weight': [None, 'balanced'],
          'scaler': config.SCALER_CHOICES,
          'selector__k': config.K_CANDIDATES}, 45),

        ('LogReg_EN', _pipe(LogisticRegression(solver='saga', penalty='elasticnet',
                                               max_iter=5000, random_state=config.RANDOM_STATE)),
         {'clf__C': loguniform(1e-2, 1e1),
          'clf__l1_ratio': uniform(0.0, 1.0),
          'clf__class_weight': [None, 'balanced'],
          'scaler': config.SCALER_CHOICES,
          'selector__k': config.K_CANDIDATES}, 60),

        ('SGD', _pipe(SGDClassifier(max_iter=2000, tol=1e-3, random_state=config.RANDOM_STATE)),
         {'clf__loss': ['hinge', 'log_loss', 'modified_huber'],
          'clf__penalty': ['l1', 'l2', 'elasticnet'],
          'clf__alpha': loguniform(1e-5, 1e-1),
          'clf__l1_ratio': uniform(0.0, 1.0),
          'clf__class_weight': [None, 'balanced'],
          'scaler': config.SCALER_CHOICES,
          'selector__k': config.K_CANDIDATES}, 60),

#        ('RandomForest', _pipe(RandomForestClassifier(random_state=config.RANDOM_STATE, n_jobs=1)),
#         {'clf__n_estimators': randint(150, 600),
#          'clf__max_depth': [2, 3, 4, 5, None],
#          'clf__min_samples_leaf': randint(1, 8),
#          'clf__max_features': ['sqrt', 'log2', None],
#          'clf__class_weight': ['balanced', 'balanced_subsample'],
#          'scaler': config.SCALER_CHOICES,
#          'selector__k': config.K_CANDIDATES}, 60),

        ('XGBoost', _pipe(XGBClassifier(eval_metric='logloss', tree_method='hist',
                                        random_state=config.RANDOM_STATE, n_jobs=1)),
         {'clf__n_estimators': randint(100, 400),
          'clf__max_depth': randint(2, 6),
          'clf__learning_rate': loguniform(1e-2, 3e-1),
          'clf__subsample': uniform(0.6, 0.4),
          'clf__colsample_bytree': uniform(0.5, 0.5),
          'clf__gamma': uniform(0.0, 5.0),
          'clf__reg_lambda': loguniform(1e-2, 1e1),
          'scaler': config.SCALER_CHOICES,
          'selector__k': config.K_CANDIDATES}, 60),

        ('Dummy', _pipe(DummyClassifier(strategy='stratified', random_state=config.RANDOM_STATE)),
         {'clf__strategy': ['stratified', 'most_frequent', 'uniform', 'prior']}, 4),
    ]


def get_focused_grid(name):
    s = config.SCALER_CHOICES
    k = config.K_CANDIDATES
    grids = {
        'LDA': ({'clf__shrinkage': ['auto', None] + list(np.round(np.linspace(0, 1, 21), 3)),
                 'clf__solver': ['lsqr', 'eigen'],
                 'scaler': s, 'selector__k': k}, 80),

        'LinearSVC': ({'clf__C': loguniform(1e-3, 1e2),
                       'clf__loss': ['hinge', 'squared_hinge'],
                       'clf__class_weight': [None, 'balanced'],
                       'scaler': s, 'selector__k': k}, 150),

        'LinearSVC_Cal': ({'clf__estimator__C': loguniform(1e-3, 1e2),
                           'clf__estimator__loss': ['hinge', 'squared_hinge'],
                           'clf__estimator__class_weight': [None, 'balanced'],
                           'clf__method': ['sigmoid', 'isotonic'],
                           'scaler': s, 'selector__k': k}, 150),

#        'SVC_RBF': ({'clf__C': loguniform(1e-2, 1e3),
#                     'clf__gamma': loguniform(1e-5, 1e1),
#                     'clf__class_weight': [None, 'balanced'],
#                     'scaler': s, 'selector__k': k}, 150),

        'LogReg_L1': ({'clf__C': loguniform(1e-3, 1e2),
                       'clf__class_weight': [None, 'balanced'],
                       'scaler': s, 'selector__k': k}, 120),

        'LogReg_EN': ({'clf__C': loguniform(1e-3, 1e2),
                       'clf__l1_ratio': uniform(0.0, 1.0),
                       'clf__class_weight': [None, 'balanced'],
                       'scaler': s, 'selector__k': k}, 150),

        'SGD': ({'clf__loss': ['hinge', 'log_loss', 'modified_huber', 'squared_hinge'],
                 'clf__penalty': ['l1', 'l2', 'elasticnet'],
                 'clf__alpha': loguniform(1e-6, 1e0),
                 'clf__l1_ratio': uniform(0.0, 1.0),
                 'clf__class_weight': [None, 'balanced'],
                 'scaler': s, 'selector__k': k}, 150),

#        'RandomForest': ({'clf__n_estimators': randint(100, 800),
#                          'clf__max_depth': [2, 3, 4, 5, 6, 8, None],
#                          'clf__min_samples_leaf': randint(1, 12),
#                          'clf__min_samples_split': randint(2, 12),
#                          'clf__max_features': ['sqrt', 'log2', None, 0.5],
#                          'clf__class_weight': [None, 'balanced', 'balanced_subsample'],
#                          'scaler': s, 'selector__k': k}, 150),

        'XGBoost': ({'clf__n_estimators': randint(100, 600),
                     'clf__max_depth': randint(2, 8),
                     'clf__learning_rate': loguniform(5e-3, 3e-1),
                     'clf__subsample': uniform(0.5, 0.5),
                     'clf__colsample_bytree': uniform(0.4, 0.6),
                     'clf__gamma': uniform(0.0, 6.0),
                     'clf__reg_lambda': loguniform(1e-2, 1e2),
                     'clf__reg_alpha': loguniform(1e-3, 1e1),
                     'clf__min_child_weight': randint(1, 8),
                     'scaler': s, 'selector__k': k}, 150),

        'Dummy': ({'clf__strategy': ['stratified', 'most_frequent', 'uniform', 'prior']}, 4),
    }
    return grids[name]
