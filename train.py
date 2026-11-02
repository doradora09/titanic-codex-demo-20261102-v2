"""Titanic demo: fixed candidate CV, untouched holdout, then one Kaggle submission.

Run: .venv/bin/python train.py
Download train.csv and test.csv from the official competition into data/ first.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
SEED = 20261102
for d in ['evidence', 'publish']:
    (ROOT / d).mkdir(exist_ok=True)
train = pd.read_csv(ROOT / 'data/train.csv')
test = pd.read_csv(ROOT / 'data/test.csv')
assert train.shape == (891, 12) and test.shape == (418, 11)
assert train.PassengerId.is_unique and test.PassengerId.is_unique
assert not set(train.PassengerId) & set(test.PassengerId)
assert 'Survived' not in test and train.Survived.isin([0, 1]).all()


def features(frame):
    out = frame.copy()
    out['FamilySize'] = out.SibSp + out.Parch + 1
    out['IsAlone'] = (out.FamilySize == 1).astype(int)
    out['FarePerPerson'] = out.Fare / out.FamilySize
    title = out.Name.str.extract(r',\s*([^.]*)\.')[0].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
    out['Title'] = title.where(title.isin(['Mr', 'Miss', 'Mrs', 'Master']), 'Rare')
    out['Deck'] = out.Cabin.str[0].fillna('Unknown')
    return out


numeric = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone', 'FarePerPerson']
categorical = ['Sex', 'Embarked', 'Title', 'Deck']


def preprocessing(scale=False, ordinal=False):
    nums = make_pipeline(SimpleImputer(strategy='median'), StandardScaler()) if scale else SimpleImputer(strategy='median')
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1) if ordinal else OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    return ColumnTransformer([
        ('numeric', nums, numeric),
        ('category', make_pipeline(SimpleImputer(strategy='most_frequent'), encoder), categorical),
    ])


X, X_test = features(train), features(test)
y = train.Survived.to_numpy()
development, holdout = train_test_split(np.arange(len(train)), test_size=.20, stratify=y, random_state=SEED)
X_dev, y_dev = X.iloc[development], y[development]
folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X_dev, y_dev))
models = {
    'Majority baseline': make_pipeline(preprocessing(), DummyClassifier(strategy='most_frequent')),
    'Logistic regression': make_pipeline(preprocessing(scale=True), LogisticRegression(C=1, max_iter=2000)),
    'Random forest': make_pipeline(preprocessing(), RandomForestClassifier(n_estimators=400, max_depth=7, min_samples_leaf=3, max_features=.8, random_state=SEED, n_jobs=2)),
    'Extra trees': make_pipeline(preprocessing(), ExtraTreesClassifier(n_estimators=400, max_depth=9, min_samples_leaf=3, max_features=.8, random_state=SEED, n_jobs=2)),
    'Hist gradient boosting': make_pipeline(preprocessing(ordinal=True), HistGradientBoostingClassifier(max_iter=180, learning_rate=.05, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=2, early_stopping=False, random_state=SEED)),
}
results = []
oof = pd.DataFrame({'PassengerId': train.iloc[development].PassengerId.to_numpy(), 'observed': y_dev})
fold_ids = np.empty(len(development), dtype=int)
for fold, (_, valid) in enumerate(folds):
    fold_ids[valid] = fold
oof['fold'] = fold_ids
for name, estimator in models.items():
    probabilities = np.zeros(len(development))
    scores = []
    for fitting, valid in folds:
        fitted = clone(estimator).fit(X_dev.iloc[fitting], y_dev[fitting])
        prob = fitted.predict_proba(X_dev.iloc[valid])[:, 1]
        probabilities[valid] = prob
        scores.append(float(accuracy_score(y_dev[valid], prob >= .5)))
    row = {'model': name, 'fold_accuracy': scores, 'cv_mean': float(np.mean(scores)), 'cv_sd': float(np.std(scores, ddof=1)), 'oof_accuracy': float(accuracy_score(y_dev, probabilities >= .5)), 'oof_auc': float(roc_auc_score(y_dev, probabilities))}
    results.append(row)
    oof[name] = probabilities
    print(f'{name}: CV {row["cv_mean"]:.5f} (SD {row["cv_sd"]:.5f})', flush=True)

# Select solely on development CV; access the holdout only after fixing the choice.
best = max((r for r in results if r['model'] != 'Majority baseline'), key=lambda r: r['cv_mean'])
selected = clone(models[best['model']]).fit(X_dev, y_dev)
holdout_prob = selected.predict_proba(X.iloc[holdout])[:, 1]
holdout_pred = (holdout_prob >= .5).astype(int)
holdout_accuracy = float(accuracy_score(y[holdout], holdout_pred))
majority = int(pd.Series(y_dev).mode().iloc[0])
baseline_accuracy = float(accuracy_score(y[holdout], np.full(len(holdout), majority)))
final_model = clone(models[best['model']]).fit(X, y)
pred = final_model.predict(X_test).astype(int)
submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': pred})
assert list(submission.columns) == ['PassengerId', 'Survived']
assert len(submission) == 418 and not submission.isna().any().any()
assert submission.PassengerId.equals(test.PassengerId) and submission.Survived.isin([0, 1]).all()
submission_path = ROOT / 'submission_codex_20261102_v2.csv'
submission.to_csv(submission_path, index=False)
oof.to_csv(ROOT / 'evidence/development_oof.csv', index=False)
pd.DataFrame({'PassengerId': train.iloc[holdout].PassengerId.to_numpy(), 'observed': y[holdout], 'probability': holdout_prob, 'prediction': holdout_pred}).to_csv(ROOT / 'evidence/holdout_predictions.csv', index=False)
summary = {
    'run_at': datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(), 'seed': SEED,
    'train_rows': len(train), 'test_rows': len(test), 'development_rows': len(development), 'holdout_rows': len(holdout),
    'survived': int(y.sum()), 'missing_train': train.isna().sum().to_dict(), 'missing_test': test.isna().sum().to_dict(),
    'duplicate_rows': int(train.duplicated().sum()), 'features': numeric + categorical,
    'models': results, 'selected_model': best['model'], 'cv_mean': best['cv_mean'],
    'holdout_accuracy': holdout_accuracy, 'holdout_correct': int((holdout_pred == y[holdout]).sum()),
    'holdout_baseline_accuracy': baseline_accuracy, 'holdout_confusion': confusion_matrix(y[holdout], holdout_pred).tolist(),
    'holdout_auc': float(roc_auc_score(y[holdout], holdout_prob)),
    'predicted_survivors': int(pred.sum()), 'submission_file': submission_path.name,
    'submission_sha256': hashlib.sha256(submission_path.read_bytes()).hexdigest(),
    'selection_rule': 'Highest 5-fold mean accuracy on development set only; untouched 20% holdout evaluated once after selection; final fit on all 891 rows.',
    'python': platform.python_version(), 'sklearn': sklearn.__version__, 'kaggle_public_score': None,
    'survival_by_sex': train.groupby('Sex').Survived.agg(['sum', 'count', 'mean']).reset_index().to_dict(orient='records'),
    'survival_by_class': train.groupby('Pclass').Survived.agg(['sum', 'count', 'mean']).reset_index().to_dict(orient='records'),
}
(ROOT / 'evidence/analysis.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
pd.DataFrame([{k: v for k, v in r.items() if k != 'fold_accuracy'} for r in results]).to_csv(ROOT / 'evidence/model_comparison.csv', index=False)
print(f'SELECTED {best["model"]}; HOLDOUT {holdout_accuracy:.5f}; OUTPUT {submission_path.name}', flush=True)
