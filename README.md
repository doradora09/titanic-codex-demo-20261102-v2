# Titanic × Codex Computer Use — 新規実験

2026-11-02。公式データの取得からPythonの学習、Kaggle提出、レポート作成、GitHub Pages公開までを行う勉強会デモです。

[分析レポートを開く](https://doradora09.github.io/titanic-codex-demo-20261102-v2/?view=1)

## 結果

| 評価 | 正解率 | 対象 |
|---|---:|---|
| 開発データの5分割CV平均 | 83.15% | 712件（モデル選択に使用） |
| 未使用holdout | 87.15% | 179件中156件正解 |
| Kaggle公開スコア | **0.76315 (76.32%)** | 新規提出1回 |

採用モデルはRandom forest。5手法を同じ分割で比較し、開発CVだけで選択した後、保留データを一度だけ評価しました。最終提出用は全891件で再学習しています。seed=20261102。異なる評価対象と学習条件であり、これらを同じ精度の推定値として扱いません。公開スコアによる再調整はしていません。

- [Kaggle採点の画面](04-kaggle-score.jpg)
- [提出ファイルと説明](03-submission.jpg)
- [公式データ取得画面](01-kaggle-data.jpg)
- [実行中の進捗カンバン](02-progress.jpg)
- [全モデルの計算結果](analysis.json) / [比較CSV](model_comparison.csv)
- [取得元と入力SHA256](provenance.json) / [採点記録](kaggle-result.json)

## 再現

[Kaggle公式データ](https://www.kaggle.com/competitions/titanic/data)から train.csv と test.csv をダウンロードし、data/ に置いて実行します。元データは再配布していません。

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python train.py
```

提出ファイルは submission_codex_20261102_v2.csv。学習を再実行してもKaggleへは自動提出しません。

## 読み取り上の限界

乗客単位の分割のため、同じ家族・チケットの相関を排除していません。holdoutは179件と小さく、Kaggleとのスコア差の原因は特定していません。性別などと生存率の関連は因果効果を示しません。

ログイン済みセッションを利用したため、パスワード再入力の実演は行っていません。ブラウザー操作はCodex Computer Use、計算はローカルPythonを使用しました。

データ出典: Will Cukierski. [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic). Kaggle, 2012.
