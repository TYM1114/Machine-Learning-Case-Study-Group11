# Home Credit Default Risk - Kaggle Case Study

## 專案概述

本專案為 Kaggle [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) 競賽實作紀錄。目標是利用申請人的歷史信用與交易資料，預測其是否會違約，評估指標為 `ROC AUC`。



## 開發過程與細節

### `test.py`

- 輸出：`submission.csv`
- 架構：`LightGBM + XGBoost`
- 權重：`0.7 / 0.3`
- 定位：最早的 1st place 風格 baseline

### `test1_v2.py`

- 輸出：`submission1_v2.csv`
- 架構：`LightGBM + XGBoost`
- 權重：`0.7 / 0.3`
- 定位：以 `test.py` 為基礎，補強 1st place writeup 中明確提到的特徵
- 新增方向：
  - `EXT_SOURCE_3` 分母比值特徵
  - `REGION_ID_POPULATION` 類別化
  - `previous_application` 的 `last 3/5`、`first 2/4`
  - `installments` 的 `last 2/3/5`、`60/90/180/365 days`、`past due`、`NUM_INSTALMENT_NUMBER`
  - `POS / credit_card` 的 `last 3/5/10`

### `test2.py`

- 輸出：`submission2.csv`
- 架構：`LightGBM + XGBoost + CatBoost`
- 權重：`0.4 / 0.3 / 0.3`
- 定位：2nd place 風格版本
- 新增方向：
  - `CatBoost`
  - `interest rate` / `trend` 類特徵

### `test3.py`

- 輸出：`submission3.csv`
- 架構：`LightGBM + XGBoost + CatBoost`
- 權重：OOF 自動搜尋
- 定位：1st + 2nd place 融合版
- 主要特點：
  - `safe_divide`
  - 更完整的 recent window features
  - OOF 選 ensemble 權重

### `test4.py`

- 輸出：`submission4.csv`
- 架構：`LightGBM + XGBoost + CatBoost`
- 權重：OOF 自動搜尋
- 定位：目前主力版本
- 主要新增：
  - `bureau / POS / installments / credit_card` 最近 `1/3/5/10` 筆特徵
  - `previous_application` 的 `last 3/5`、`first 2/4`
  - `installments past due` 與更多 slice features
  - LightGBM importance-based feature selection
  - `XGBoost / CatBoost` GPU fallback

### `test5.py`

- 輸出：`submission5.csv`
- 架構：`LightGBM + XGBoost + CatBoost`
- 權重：OOF 自動搜尋
- 定位：以 `test4.py` 為基礎，加入 `EXT_SOURCE prediction / residual features`
- 主要新增：
  - 用 application 端欄位預測 `EXT_SOURCE_1/2/3`
  - 加入 `APP_EXT_SOURCE_*_PRED`
  - 加入 `APP_EXT_SOURCE_*_DIFF`
  - 加入 `APP_EXT_SOURCE_*_MISSING`
- 加入 `APP_EXT_SOURCE_PRED_MEAN / STD`
- 加入 `APP_EXT_SOURCE_DIFF_ABS_SUM / STD`
- 狀態：較test4差

### `test6.py`

- 輸出：`submission6.csv`
- 架構：`LightGBM + XGBoost + CatBoost + level-2 stacking + constrained rank blend`
- 權重：OOF 自動選擇 stack / blend
- 定位：根據 `note/home-credit-default-risk-v6.ipynb` 思路重建的全新主線
- 主要新增：
  - 更完整的 `application` 交互特徵與 `GP-style` 特徵
  - `bureau / previous / POS / credit_card / installments` 的 `time window / time-weighted / trend / per-loan / last-record` 特徵
  - `previous / bureau / installments` row-level sub-model features
  - frequency encoding / groupby ratio features / OOF target encoding
  - `KNN_TARGET_200 / 500`
  - null-importance feature selection
  - `LGB seed averaging + logistic stacking + constrained rank blend`
- 狀態：已完成提交驗證，為目前最佳單一路線

### `blend.py`

- 輸出：`submission_blend.csv`
- 功能：將 `submission.csv` 與 `submission2.csv` 做 rank blend
- 設定：
  - `submission.csv`: `0.85`
  - `submission2.csv`: `0.15`

### `blend_v2.py`

- 輸出：
  - `submission_blend_v2_a.csv`
  - `submission_blend_v2_b.csv`
  - `submission_blend_v2_c.csv`
  - `submission_blend_v2_d.csv`
  - `submission_blend_v2_raw.csv`
- 功能：
  - 分析 `submission1_v2 / submission3 / submission4 / submission_blend` 的 prediction correlation
  - 產出幾個以 `submission4` 為主、搭配低相關輔助線的 blend 候選

### `blend_v3.py`

- 輸出：
  - `submission_blend_v3_a.csv`
  - `submission_blend_v3_b.csv`
  - `submission_blend_v3_c.csv`
  - `submission_blend_v3_d.csv`
  - `submission_blend_v3_e.csv`
  - `submission_blend_v3_f.csv`
- 功能：
  - 以 `submission6.csv` 為主體，分析與舊主線 submission 的 prediction correlation
  - 產出幾個以 `submission6` 為核心、搭配舊 submission 的 raw / rank blend 候選

## Kaggle 提交成績

| Submission | Public Score | Private Score | 說明 |
| --- | ---: | ---: | --- |
| `submission.csv` | 0.79791 | 0.79181 | `test.py` 輸出 |
| `submission1_v2.csv` | 0.79763 | 0.79385 | `test1_v2.py` 輸出，強化版 1st place baseline |
| `submission2.csv` | 0.79554 | 0.79267 | `test2.py` 輸出 |
| `submission3.csv` | 0.79792 | 0.79368 | `test3.py` 輸出 |
| `submission4.csv` | 0.79790 | 0.79506 | `test4.py` 輸出，private 目前最佳 |
| `submission_blend.csv` | 0.79830 | 0.79268 | `blend.py` 輸出，public 目前最佳 |
| `submission_blend_v2_a.csv` | 0.79864 | 0.79533 | `blend_v2.py` 候選，rank blend |
| `submission_blend_v2_b.csv` | 0.79892 | 0.79545 | `blend_v2.py` 候選，rank blend |
| `submission_blend_v2_c.csv` | 0.79904 | 0.79540 | `blend_v2.py` 候選，rank blend |
| `submission_blend_v2_d.csv` | 0.79917 | 0.79553 | `blend_v2.py` 候選，rank blend，public 目前最佳 |
| `submission_blend_v2_raw.csv` | 0.79886 | 0.79554 | `blend_v2.py` 候選，raw blend，private 目前最佳 |
| `submission5.csv` | 0.79721 | 0.79487 | `test5.py` 輸出，加入 `EXT_SOURCE` prediction / residual features，但未優於 `test4.py` |
| `submission6.csv` | 0.80543 | 0.80174 | `test6.py` 輸出，根據 `v6` notebook 重建的新主線，目前整體最佳 |
| `submission_blend_v3_a.csv` | 0.80577 | 0.80181 | `blend_v3.py` 候選，`submission6 + submission_blend` rank blend |
| `submission_blend_v3_b.csv` | 0.80587 | 0.80185 | `blend_v3.py` 候選，public 目前最佳 |
| `submission_blend_v3_c.csv` | 0.80573 | 0.80202 | `blend_v3.py` 候選，raw blend |
| `submission_blend_v3_d.csv` | 0.80548 | 0.80216 | `blend_v3.py` 候選，raw blend，private 目前最佳 |
| `submission_blend_v3_e.csv` | 0.80572 | 0.80168 | `blend_v3.py` 候選，rank blend |
| `submission_blend_v3_f.csv` | 0.80565 | 0.80191 | `blend_v3.py` 候選，raw blend |

## 最新進度

- `test1_v2.py` 已驗證：純 1st place 強化線是有效的
  - 相較 `submission.csv`
  - Private: `0.79181 -> 0.79385`
  - Public: `0.79791 -> 0.79763`

- `test4.py` 已驗證：混合 1st / 2nd / 3rd / 4th place 思路後，private 明顯提升
  - 相較 `submission3.csv`
  - Private: `0.79368 -> 0.79506`
  - Public: `0.79792 -> 0.79790`

- `blend_v2.py` 已驗證：以 `submission4` 為主，搭配低相關 submission 做 blend 是有效的
  - `submission_blend_v2_raw.csv` 取得目前最高 private：`0.79554`
  - `submission_blend_v2_d.csv` 取得目前最高 public：`0.79917`
  - `raw blend` 這次略優於 `rank blend`，表示原始 prediction scale 仍保有可用訊號

- `test5.py` 已驗證：目前這版 `EXT_SOURCE prediction / residual features` 沒有帶來增益
  - 相較 `submission4.csv`
  - Private: `0.79506 -> 0.79487`
  - Public: `0.79790 -> 0.79721`
  - Full CV: `0.796940 -> 0.796719`
  - 結論：這條實作路線目前不如 `test4.py`

- `test6.py` 已驗證：根據 `v6` notebook 重建的新路線明顯優於既有主線
  - 相較 `submission_blend_v2_raw.csv`
  - Private: `0.79554 -> 0.80174`
  - Public: `0.79886 -> 0.80543`
  - 結論：`v6` 這條重型特徵工程 + sub-model + encoding + KNN + stack/blend 路線成立，已取代 `test4.py / blend_v2.py` 成為主線

- `blend_v3.py` 已驗證：以 `submission6` 為主、混入少量舊主線輸出仍可再提升
  - 相較 `submission6.csv`
  - Private 最佳：`0.80174 -> 0.80216` (`submission_blend_v3_d.csv`)
  - Public 最佳：`0.80543 -> 0.80587` (`submission_blend_v3_b.csv`)
  - 結論：`submission6` 雖然已很強，但和舊 submission 做小比例 blend 仍保有額外訊號

- 目前最佳版本：
  - 若以 `Private Score` 為主：`submission_blend_v3_d.csv`
  - 若以 `Public Score` 為主：`submission_blend_v3_b.csv`

- 目前最合理的主線：
  - 保留 `test1_v2.py` 作為 1st place 強化 baseline
  - 以 `test6.py` 作為目前主力模型來源
  - 以 `blend_v3.py` 作為目前主力融合路線
  - `test4.py / blend_v2.py` 保留作為舊主線與對照組
  - `test5.py` 保留作為已驗證支線，但暫時不作為主力方向
  - 後續若要再提升，優先沿 `submission6 + 舊主線低相關輸出` 做小範圍權重微調

## 目前觀察

1. `recent 1/3/5/10 records` 類特徵是有效的，這是 `test4.py` 明顯勝過 `test3.py` 的主因之一。
2. feature selection 是有效的，`test4.py` 將特徵從 `3160` 篩到 `901` 後仍提升 private score。
3. `LightGBM` 目前在本機不是 CUDA build，實際執行時會 fallback 到 CPU。
4. `XGBoost` 與 `CatBoost` 目前有跑到 GPU。
5. `submission_blend.csv` 雖然 private 不強，但和主線模型的 prediction correlation 明顯較低，適合拿來做輔助 blend。
6. `blend_v2` 結果顯示，這一輪 `raw blend` 比 `rank blend` 略強，後續值得優先在 raw 權重附近微調。
7. 目前這版 `EXT_SOURCE` 預測 / 殘差特徵沒有產生額外增益，至少在現有實作下不值得優先深挖。
8. `test6.py` 顯示，單純沿 `test4.py` 小修小補的上限有限；改走更完整的 `v6` 路線後，private / public 都有結構性提升。
9. row-level sub-model features、encoding feature block、KNN target features 與 level-2 ensemble 的組合，至少在這次提交中明顯比舊版簡單 OOF blend 更有效。
10. `blend_v3` 結果顯示，`submission6` 與舊主線 submission 之間仍有可用差異訊號，且這一輪 `raw blend` 依然比多數 `rank blend` 更強。

## 執行方式

### 安裝依賴

```bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost
```


### 對應輸出

- `python test.py` -> `submission.csv`
- `python test1_v2.py` -> `submission1_v2.csv`
- `python test2.py` -> `submission2.csv`
- `python test3.py` -> `submission3.csv`
- `python test4.py` -> `submission4.csv`
- `python test5.py` -> `submission5.csv`
- `python test6.py` -> `submission6.csv`
- `python blend.py` -> `submission_blend.csv`
- `python blend_v2.py` -> 多個 `submission_blend_v2_*.csv`
- `python blend_v3.py` -> 多個 `submission_blend_v3_*.csv`

## 硬體與執行注意事項

- `XGBoost` 預設使用 GPU (`device="cuda"`)
- `CatBoost` 預設使用 GPU (`task_type="GPU"`)
- `LightGBM` 會優先嘗試 CUDA；若安裝版本不支援 GPU，腳本會 fallback 到 CPU
- 全量資料與多表聚合特徵吃記憶體，建議在 RAM 較充足的環境執行

## 參考文件

- 詳細的 1st / 2nd / 3rd / 4th place 技術整理請見 [reference.md]
