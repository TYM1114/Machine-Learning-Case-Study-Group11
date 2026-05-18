# Home Credit Default Risk References

## 說明

這份文件整理 Home Credit Default Risk 競賽前 4 名 solution 的技術重點，作為目前資料夾內 `test.py`、`test2.py`、`test3.py` 與後續 `test4.py` 改版的參考。

這一版內容來源包含：
## 官方連結

- 1st place: https://www.kaggle.com/competitions/home-credit-default-risk/writeups/home-aloan-1st-place-solution
- 2nd place: https://www.kaggle.com/competitions/home-credit-default-risk/writeups/ikiri-ds-2nd-place-solution-team-ikiri-ds
- 3rd place: https://kaggle.com/competitions/home-credit-default-risk/writeups/alijs-evgeny-3rd-place-solution
- 4th place: https://www.kaggle.com/competitions/home-credit-default-risk/writeups/quad-machine-4th-place-sharing-and-tips-about-havi

## 1st Place: Home Aloan

### 核心觀點

- 他們認為這題最重要的兩件事是：
  - 聰明而有效的特徵
  - 多樣化的 base algorithms
- 題目本質是高度異質、跨來源、跨時間的信用風控資料問題
- 最後成績不是靠單一模型，而是靠 feature diversity + model diversity

### 特徵工程主線

- 基礎骨架是很多公開 kernel 都有的多表聚合特徵
  - 將 many-to-one tables 聚合後接回 `application`
  - 這一層約有 `700` 個特徵

- `Ryan` 的做法：
  - 先做 `SK_ID_PREV`、`SK_ID_BUREAU` 的基本聚合
  - 在 `application_train.csv` 上做大量 division / subtraction features
  - 特別提到用 `EXT_SOURCE_3` 做分母的除法特徵，對 CV 與 LB 都有正向幫助
  - 對 application 端類別特徵做 label encoding
  - 對 `previous_application.csv` 中最後一次申請的類別特徵做 label encoding
  - 使用多種 slice aggregates：
    - `previous_application`: last `3 / 5`、first `2 / 4`
    - `installments_payments`: last `2 / 3 / 5`
    - `installments_payments`: `NUM_INSTALMENT_NUMBER = 1 / 2 / 3 / 4`
    - `installments_payments`: last `60 / 90 / 180 / 365` days
    - past-due installments 的聚合
    - `POS_CASH_balance`、`credit_card_balance` 採類似方法
  - `previous_application` 額外做 lag features，最遠到 last `5` applications

- `Olivier` 的做法：
  - 從 public kernel 延伸出 yearly interest rate
  - 這成為模型中分數最高的一批特徵

- `Phil` 提到的重要特徵：
  - `neighbors_target_mean_500`
  - `REGION_ID_POPULATION` 當類別特徵而不是數值
  - `debt_credit_ratio_None`
  - `credit_annuity_ratio`
  - 最近一次 previous application 的 `PRODUCT_COMBINATION`
  - `bureau` 的 `DAYS_CREDIT_mean`
  - `credit_goods_price_ratio`
  - active bureau loans 的最近 `DAYS_CREDIT`
  - `credit_downpayment`
  - `AGE_INT`
  - 最近 installments 的 payment ratio 類特徵
  - `annuity_to_max_installment_ratio`

- `Yang` 的做法：
  - 使用來自公開方案的 last `3 / 5 / 10` credit card、installment、POS 特徵
  - 改寫時間區間，讓特徵變異更大
  - 用 weighted mean，以時間當權重，建立 annuity / credit / payment 行為特徵
  - 建立由 income、payment、time 組合出的 KPI 類特徵

### 特徵選擇

- `Bojan` 用相對簡單但效果很好的方法做 feature reduction：
  - categorical 做 frequency encoding
  - numeric + encoded categorical 一起用 `Ridge regression` 做 forward feature selection
- 他把超過 `1600` 個特徵降到約 `240`
- 後來加入 Olivier 特徵後變成 `287`
- 這組 `287` 特徵可以做出約 `0.7985` CV，LB 約 `0.802 - 0.803`
- 最後隨著團隊更多 feature set 合併，superset 擴張到約 `1800 - 2000` features

### Base Models

- 全部 base models 都用 `5-fold StratifiedKFold`

- `Olivier`：
  - `LightGBM`
  - `FastRGF`
  - `FFM`，但效果不理想

- `Bojan`：
  - `XGBoost`
  - `LightGBM`
  - `CatBoost`
  - `Linear Regression`
  - `XGBoost` 多數用 `gpu_hist`
  - `LightGBM` 在 CPU 上訓練
  - `CatBoost` 單模較弱，但對 meta-feature diversity 有幫助

- `Ryan / Yang`：
  - 多個 `LightGBM`
  - 不同特徵集的組合模型

- `Michael`：
  - `DAE + NN`
  - 單純 NN 落後於樹模型，但在最後 blend 中有價值

### Ensemble / Stacking

- 整體使用 `3` 層 ensemble pipeline

- L1：
  - 累積 `90+` base predictions
  - stackers 包含：
    - `NN`
    - `XGBoost`
    - `LightGBM`
    - `Hill Climber linear model`

- L2：
  - `NN`
  - `ExtraTree`
  - `Hill Climber`
  - 並加入少量 raw feature restacking

- Final：
  - 將 3 個 L2 預測等權平均

- `Phil` 的 ExtraTrees stacker：
  - `max_depth=4`
  - `min_samples_leaf=1000`
  - 只用了 `7` 個 L2 模型與一個 raw feature `AMT_INCOME_TOTAL`

### 其他洞見

- 團隊沒有把大量時間花在 hyperparameter tuning
- 他們認為不同超參、不同模型、不同特徵集的 ensemble，足以補掉單模未極致調參的缺口
- 曾嘗試用模型補 `EXT_*` 缺值，但對 base models 沒有幫助
- 賽後回看發現，前三個 base models 都足以進 top 10，且簡單平均就可能夠拿第一
- 他們的結論非常明確：
  - feature engineering 最重要
  - feature selection 次之
  - ensemble 與 model diversity 是加成

### 對目前程式的對應

- [test.py](C:/Users/TYM/Downloads/case%20study/test.py:101) 已實作大量 application ratio features
- [test.py](C:/Users/TYM/Downloads/case%20study/test.py:127) 已實作 `neighbors_target_mean`
- [test.py](C:/Users/TYM/Downloads/case%20study/test.py:157) 到 [test.py](C:/Users/TYM/Downloads/case%20study/test.py:244) 已實作 bureau / previous / POS / installments / credit card 聚合
- [test.py](C:/Users/TYM/Downloads/case%20study/test.py:285) 使用 `LightGBM + XGBoost`

### 對後續的啟示

- `EXT_SOURCE` 相關特徵仍然值得繼續往下挖
- `last 3 / 5 / 10`、lag、slice aggregates 這一條支線值得補回來
- 強力 feature selection 很重要，不能只靠一直堆特徵

## 2nd Place: ikiri_DS

### 核心觀點

- 這不是單一路線解法，而是大型團隊協作下的多樣性工程
- 重點在：
  - feature diversity
  - model diversity
  - blending
  - train/test shift 管理

### 成員分工重點

- `ONODERA`
  - 專注 feature engineering

- `RK`
  - 各種 dimension reduction：`PCA`, `UMAP`, `T-SNE`, `LDA`
  - genetic programming features
  - 從約 `1TB` feature pool 做 brute-force feature search
  - 演算法比較與參數調整
  - blending 時使用直接 AUC 最大化方法

- `Yuya Yamamoto`
  - interest rate feature
  - train/test data difference
  - neural network residual correction

- `tosh`
  - neural networks

- `ireko8`
  - DAE

- `tereka`
  - CNN / RNN

- `takuoko`
  - 任務是製造模型多樣性
  - 一個 LGBM 單模約到 private `0.79966` / public `0.80328`

- `Angus / Shuo-Jen Chang`
  - 用自己的特徵與模型技巧生成多樣性

- `Giba`
  - user-id based post-processing

- `Maxwell`
  - `train_app` 與 `bureau` 的 meta features
  - `LGBM`, `ExtraTree` 做模型多樣性
  - adversarial validation blending

### 2nd Place 的主要技術點

- 非常重視 train/test 不同分布
- 公開投影片提到 blending 會刻意管理不同性質的輸出：
  - `CV ~ Public` 的 balanced 輸出
  - `CV > Public` 的 slightly overfit 輸出
  - `CV < Public` 的 slightly underfit 輸出
- `previous_application` 的 interest rate feature 很重要
- 他們的投影片指出：
  - current 沒有 repayment period
  - 但可用 previous 的資料先訓練模型去預測 current 的 repayment period
  - 再從預測出的 repayment period 反推出利率
- `POS_CASH` 很髒，清理後對分數有幫助

### Angus / Shuo-Jen Chang 附件重點

- 從 `kxx` kernel 出發，主要做 model tricks

- categorical 嘗試過：
  - n-way interactions
  - `Weight of Evidence`
  - xgb-embedding 再餵給 lgbm
  - count encoding
  - label encoding
  - one-hot encoding
  - OOF likelihood encoding

- numeric 嘗試過：
  - 對重要特徵做 grouped mean，例如：
    - `EXT_1/2/3`
    - `AMT_ANNUITY`
    - `AMT_CREDIT`
    - `AMT_CREDIT / AMT_ANNUITY`
  - 用全表訓練 LGB 去預測 `EXT_1/2/3`
  - 用 `pred_ext` 做衍生特徵
  - 用 `(actual - pred)` 或 `(actual - grouped)` 做 diff features
  - 對 `EXT`、`pred_EXT` 做 k-means
  - ridge OOF trick
  - quantile / histogram binning
  - branden 的 8 個利率相關特徵

- 他自己總結「有效」的：
  - `label encoding`
  - grouped mean
  - LGB prediction of important features
  - diff features

- 他自己總結「無效」的：
  - n-way interactions
  - WoE
  - xgb-embedding
  - count encoding
  - one-hot encoding
  - likelihood encoding
  - ridge trick
  - binning

- table-level 聚合策略：
  - numeric 主要做 `max / min / mean / sum`
  - categorical 主要做 count / ratio
  - time windows 包含過去 `6m / 1yr / 2yr / 3yr / 5yr`

- 他提到 `super heavy regularization` 的 LGBM 在這題有效

### 對目前程式的對應

- [test2.py](C:/Users/TYM/Downloads/case%20study/test2.py:89) 加入 application 端特徵
- [test2.py](C:/Users/TYM/Downloads/case%20study/test2.py:151) 加入 `PREV_INTEREST_EST`
- [test2.py](C:/Users/TYM/Downloads/case%20study/test2.py:207) 使用 `LightGBM + XGBoost + CatBoost`
- [blend.py](C:/Users/TYM/Downloads/case%20study/blend.py:14) 雖然不是原始 2nd place blending，但已經在做 leaderboard-aware blend

### 對後續的啟示

- `previous_application` 不應只做簡單聚合，利率 / 期數推導還能繼續深化
- `EXT_SOURCE` prediction / diff features 是很值得吸收的一條支線
- 要把模型多樣性做在不同特徵視角上，而不是只在同一張 full table 上換模型

## 3rd Place: alijs & Evgeny

### 核心觀點

- 不用 leak、duplicate rows 或 post-processing trick
- `Evgeny` 走信用評分業務邏輯導向
- `alijs` 走統計聚合與模型多樣性導向

### Evgeny 的做法

- 對每個 block 分別建模：
  - base application
  - last application
  - bureau
  - credit cards
  - installment
- 主模型使用子模型預測值與部分子模型特徵
- 主模型最後只有 `124` 個特徵
- 全部建立過的特徵超過 `1000`，實際保留約 `250`
- 用 local CV 逐個增減特徵，做手動 feature selection
- `bureau` 與 `previous_application` 曾使用非聚合建模：
  - 不先按 `SK_ID_CURR` 聚合
  - 直接用 row-level 資料建模
  - 最後再對 `SK_ID_CURR` 平均預測
- 建立模型預測 `EXT_SOURCE`
  - 預測值本身有用
  - `pred - actual` 的差值也有用
- 也試過預測收入，但沒有成功

### alijs 的做法

- 盡量保留不同實驗，不急著丟掉
- 從約 `50` 次 model run 中挑出 `7` 個最不相關的模型做平均
- 最終做 second-level stacking
- 最佳提交是 4 個 second-level stacker 的簡單平均：
  - `LightGBM`
  - `Random Forest`
  - `Extra Trees`
  - `Linear Regression`
- 每個 stacker 吃約 `15` 個 first-level model predictions，加上一些 raw features
- 最終最高 private LB 的版本也是最高 CV 的版本

### 對目前程式的啟示

- `test3.py` 雖然有 ensemble，但還不是 block-wise modeling
- 目前仍缺：
  - 強力 feature selection
  - `EXT_SOURCE` prediction / residual features
  - 非聚合 row-level bureau / previous 子模型
  - 真正的 second-level stacking

## 4th Place: Quad Machine

### 核心觀點

- 關鍵技巧之一是 trend features
- 特別強調：
  - `installments`
  - `POS`
  - `bureau`
  - 最近 `1 / 3 / 5 / 10` 筆記錄特徵
- 在 feature subset 與 full feature set 上訓練大量 diverse models
- 模型數量超過 `200`

### 其他技術點

- Bayesian optimization 過程中，會把 prediction 存起來當 OOF 資產
- 在 stacking layer 前，先對 `100+ OOF` 做 feature selection
- 最後用 LightGBM 做 stacking，可提升約 `0.0005` CV
- 做了 revolving loan 的手動修正：
  - 若預測值大於 `0.4`，乘以 `0.8`

### 團隊運作方式

- 一人專注 stacking 與實驗追蹤
- 其他人專注產生 features 與 OOF
- 最後一週再回到大表上榨最後一點增益

### 對目前程式的啟示

- `test3.py` 已有 recent window，但目前主體是 `12M / 24M`
- 與 4th place 最大差距在：
  - 還沒有最近 `1/3/5/10` 筆記錄特徵
  - 還沒有真正的 OOF 資產管理
  - 還沒有 stacking 前的 OOF feature selection
  - 還沒有在不同 feature subset 上系統性做 diverse models

## 對目前三支程式的映射

### `test.py`

- 主要偏向 1st place 風格
- 關鍵點：
  - `EXT_SOURCE` 衍生特徵
  - `Neighbors Target Mean`
  - 多表聚合
  - `LightGBM + XGBoost`

### `test2.py`

- 主要偏向 2nd place 風格
- 關鍵點：
  - `PREV_INTEREST_EST`
  - trend 類特徵
  - `CatBoost` 加入異質模型集成

### `test3.py`

- 目前是 1st + 2nd place 的融合版
- 相較前兩版更接近可持續優化的主幹，因為：
  - ratio 特徵使用 `safe_divide`
  - recent features 更完整
  - ensemble 權重改由 OOF 搜尋
- 但相對 3rd / 4th place，仍缺：
  - `last 1/3/5/10 records`
  - 強力 feature selection
  - block-wise 子模型
  - second-level stacking
  - `EXT_SOURCE` prediction / residual features

## 建議的下一步實作優先順序

1. 在 `bureau / POS / installments` 加入最近 `1/3/5/10` 筆聚合特徵
2. 在 `test3.py` 基礎上加入 feature selection
3. 建立 block-wise 子模型 OOF：
   - application-only
   - bureau-only
   - previous-only
   - installments-only
   - cards-only
4. 把 block-wise OOF 餵給 second-level stacker
5. 研究 `EXT_SOURCE_1/2/3` 的 prediction / residual features
6. 最後才考慮 leaderboard trick 類修正，例如 revolving loan manual correction

## 來源補充

### 直接來源

- 1st、2nd、3rd、4th place 原文由你提供

### 可驗證的外部來源

- 2nd place 公開投影片：
  - https://speakerdeck.com/hoxomaxwell/home-credit-default-risk-2nd-place-solutions
- Shuo-Jen Chang 附件：
  - https://storage.googleapis.com/kaggle-forum-message-attachments/379776/10237/HC%20-%20Brief%20solution%20from%20Shuo-Jen%20Chang.html
- 外部摘要整理：
  - https://lonepatient.top/2018/09/09/kaggle-home-credit-default-risk.html
- 關於 adversarial blending 的整理：
  - https://upura.hatenablog.com/entry/2018/09/02/210639

## 使用方式

- 要回頭說明 `test.py / test2.py / test3.py` 的設計來源時，可以先看這份文件
- 要開始規劃 `test4.py` 時，優先參考這份文件最後一節的改版順序

## v6 Notebook (Private 0.80152) 實作細節

這份筆記本展示了如何透過深度的特徵工程與嚴謹的特徵篩選，達到 Private 0.80152 的高分。與目前主力版本 `test4.py` 的主要差異及啟示如下：

### 1. 進階特徵工程 (Feature Engineering)
- **Time-weighted (時間權重衰減) 特徵**：使用指數衰減函數，給予近期記錄較高的權重，並計算加權平均（`time_weighted_agg`）。
- **Trend (趨勢斜率) 特徵**：計算特徵隨時間變化的線性迴歸斜率（`compute_trend`），捕捉使用者行為的變化趨勢。
- **Sub-Model OOF 特徵 (Block-wise Modeling)**：針對 `previous_application`、`bureau`、`installments` 建立 row-level 的子模型（例如用 LGBM 預測單筆記錄的違約機率），再將預測結果聚合（mean, max, std）後作為主模型的特徵。這是極大提升預測能力的關鍵。
- **Target Encoding 與組合特徵**：建立多個類別組合特徵（如 `AGE_RANGE__NAME_EDUCATION_TYPE`），並使用帶有平滑化 (smoothing) 的 Out-of-Fold Target Encoding，取代單純的 One-Hot Encoding。
- **KNN Target Features**：使用 KNN 在特定特徵（如 `EXT_SOURCE` 等）上找鄰居，並將鄰居的 Target 平均作為新特徵。

### 2. 特徵篩選的嚴謹度 (Feature Selection)
- **Null-Importance Selection**：將 Target 打亂 (shuffle) 並訓練模型多次，取得「純雜訊」下的特徵重要性，然後對比「實際重要性」與「雜訊重要性」，淘汰訊號微弱的特徵。相較於單純依靠 Importance > 0，此方法更能抵抗 overfitting。
- **高相關性過濾**：移除特徵間相關係數大於 0.985 的冗餘特徵。

### 3. 模型集成與多樣性 (Model Diversity)
- **特徵子集 (Feature Subsets)**：並非只用單一特徵集訓練不同的算法，而是透過切割特徵子集（例如：全部特徵、Top 400 特徵、移除 GP 特徵）結合不同的 Random Seed 來訓練多個 LightGBM 模型，以提升模型間的差異性與集成效果。

### 對後續改版的啟示
- 後續版本應優先考慮實作 **Sub-Model OOF 特徵**。
- 將時間特徵從單純的「最近幾筆」升級為 **Time-weighted** 與 **Trend** 特徵。
- 淘汰現有的簡單 Feature Selection，改用 **Null-Importance** 方法。