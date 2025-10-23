# ml_test_space

## 目的
- 驗證「外部事件」是否對每日實際值（`Actual`）具有關鍵影響。
- 先以「不含事件特徵」的基線模型學習一般結構（趨勢、季節、週期、假日），再以殘差與事件對齊、統計檢定與可視化分析進行驗證。

## 資料
- `filtered_df.csv`：日頻資料，主要欄位包含：
  - 日期與地點：`FT_CalendarDate`, `Location`
  - 週期特徵：`DayOfWeek`, `Month`, `DayOfMonth`, `Year`, `is_weekend`, `is_month_start`
  - 假日相關：`isMacauHoliday`, `isChinaHoliday`, `consecutiveChina`, `consecutiveMacau`, `days_to_holiday`, `holiday_weight`, `holiday_decay`, `weekendHoliday`
  - 事件特徵（可選）：`EventSize_MinMax`, `EventSize_Category`
  - 目標值：`Actual`
- Notebook：`Event Features Exp.ipynb`
- 初版構想：`note.txt`

## 實驗設計（高層流程）
1. 建立不含事件特徵的基線模型
   - 僅使用一般性與週期/假日特徵，明確排除事件特徵。
   - 採用時間切分（例：最後 4 週作為驗證集）以獲得 out-of-sample 預測。
2. 產生預測與殘差
   - 計算殘差 \( r_t = y_t - \hat{y}_t \)。
   - 以穩健準則（如 MAD×k 或 Top 10%）定義「大殘差日」。
3. 事件對齊與統計檢定
   - 檢查「大殘差日」與事件日（或事件窗口）是否顯著重疊。
   - 使用 Fisher’s exact test / 超幾何檢定或 permutation test 進行檢定。
   - 以 |殘差| 作為分數，計算 ROC-AUC 以評估對事件日的辨識能力。
4. 結論與可視化
   - 報告事件相關性的證據（顯著性、AUC、效果方向與大小）。
   - 繪製殘差時間序列與事件疊加圖，輔以事件窗口（±K 天）分析。

## 強化分析（可選）
- 增益測試：加入事件特徵後是否顯著降低 MAE/RMSE/sMAPE（可用 Diebold–Mariano test 檢定）。
- 反事實估計：以結構性時間序列（BSTS/CausalImpact）、SARIMAX/Prophet（含外生變數）在「無事件」假設下估計事件期間的累積影響與信賴區間。
- Placebo 與穩健性：隨機打散事件日期重複模擬；多事件類別/窗口時進行 FDR（Benjamini–Hochberg）校正。

## 評估指標
- 主要：MAE、RMSE、sMAPE（視零值比例選用）。
- 輔助：ROC-AUC（以 |殘差| 區分事件/非事件日）。

## 可重現步驟（Notebook）
1) 讀取與清理資料（確保日頻補齊、型別正確；訓練時排除事件特徵）
2) 建立時間切分；訓練基線模型；產生 out-of-sample 預測與殘差
3) 定義大殘差日並與事件對齊；執行統計檢定與 ROC-AUC
4) 視覺化殘差與事件窗口；輸出報告與圖表
5)（可選）增益測試與反事實估計

## 假設與限制
- 本流程側重關聯性而非嚴格因果；若需因果結論，建議加入對照設計（差異中的差異、合成控制等）。
- 事件表需與 `date + location` 精準對齊；多事件同日需定義聚合策略（sum/mean/max/count）。
- 若多地點異質性高，建議分組建模或於共模中加入地點特徵並採分層時間切分。

## 專案檔案
- `Event Features Exp.ipynb`：實驗流程與分析
- `filtered_df.csv`：資料輸入
- `note.txt`：初版計畫
- `README.md`：實驗目的、方法與重現說明（本文）