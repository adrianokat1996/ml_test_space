#!/usr/bin/env python3
"""
Event Impact Experiment (Config-driven)

This script implements the workflow from the notebook:
- Train a baseline model WITHOUT event features to capture normal structure
- Compute out-of-sample residuals and mark large-residual days using MAD threshold
- Align large residuals with event days and run Fisher's exact test (if SciPy available)
- Compute ROC-AUC using |residual| to discriminate event vs non-event days

Configuration:
- The script loads configuration from a JSON file.
- Config path resolution order:
  1) ENV var `EVENT_IMPACT_CONFIG`
  2) `event_impact_config.json` next to this script

Run:
  python /Users/katadriano/Documents/ml_test_space/event_impact_experiment.py

Optional:
  export EVENT_IMPACT_CONFIG=/abs/path/to/event_impact_config.json
"""

import os
import sys
import math
import json
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score

try:
    from scipy.stats import fisher_exact
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return math.sqrt(mean_squared_error(y_true, y_pred))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom[denom == 0] = 1.0
    return 100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / denom)


def mad(x: np.ndarray) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def load_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if 'FT_CalendarDate' not in df.columns:
        raise ValueError("Expected column 'FT_CalendarDate'")
    df['FT_CalendarDate'] = pd.to_datetime(df['FT_CalendarDate'])
    df = df.sort_values(['Location', 'FT_CalendarDate']).reset_index(drop=True)

    numeric_cols_guess = [
        'DayOfWeek','Month','isWeekend','isMacauHoliday','isChinaHoliday',
        'consecutiveChina','consecutiveMacau','DayOfMonth','Year','is_weekend',
        'is_month_start','EventSize_MinMax','days_to_holiday','holiday_weight',
        'holiday_decay','weekendHoliday','Actual'
    ]
    for c in numeric_cols_guess:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    if 'EventSize_Category' in df.columns:
        df['EventSize_Category'] = df['EventSize_Category'].astype(str)

    return df


def get_baseline_features(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    all_features = [
        'DayOfWeek','Month','isWeekend','isMacauHoliday','isChinaHoliday',
        'consecutiveChina','consecutiveMacau','DayOfMonth','Year','is_weekend',
        'is_month_start','days_to_holiday','holiday_weight','holiday_decay','weekendHoliday'
    ]
    base = [c for c in all_features if c in df.columns]
    cat_candidates = [
        'DayOfWeek','Month','DayOfMonth','Year','is_weekend','is_month_start',
        'isWeekend','isMacauHoliday','isChinaHoliday','weekendHoliday'
    ]
    cat = [c for c in cat_candidates if c in base]
    num = [c for c in base if c not in cat]
    return base, cat, num


def time_split(df_year: pd.DataFrame, val_days: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df_year.empty:
        raise ValueError("Empty year slice for splitting")
    last_dt = df_year['FT_CalendarDate'].max()
    val_start = last_dt - pd.Timedelta(days=val_days-1)
    train_df = df_year.loc[df_year['FT_CalendarDate'] < val_start].copy()
    val_df = df_year.loc[(df_year['FT_CalendarDate'] >= val_start) & (df_year['FT_CalendarDate'] <= last_dt)].copy()
    if train_df.empty or val_df.empty:
        raise ValueError("Train/Val split resulted in empty sets. Consider reducing val_days.")
    return train_df, val_df


def build_pipeline(cat: List[str], num: List[str], n_estimators: int, random_state: int) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat),
            ('num', StandardScaler(), num)
        ],
        remainder='drop'
    )
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1
    )
    pipe = Pipeline(steps=[('prep', preprocess), ('model', model)])
    return pipe


def compute_event_flag(df_like: pd.DataFrame) -> np.ndarray:
    if 'EventSize_Category' in df_like.columns:
        cat_flag = df_like['EventSize_Category'].astype(str).str.lower() != 'none'
    else:
        cat_flag = pd.Series(False, index=df_like.index)
    if 'EventSize_MinMax' in df_like.columns:
        num_flag = (df_like['EventSize_MinMax'].fillna(0) > 0)
    else:
        num_flag = pd.Series(False, index=df_like.index)
    return (cat_flag | num_flag).astype(int).values


def fisher_and_auc(large: np.ndarray, evt: np.ndarray, abs_res: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    A = int((large & (evt==1)).sum())
    B = int((large & (evt==0)).sum())
    C = int(((~large) & (evt==1)).sum())
    D = int(((~large) & (evt==0)).sum())
    fisher_or, fisher_p = None, None
    if SCIPY_AVAILABLE:
        try:
            table = np.array([[A, B],[C, D]])
            fisher_or, fisher_p = fisher_exact(table, alternative='greater')
        except Exception:
            fisher_or, fisher_p = None, None
    auc = None
    try:
        auc = roc_auc_score(evt, abs_res)
    except Exception:
        auc = None
    return (
        None if fisher_or is None else float(fisher_or),
        None if fisher_p is None else float(fisher_p),
        None if auc is None else float(auc)
    )


def export_combined_table(train_df: pd.DataFrame,
                          val_df_with_preds: pd.DataFrame,
                          baseline_features: List[str],
                          out_path: str) -> None:
    train_df = train_df.copy()
    val_df = val_df_with_preds.copy()
    train_df['Split'] = 'train'
    val_df['Split'] = 'valid'
    combined = pd.concat([train_df, val_df], axis=0, ignore_index=True, sort=False)
    key_cols = ['FT_CalendarDate','Location']
    event_features = ['EventSize_MinMax','EventSize_Category']
    extra_cols = ['Split','y_pred','residual','large_residual','event_day']
    ordered = [c for c in key_cols + ['Actual'] + extra_cols + baseline_features + event_features if c in combined.columns]
    combined = combined[ordered].sort_values(['Location','FT_CalendarDate']).reset_index(drop=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    combined.to_csv(out_path, index=False)


def load_config(config_path: Optional[str] = None) -> dict:
    """Load JSON config from env or default file and apply defaults."""
    # Resolve path
    resolved = config_path or os.environ.get('EVENT_IMPACT_CONFIG')
    if not resolved:
        resolved = os.path.join(os.path.dirname(__file__), 'event_impact_config.json')
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Config not found: {resolved}")
    with open(resolved, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    # Defaults
    cfg.setdefault('csv', '/Users/katadriano/Documents/ml_test_space/filtered_df.csv')
    cfg.setdefault('year', 2024)
    cfg.setdefault('val_days', 28)
    cfg.setdefault('k_mad', 3.0)
    cfg.setdefault('active_location', None)
    cfg.setdefault('n_estimators', 300)
    cfg.setdefault('random_state', 42)
    cfg.setdefault('export_combined', False)
    cfg.setdefault('export_multiloc_summary', False)
    cfg.setdefault('out_dir', '/Users/katadriano/Documents/ml_test_space')

    return cfg


def analyze_single_location(df_all: pd.DataFrame,
                            location: str,
                            year: int,
                            val_days: int,
                            k_mad: float,
                            n_estimators: int,
                            random_state: int,
                            export_combined: bool,
                            out_dir: str) -> pd.DataFrame:
    d = df_all.loc[df_all['Location'] == location].copy()
    d_yr = d.loc[d['Year'] == year].copy()
    if d_yr.empty:
        raise ValueError(f"No data for location {location} in year {year}")
    baseline_features, cat, num = get_baseline_features(d_yr)
    train_df, val_df = time_split(d_yr, val_days)
    X_tr, y_tr = train_df[baseline_features], train_df['Actual']
    X_va, y_va = val_df[baseline_features], val_df['Actual']
    pipe = build_pipeline(cat, num, n_estimators, random_state)
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_va)
    res = y_va.values - pred
    abs_res = np.abs(res)
    m = mad(abs_res)
    med = float(np.median(abs_res))
    thr = med + k_mad * m
    large = abs_res >= thr
    evt = compute_event_flag(val_df)
    fisher_or, fisher_p, auc = fisher_and_auc(large, evt, abs_res)

    val_df = val_df.copy()
    val_df['y_pred'] = pred
    val_df['residual'] = res
    val_df['abs_residual'] = abs_res
    val_df['large_residual'] = large
    val_df['event_day'] = evt

    if export_combined:
        out_path = os.path.join(out_dir, f"predictions_{year}_{location.replace(' ', '_').replace('/', '_')}.csv")
        export_combined_table(train_df, val_df, baseline_features, out_path)

    summary = pd.DataFrame([{
        'Location': location,
        'TrainDays': int(len(train_df)),
        'ValDays': int(len(val_df)),
        'MAE': float(mean_absolute_error(y_va, pred)),
        'RMSE': float(rmse(y_va, pred)),
        'sMAPE': float(smape(y_va, pred)),
        'MedianAbsRes': float(med),
        'MAD_AbsRes': float(m),
        'Threshold': float(thr),
        'NumLargeRes': int(large.sum()),
        'NumEventsInVal': int((evt==1).sum()),
        'Fisher_OR': fisher_or,
        'Fisher_p': fisher_p,
        'AUC_absRes_vs_Event': auc
    }])
    return summary


def analyze_multi_location(df_all: pd.DataFrame,
                           year: int,
                           val_days: int,
                           k_mad: float,
                           n_estimators: int,
                           random_state: int) -> pd.DataFrame:
    results = []
    locs = sorted(df_all['Location'].dropna().unique())
    for loc in locs:
        try:
            d = df_all.loc[df_all['Location'] == loc].copy()
            d_yr = d.loc[d['Year'] == year].copy()
            if d_yr.empty or d_yr['FT_CalendarDate'].nunique() < 40:
                continue
            base, cat, num = get_baseline_features(d_yr)
            train_df, val_df = time_split(d_yr, val_days)
            X_tr, y_tr = train_df[base], train_df['Actual']
            X_va, y_va = val_df[base], val_df['Actual']
            pipe = build_pipeline(cat, num, n_estimators, random_state)
            pipe.fit(X_tr, y_tr)
            pred = pipe.predict(X_va)
            res = y_va.values - pred
            abs_res = np.abs(res)
            m = mad(abs_res)
            med = float(np.median(abs_res))
            thr = med + k_mad * m
            large = abs_res >= thr
            evt = compute_event_flag(val_df)
            fisher_or, fisher_p, auc = fisher_and_auc(large, evt, abs_res)
            results.append({
                'Location': loc,
                'TrainDays': int(len(train_df)),
                'ValDays': int(len(val_df)),
                'MAE': float(mean_absolute_error(y_va, pred)),
                'RMSE': float(rmse(y_va, pred)),
                'sMAPE': float(smape(y_va, pred)),
                'MedianAbsRes': float(med),
                'MAD_AbsRes': float(m),
                'Threshold': float(thr),
                'NumLargeRes': int(large.sum()),
                'NumEventsInVal': int((evt==1).sum()),
                'Fisher_OR': fisher_or,
                'Fisher_p': fisher_p,
                'AUC_absRes_vs_Event': auc
            })
        except Exception as e:
            warnings.warn(f"Skipping {loc} due to error: {e}")
            continue
    return pd.DataFrame(results)


def main(config_path: Optional[str] = None) -> int:
    cfg = load_config(config_path)

    df = load_data(cfg['csv'])
    print(f"Loaded data: {len(df)} rows, {df['Location'].nunique()} locations, date range: {df['FT_CalendarDate'].min()} -> {df['FT_CalendarDate'].max()}")

    # Single-location path
    if cfg.get('active_location'):
        summary = analyze_single_location(
            df_all=df,
            location=cfg['active_location'],
            year=int(cfg['year']),
            val_days=int(cfg['val_days']),
            k_mad=float(cfg['k_mad']),
            n_estimators=int(cfg['n_estimators']),
            random_state=int(cfg['random_state']),
            export_combined=bool(cfg['export_combined']),
            out_dir=str(cfg['out_dir'])
        )
        print("Single-location summary:")
        print(summary.to_string(index=False))
        if cfg.get('export_combined'):
            print("Combined predictions CSV exported.")

    # Multi-location summary path
    if cfg.get('export_multiloc_summary') or not cfg.get('active_location'):
        multi = analyze_multi_location(
            df_all=df,
            year=int(cfg['year']),
            val_days=int(cfg['val_days']),
            k_mad=float(cfg['k_mad']),
            n_estimators=int(cfg['n_estimators']),
            random_state=int(cfg['random_state'])
        )
        if not multi.empty:
            multi_sorted = multi.sort_values(['Fisher_p','AUC_absRes_vs_Event'], ascending=[True, False])
            print("Multi-location summary (top 20 by Fisher p ascending, AUC desc):")
            print(multi_sorted.head(20).to_string(index=False))
            if cfg.get('export_multiloc_summary'):
                out_path = os.path.join(str(cfg['out_dir']), f"multi_location_summary_{int(cfg['year'])}.csv")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                multi_sorted.to_csv(out_path, index=False)
                print(f"Saved multi-location summary: {out_path}")
        else:
            print("No multi-location summary produced (possibly insufficient data per location).")

    return 0


if __name__ == '__main__':
    sys.exit(main())


