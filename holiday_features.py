"""
Holiday features utilities

Build a holiday calendar with features similar to Holiday_calendar.csv:
- CalendarDate
- isWeekend
- isMacauHoliday
- isChinaHoliday
- consecutiveChina
- consecutiveMacau

Usage (example):
    from holiday_features import generate_holiday_features, save_holiday_calendar

    macau_holidays = ["2025-10-07", "2025-10-02", "2025-10-01", "2025-04-04"]
    china_holidays = ["2025-10-01", "2025-10-02", "2025-10-03", "2025-10-04",
                      "2025-10-05", "2025-10-06", "2025-10-07", "2025-10-08",
                      "2025-04-05", "2025-04-06", "2025-04-07"]

    df = generate_holiday_features(
        start_date="2025-03-29",
        end_date="2025-10-08",
        macau_holiday_dates=macau_holidays,
        china_holiday_dates=china_holidays
    )
    save_holiday_calendar(df, "/Users/katadriano/Documents/ml_test_space/Holiday_calendar.generated.csv")
"""

from __future__ import annotations

import os
from typing import Iterable, List, Optional, Set

import pandas as pd
import numpy as np
try:
    import holidays  # pip install holidays
    HOLIDAYS_AVAILABLE = True
except Exception:
    HOLIDAYS_AVAILABLE = False


def _to_datetime_series(dates: Iterable) -> pd.DatetimeIndex:
    """Convert a collection of date-like objects/strings to a DatetimeIndex (UTC-naive)."""
    if dates is None:
        return pd.DatetimeIndex([])
    return pd.to_datetime(list(dates))


def _consecutive_counts(binary_flags: np.ndarray) -> np.ndarray:
    """Return per-position consecutive counts for runs of ones; zeros yield 0.

    Example: [1,1,0,1] -> [1,2,0,1]
    """
    counts = np.zeros(len(binary_flags), dtype=int)
    run = 0
    for i, flag in enumerate(binary_flags):
        if flag:
            run += 1
            counts[i] = run
        else:
            run = 0
            counts[i] = 0
    return counts


def generate_holiday_features(
    start_date: str,
    end_date: str,
    macau_holiday_dates: Iterable,
    china_holiday_dates: Iterable,
    weekend_days: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Generate holiday-related features for a date range.

    Parameters
    ----------
    start_date : str
        Inclusive start date (e.g., '2025-01-01').
    end_date : str
        Inclusive end date (e.g., '2025-12-31').
    macau_holiday_dates : Iterable
        Collection of date strings or datetime-like objects representing Macau holidays.
    china_holiday_dates : Iterable
        Collection of date strings or datetime-like objects representing China holidays.
    weekend_days : list[int], optional
        Weekday indices treated as weekend. Default [5, 6] (Saturday=5, Sunday=6).

    Returns
    -------
    pd.DataFrame
        Columns: ['CalendarDate','isWeekend','isMacauHoliday','isChinaHoliday',
                  'consecutiveChina','consecutiveMacau']
        CalendarDate is a datetime64[ns] column.
    """

    if weekend_days is None:
        weekend_days = [5, 6]

    date_index = pd.date_range(start=pd.to_datetime(start_date), end=pd.to_datetime(end_date), freq='D')
    macau_idx = _to_datetime_series(macau_holiday_dates)
    china_idx = _to_datetime_series(china_holiday_dates)

    # Base frame
    df = pd.DataFrame({'CalendarDate': date_index})
    df['weekday'] = df['CalendarDate'].dt.weekday
    df['isWeekend'] = df['weekday'].isin(weekend_days).astype(int)

    # Holiday flags
    macau_set = set(pd.to_datetime(macau_idx).date)
    china_set = set(pd.to_datetime(china_idx).date)
    dates_date = df['CalendarDate'].dt.date
    df['isMacauHoliday'] = dates_date.map(lambda d: 1 if d in macau_set else 0).astype(int)
    df['isChinaHoliday'] = dates_date.map(lambda d: 1 if d in china_set else 0).astype(int)

    # Extend holiday flags to weekends adjacent to holidays (連假邏輯)
    china_flag = df['isChinaHoliday'].astype(bool)
    macau_flag = df['isMacauHoliday'].astype(bool)
    weekend_flag = df['isWeekend'].astype(bool)
    china_weekend_adj = weekend_flag & (
        china_flag.shift(1, fill_value=False) | china_flag.shift(-1, fill_value=False)
    )
    macau_weekend_adj = weekend_flag & (
        macau_flag.shift(1, fill_value=False) | macau_flag.shift(-1, fill_value=False)
    )
    df['isChinaHoliday'] = (china_flag | china_weekend_adj).astype(int)
    df['isMacauHoliday'] = (macau_flag | macau_weekend_adj).astype(int)

    # Consecutive counts (per holiday system)
    df = df.sort_values('CalendarDate').reset_index(drop=True)
    df['consecutiveChina'] = _consecutive_counts(df['isChinaHoliday'].values.astype(bool))
    df['consecutiveMacau'] = _consecutive_counts(df['isMacauHoliday'].values.astype(bool))

    # Order and dtypes
    out_cols = [
        'CalendarDate',
        'isWeekend',
        'isMacauHoliday',
        'isChinaHoliday',
        'consecutiveChina',
        'consecutiveMacau',
    ]
    return df[out_cols]


def _build_holiday_date_set(country_code: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> Set[pd.Timestamp]:
    """Build a set of holiday dates using the `holidays` package for a country/region code.

    Only dates within [start_date, end_date] are returned.
    """
    if not HOLIDAYS_AVAILABLE:
        raise ImportError("The 'holidays' package is required. Install with: pip install holidays")
    years = list(range(start_date.year, end_date.year + 1))
    prov = holidays.country_holidays(country=country_code, years=years)
    dates = pd.to_datetime(list(prov.keys()))
    dates = dates[(dates >= start_date) & (dates <= end_date)]
    return set(dates.normalize().date)


def generate_holiday_features_by_package(
    start_date: str,
    end_date: str,
    china_code: str = "CN",
    macau_code: str = "MO",
    weekend_days: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Generate holiday features using the `holidays` package for China and Macau.

    Parameters
    ----------
    start_date : str
        Inclusive start date (e.g., '2025-01-01').
    end_date : str
        Inclusive end date (e.g., '2025-12-31').
    china_code : str
        Country code for China in the holidays package (default 'CN').
    macau_code : str
        Country/region code for Macau in the holidays package (default 'MO').
    weekend_days : list[int], optional
        Weekday indices treated as weekend. Default [5, 6] (Saturday=5, Sunday=6).

    Returns
    -------
    pd.DataFrame
        Columns: ['CalendarDate','isWeekend','isMacauHoliday','isChinaHoliday',
                  'consecutiveChina','consecutiveMacau']
    """
    if weekend_days is None:
        weekend_days = [5, 6]

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    date_index = pd.date_range(start=start_ts, end=end_ts, freq='D')

    china_set = _build_holiday_date_set(china_code, start_ts, end_ts)
    macau_set = _build_holiday_date_set(macau_code, start_ts, end_ts)

    df = pd.DataFrame({'CalendarDate': date_index})
    df['weekday'] = df['CalendarDate'].dt.weekday
    df['isWeekend'] = df['weekday'].isin(weekend_days).astype(int)

    dates_date = df['CalendarDate'].dt.date
    df['isChinaHoliday'] = dates_date.map(lambda d: 1 if d in china_set else 0).astype(int)
    df['isMacauHoliday'] = dates_date.map(lambda d: 1 if d in macau_set else 0).astype(int)

    # Extend holiday flags to weekends adjacent to holidays (連假邏輯)
    china_flag = df['isChinaHoliday'].astype(bool)
    macau_flag = df['isMacauHoliday'].astype(bool)
    weekend_flag = df['isWeekend'].astype(bool)
    china_weekend_adj = weekend_flag & (
        china_flag.shift(1, fill_value=False) | china_flag.shift(-1, fill_value=False)
    )
    macau_weekend_adj = weekend_flag & (
        macau_flag.shift(1, fill_value=False) | macau_flag.shift(-1, fill_value=False)
    )
    df['isChinaHoliday'] = (china_flag | china_weekend_adj).astype(int)
    df['isMacauHoliday'] = (macau_flag | macau_weekend_adj).astype(int)

    df = df.sort_values('CalendarDate').reset_index(drop=True)
    df['consecutiveChina'] = _consecutive_counts(df['isChinaHoliday'].values.astype(bool))
    df['consecutiveMacau'] = _consecutive_counts(df['isMacauHoliday'].values.astype(bool))

    return df[['CalendarDate','isWeekend','isMacauHoliday','isChinaHoliday','consecutiveChina','consecutiveMacau']]


def get_holiday_df(start_date: str, end_date: str) -> pd.DataFrame:
    """Convenience wrapper that returns holiday features for CN and MO using holidays package."""
    return generate_holiday_features_by_package(start_date=start_date, end_date=end_date)


def save_holiday_calendar(df: pd.DataFrame, path: str, date_format: Optional[str] = None) -> None:
    """Save the holiday features DataFrame to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame returned by generate_holiday_features.
    path : str
        Output CSV path.
    date_format : Optional[str]
        If provided, formats CalendarDate as string using given strftime pattern
        before saving (e.g., '%Y-%m-%d' or '%-m/%-d/%Y'). If None, writes datetime
        values directly letting pandas handle formatting.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    to_save = df.copy()
    if date_format:
        try:
            to_save['CalendarDate'] = to_save['CalendarDate'].dt.strftime(date_format)
        except Exception:
            # Fallback for platforms where %-m/%-d is unsupported
            safe_format = date_format.replace('%-m', '%m').replace('%-d', '%d')
            to_save['CalendarDate'] = to_save['CalendarDate'].dt.strftime(safe_format)
    to_save.to_csv(path, index=False)


__all__ = [
    'generate_holiday_features',
    'generate_holiday_features_by_package',
    'get_holiday_df',
    'save_holiday_calendar',
]


