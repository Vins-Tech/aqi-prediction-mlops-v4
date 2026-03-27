import pandas as pd
import numpy as np
import requests
import joblib
import holidays
from datetime import timedelta
from dotenv import load_dotenv
import os

load_dotenv()

sheet_id = os.getenv("SHEET_ID")
apikey = os.getenv("GOOGLE_API_KEY")

raw_weather_cols = joblib.load("artifacts/raw_weather_cols.joblib")
weather_cols = joblib.load("artifacts/weather_cols.joblib")
aqi_cols = joblib.load("artifacts/aqi_cols.joblib")
date_cols = joblib.load("artifacts/date_cols.joblib")
FEATURES = joblib.load("artifacts/selected_features.joblib")


def call_sheets():
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    df = pd.read_csv(url)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df


def call_weather(start, end):
    url = f'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/12.9135218%2C%2077.5950804/{start}/{end}?unitGroup=metric&elements=remove%3Aconditions%2Cremove%3AdatetimeEpoch%2Cremove%3Afeelslike%2Cremove%3Afeelslikemax%2Cremove%3Afeelslikemin%2Cremove%3Ahumidity%2Cremove%3Aname%2Cremove%3Aprecipprob%2Cremove%3Asevererisk%2Cremove%3Asnow%2Cremove%3Asnowdepth%2Cremove%3Astations%2Cremove%3Asunrise%2Cremove%3Asunset%2Cremove%3Atemp%2Cremove%3Atempmax%2Cremove%3Atempmin%2Cremove%3Auvindex&include=days&key={apikey}&contentType=json'
    resp = requests.get(url)
    df = pd.DataFrame(resp.json()['days'])
    return df


def build_weather_features(weather_df):
    df = weather_df.copy()
    df = df[raw_weather_cols]
    df['date'] = pd.to_datetime(df['datetime'])
    df = df.drop(columns=['datetime'])
    df = df.sort_values('date')

    df['preciptype'] = df['preciptype'].fillna("no")
    lst = df['preciptype'].tolist()
    result = []
    for i in range(len(lst)):
        result.append(lst[i][0])
    df['preciptype'] = result
    df["preciptype"] = df["preciptype"].map({"rain": 1, "n": 0}).astype(int)

    df["icon_clear-day"] = 0
    df["icon_partly-cloudy-day"] = 0
    df["icon_rain"] = 0
    icon_map = {
        "clear-day": "icon_clear-day",
        "partly-cloudy-day": "icon_partly-cloudy-day",
        "rain": "icon_rain"
    }
    for icon_value, col_name in icon_map.items():
        df.loc[df["icon"] == icon_value, col_name] = 1
    df = df.drop(columns=['icon'])

    return df


def build_lag_roll_features(df, feature_cols):
    for col in feature_cols:
        for lag in [1, 3, 7]:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    for col in feature_cols:
        df[f"{col}_roll_mean_3"] = df[col].shift(1).rolling(3).mean()
        df[f"{col}_roll_mean_7"] = df[col].shift(1).rolling(7).mean()
        df[f"{col}_roll_mean_14"] = df[col].shift(1).rolling(14).mean()

    df["rain_days_last_3"] = df["preciptype"].shift(1).rolling(3).sum()
    df["rain_days_last_7"] = df["preciptype"].shift(1).rolling(7).sum()

    df["wind_dispersion_index"] = df["windspeed_lag_1"] * df["visibility_lag_1"]
    df["stagnation_index"] = df["pressure_lag_1"] / (df["windspeed_lag_1"] + 1)

    return df


def build_aqi_features(df):
    for lag in [1, 2, 3, 5, 7, 14, 21, 30]:
        df[f"aqi_lag_{lag}"] = df["aqipm25"].shift(lag)

    for w in [3, 7, 14, 30]:
        df[f"aqi_roll_mean_{w}"] = df["aqipm25"].shift(1).rolling(w).mean()
        df[f"aqi_roll_std_{w}"] = df["aqipm25"].shift(1).rolling(w).std()

    df["aqi_roll_min_7"] = df["aqipm25"].shift(1).rolling(7).min()
    df["aqi_roll_max_7"] = df["aqipm25"].shift(1).rolling(7).max()

    return df


def build_date_features(df):
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek
    df["dayofyear"] = df["date"].dt.dayofyear
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)

    india_holidays = holidays.India()
    df["is_holiday"] = df["date"].apply(lambda x: 1 if x in india_holidays else 0)
    df["is_pre_holiday"] = (df["date"] + pd.Timedelta(days=1)).dt.date.isin(india_holidays).astype(int)
    df["is_post_holiday"] = (df["date"] - pd.Timedelta(days=1)).dt.date.isin(india_holidays).astype(int)

    return df


def hist_aqi_avg(df, day_offset):
    results = []
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['year'] = df['date'].dt.year

    for idx, row in df.iterrows():
        target_date = row["date"] + pd.Timedelta(days=day_offset)
        hist = df[
            (df["month"] == target_date.month) &
            (df["day"] == target_date.day) &
            (df["year"] < row["year"])
        ]["aqipm25"]
        results.append(hist.mean() if len(hist) > 0 else np.nan)

    return results


def build_features():
    print("Loading latest.csv...")
    latest_df = pd.read_csv("data/latest.csv")
    latest_df['date'] = pd.to_datetime(latest_df['date'])
    latest_max_date = latest_df['date'].max()
    print(f"Latest.csv has {len(latest_df)} rows, up to {latest_max_date.date()}")

    print("Loading Google Sheets...")
    sheet_df = call_sheets()
    print(f"Google Sheets has {len(sheet_df)} rows, up to {sheet_df['date'].max().date()}")

    # only dates strictly after latest.csv max date
    new_dates_df = sheet_df[sheet_df['date'] > latest_max_date].copy()
    print(f"New dates to engineer features for: {len(new_dates_df)}")

    if len(new_dates_df) == 0:
        print("No new dates found. latest.csv is already up to date.")
        return

    # fetch weather only for new dates + 14 day buffer for lag features
    start = new_dates_df['date'].min() - timedelta(days=14)
    end = new_dates_df['date'].max()
    print(f"Fetching weather from {start.date()} to {end.date()}...")
    weather_raw = call_weather(start.date(), end.date())
    weather_df = build_weather_features(weather_raw)

    # use full Google Sheets as AQI history
    combined_aqi = sheet_df[['date', 'aqipm25']].copy()
    combined_aqi = combined_aqi.sort_values('date').reset_index(drop=True)

    # build aqi lag/roll features on full history
    combined_aqi = build_aqi_features(combined_aqi)

    # build hist aqi features
    combined_aqi["aqi_hist_prev_day_avg"] = hist_aqi_avg(combined_aqi.copy(), -1)
    combined_aqi["aqi_hist_same_day_avg"] = hist_aqi_avg(combined_aqi.copy(), 0)
    combined_aqi["aqi_hist_next_day_avg"] = hist_aqi_avg(combined_aqi.copy(), 1)

    # build weather lag/roll features
    weather_feature_cols = [c for c in weather_df.columns if c != 'date']
    weather_df = build_lag_roll_features(weather_df, weather_feature_cols)

    # extract only new dates
    new_dates_list = list(new_dates_df['date'])
    new_aqi_features = combined_aqi[combined_aqi['date'].isin(new_dates_list)].copy()
    new_weather_features = weather_df[weather_df['date'].isin(new_dates_list)].copy()

    # build date features
    new_aqi_features = build_date_features(new_aqi_features)

    # merge aqi + weather
    merged = new_aqi_features.merge(
        new_weather_features[['date'] + weather_cols],
        on='date',
        how='inner'
    )

    # keep only columns matching latest.csv
    final_cols = list(latest_df.columns)
    merged = merged[[c for c in final_cols if c in merged.columns]]

    # drop rows with NaN in features
    merged = merged.dropna(subset=FEATURES)
    print(f"New rows engineered successfully: {len(merged)}")

    if len(merged) == 0:
        print("No valid rows after feature engineering. Check API data.")
        return

    # append to latest_df and save
    updated_df = pd.concat([latest_df, merged])
    updated_df = updated_df.drop_duplicates(subset=['date'])
    updated_df = updated_df.sort_values('date').reset_index(drop=True)
    updated_df.to_csv("data/latest.csv", index=False)

    print(f"latest.csv updated: {len(updated_df)} rows, up to {updated_df['date'].max().date()}")


if __name__ == "__main__":
    build_features()