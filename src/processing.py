import pandas as pd
import numpy as np

# --------------------
# 2차 정밀 전처리
# --------------------
def clean_outliers_rolling(df, column='Close', window=48, sigma=4):
    print("\n정밀 전처리 시작")

    df_clean = df.copy()
    rolling_mean = df_clean[column].rolling(window=window, center=True).mean()
    rolling_std = df_clean[column].rolling(window=window, center=True).std()

    upper = rolling_mean + (sigma * rolling_std)
    lower = rolling_mean - (sigma * rolling_std)

    outliers = (df_clean[column] > upper) | (df_clean[column] < lower)

    if outliers.sum() > 0:
        df_clean.loc[outliers, column] = np.nan
        df_clean[column] = df_clean[column].interpolate()

    print("정밀 전처리 완료\n")
    return df_clean

# --------------------
# 1차 전체 전처리
# --------------------
def preprocess_data(df_prices, df_notices):
    print("\n데이터 전처리 시작")

    # 1. 기본 인덱스 설정
    df = df_prices.copy()
    if 'logged_at' in df.columns:
        df['logged_at'] = pd.to_datetime(df['logged_at'])
        df.set_index('logged_at', inplace=True)
    
    # [추가] 만약 외부에서 'price'라는 이름으로 바꿔서 보냈다면 다시 'current_min_price'로 인식하게 함
    if 'price' in df.columns and 'current_min_price' not in df.columns:
        df = df.rename(columns={'price': 'current_min_price'})

    if 'logged_at' in df.columns:
        df['logged_at'] = pd.to_datetime(df['logged_at'])
        df.set_index('logged_at', inplace=True)

    
    # 2. 1차 이상치 전처리
    raw_window = 432  # 약 3일치 (10분 : 6 * 24 * 3)
    raw_sigma = 7
    
    r_mean = df['current_min_price'].rolling(window=raw_window, center=True).mean()
    r_std = df['current_min_price'].rolling(window=raw_window, center=True).std()
    
    upper = r_mean + (raw_sigma * r_std)
    lower = r_mean - (raw_sigma * r_std)
    
    mask = (df['current_min_price'] > upper) | (df['current_min_price'] < lower)
    if mask.sum() > 0:
        print(f"원본 데이터에서 {mask.sum()}개의 큰 이상치 제거")
        df.loc[mask, 'current_min_price'] = np.nan
        df['current_min_price'] = df['current_min_price'].interpolate()

    # 2. 30분 Resampling
    df_res = df['current_min_price'].resample('30min').agg(['first', 'max', 'min', 'last', 'mean'])
    df_res.columns = ['Open', 'High', 'Low', 'Close', 'Price_Mean']
    df_res = df_res.ffill().bfill()
    
    # 2. [핵심] 정밀 이상치 제거
    df_refined = clean_outliers_rolling(df_res, column='Close', window=48, sigma=4)
    
    # 3. GPT 점수 매핑
    df_refined['GPT_Score'] = 0.0
    if not df_notices.empty:
        for _, row in df_notices.iterrows():
            n_date = pd.to_datetime(row['notice_date'])
            s = n_date.replace(hour=10, minute=0, second=0)
            e = (n_date + pd.Timedelta(days=7)).replace(hour=6, minute=0, second=0)
            mask = (df_refined.index >= s) & (df_refined.index < e)
            df_refined.loc[mask, 'GPT_Score'] = row['gpt_score']

    # 4. 피처 생성 (MA, RSI, Lag)
    df_ml = df_refined.copy()
    df_ml['MA_5'] = df_ml['Close'].rolling(5).mean()
    df_ml['MA_48'] = df_ml['Close'].rolling(48).mean()
    df_ml['Std_20'] = df_ml['Close'].rolling(20).std()
    
    delta = df_ml['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df_ml['RSI'] = 100 - (100 / (1 + rs))
    
    df_ml['Close_Lag1'] = df_ml['Close'].shift(1)
    df_ml['Close_Lag2'] = df_ml['Close'].shift(2)
    df_ml['GPT_Lag1'] = df_ml['GPT_Score'].shift(1)
    df_ml['Hour'] = df_ml.index.hour
    df_ml['DayOfWeek'] = df_ml.index.dayofweek
    
    # Target (학습용)
    df_ml['Target'] = df_ml['Close'].shift(-1)
    
    print("데이터 정제 완료\n")
    return df_ml.dropna()
