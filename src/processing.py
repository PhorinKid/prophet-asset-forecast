import pandas as pd
import numpy as np

def clean_outliers_rolling(df, column='Close', window=48, sigma=4):

    df_clean = df.copy()
    rolling_mean = df_clean[column].rolling(window=window, center=True).mean()
    rolling_std = df_clean[column].rolling(window=window, center=True).std()

    upper = rolling_mean + (sigma * rolling_std)
    lower = rolling_mean - (sigma * rolling_std)

    outliers = (df_clean[column] > upper) | (df_clean[column] < lower)

    if outliers.sum() > 0:
        df_clean.loc[outliers, column] = np.nan
        df_clean[column] = df_clean[column].interpolate()

    return df_clean

def preprocess_data(df_prices, df_notices):
    """
    전체 데이터 전처리 파이프라인
    1. 인덱싱 및 컬럼 정리
    2. 1차 이상치 제거 (Raw Data)
    3. 리샘플링 (30분)
    4. 2차 정밀 이상치 제거
    5. 피처 엔지니어링 (MA, RSI, Lag, Time)
    """
    # 1. 인덱스 및 컬럼 설정
    df = df_prices.copy()
    if 'logged_at' in df.columns:
        df['logged_at'] = pd.to_datetime(df['logged_at'])
        df.set_index('logged_at', inplace=True)
    
    if 'price' in df.columns and 'current_min_price' not in df.columns:
        df = df.rename(columns={'price': 'current_min_price'})

    # 2. 1차 이상치 제거 (Raw Data, 약 3일 윈도우)
    raw_window = 432
    raw_sigma = 7

    r_mean = df['current_min_price'].rolling(window=raw_window, center=True).mean()
    r_std = df['current_min_price'].rolling(window=raw_window, center=True).std()

    upper = r_mean + (raw_sigma * r_std)
    lower = r_mean - (raw_sigma * r_std)

    mask = (df['current_min_price'] > upper) | (df['current_min_price'] < lower)
    if mask.sum() > 0:
        df.loc[mask, 'current_min_price'] = np.nan
        df['current_min_price'] = df['current_min_price'].interpolate()

    # 3. 30분 단위 Resampling
    df_res = df['current_min_price'].resample('30min').agg(['first', 'max', 'min', 'last', 'mean'])
    df_res.columns = ['Open', 'High', 'Low', 'Close', 'Price_Mean']
    df_res = df_res.ffill().bfill()
    
    # 4. 2차 정밀 이상치 제거
    df_refined = clean_outliers_rolling(df_res, column='Close', window=48, sigma=4)
    
    # 5. GPT 점수 매핑
    df_refined['GPT_Score'] = 0.0
    if df_notices is not None and not df_notices.empty:
        for _, row in df_notices.iterrows():
            n_date = pd.to_datetime(row['notice_date'])
            # 업데이트 점검: 수요일 10:00 ~ 다음주 수요일 06:00
            s = n_date.replace(hour=10, minute=0, second=0)
            e = (n_date + pd.Timedelta(days=7)).replace(hour=6, minute=0, second=0)
            
            mask = (df_refined.index >= s) & (df_refined.index < e)
            df_refined.loc[mask, 'GPT_Score'] = row['gpt_score']

    # 6. 피처 엔지니어링
    df_ml = df_refined.copy()
    
    # 이동평균 및 변동성
    df_ml['MA_5'] = df_ml['Close'].rolling(5).mean()
    df_ml['MA_48'] = df_ml['Close'].rolling(48).mean()
    df_ml['Std_20'] = df_ml['Close'].rolling(20).std()
    
    # RSI (Relative Strength Index)
    delta = df_ml['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    
    # 0으로 나누기 방지
    rs = gain / loss.replace(0, np.nan)
    df_ml['RSI'] = 100 - (100 / (1 + rs))
    df_ml['RSI'] = df_ml['RSI'].fillna(50) # 계산 불가 시 중립값(50)
    
    # Lag 피처 (과거 데이터)
    df_ml['Close_Lag1'] = df_ml['Close'].shift(1)
    df_ml['Close_Lag2'] = df_ml['Close'].shift(2)
    df_ml['GPT_Lag1'] = df_ml['GPT_Score'].shift(1)
    
    # 시간 피처
    df_ml['Hour'] = df_ml.index.hour
    df_ml['DayOfWeek'] = df_ml.index.dayofweek
    
    # Target (Next Close)
    df_ml['Target'] = df_ml['Close'].shift(-1)
    
    return df_ml.dropna()
