import os
import joblib
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from neuralprophet import NeuralProphet, set_random_seed
import shutil

FEATURES = ['MA_5', 'MA_48', 'Std_20', 'RSI', 'Close_Lag1', 'Close_Lag2', 'GPT_Lag1', 'Hour', 'DayOfWeek']

class ModelFactory:
    def __init__(self, base_path="models"):
        self.base_path = base_path
        self.lgbm = None
        self.xgb = None
        self.np_model = None

    # --------------------
    # 저장 폴더 이름
    # --------------------
    def _get_model_path(self, item_id):
        return os.path.join(self.base_path, str(item_id))
    
    # --------------------
    # 1. 모델 학습
    # --------------------
    def train_all(self, df_ml):
        print("\n모델 학습 시작")
        
        # 데이터 분리
        X = df_ml[FEATURES]
        y = df_ml['Target']
        
        # 1. LightGBM
        print("LightGBM 학습 중")
        self.lgbm = LGBMRegressor(n_estimators=1000, learning_rate=0.01, verbose=-1, random_state=42)
        self.lgbm.fit(X, y)
        
        # 2. XGBoost
        print("XGBoost 학습 중")
        self.xgb = XGBRegressor(n_estimators=1000, learning_rate=0.01, random_state=42)
        self.xgb.fit(X, y)
        
        # 3. NeuralProphet (딥러닝)
        print("NeuralProphet 학습 중")
        # NP 전용 데이터 포맷 변환
        df_np = df_ml.reset_index().rename(columns={'logged_at': 'ds', 'Close': 'y'})[['ds', 'y', 'GPT_Score']]
        df_np['ds'] = pd.to_datetime(df_np['ds'])
        
        set_random_seed(42)
        self.np_model = NeuralProphet(
            n_forecasts=1,
            n_lags=24,
            learning_rate=0.01,
            epochs=100,
            batch_size=64,
            trainer_config={
                "enable_checkpointing": False,  # 체크포인트 저장 안 함
                "logger": False,                # 로그 파일 생성 안 함
                "enable_progress_bar": False    # 진행바 끄기 (EC2 로그 깔끔하게)
            }
        )
        self.np_model.add_future_regressor("GPT_Score")
        self.np_model.fit(df_np, freq="30min", progress=None)

        if os.path.exists("lightning_logs"):
            shutil.rmtree("lightning_logs")
        
        print("모델 학습 완료\n")
    
    # --------------------
    # 2. 모델 저장
    # --------------------
    def save_models(self, item_id):
        print("\n모델 저장 시작")

        save_path = self._get_model_path(item_id)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        joblib.dump(self.lgbm, f"{save_path}/lgbm_model.pkl")
        joblib.dump(self.xgb, f"{save_path}/xgb_model.pkl")
        joblib.dump(self.np_model, f"{save_path}/np_model.pkl")

        print(f"[Item ID: {item_id}] 모델 저장 완료 -> {save_path}\n")

    # --------------------
    # 3. 모델 로드
    # --------------------
    def load_models(self, item_id):
        print(f"\n모델 로딩 시작 ({item_id})")

        load_path = self._get_model_path(item_id)

        if os.path.exists(load_path):
            self.lgbm = joblib.load(f"{load_path}/lgbm_model.pkl")
            self.xgb = joblib.load(f"{load_path}/xgb_model.pkl")
            self.np_model = joblib.load(f"{load_path}/np_model.pkl")
            print(f"[Item ID: {item_id}] 모델 로드 완료\n")
        else:
            raise FileNotFoundError(f"[ID : {item_id}]에 해당하는 모델 파일이 없습니다.\n")

    # --------------------
    # 4. 앙상블
    # --------------------
    def predict_ensemble(self, input_row, np_df):
        """
        [예측 서비스용] 3대장 앙상블 예측값을 반환합니다.
        input_row: 머신러닝용 피처가 들어있는 DataFrame (1줄)
        np_df: NeuralProphet용 히스토리 데이터
        """
        # 1. ML 예측
        p_lgbm = self.lgbm.predict(input_row[FEATURES])[0]
        p_xgb = self.xgb.predict(input_row[FEATURES])[0]
        
        # 2. NP 예측
        forecast = self.np_model.predict(np_df)
        p_np = forecast['yhat1'].iloc[-1]
        
        # 3. 앙상블 (55 : 35 : 10)
        final_price = (p_lgbm * 0.55) + (p_xgb * 0.35) + (p_np * 0.10)
        
        return int(final_price), p_lgbm, p_xgb, p_np
    
    # --------------------
    # 5. 3일 예측
    # --------------------
    def predict_future_3days(self, df_ml, item_id, df_notices):
        print(f"\n[Item {item_id}] 향후 3일간의 가격 흐름 분석 시작")
        
        # 1. 미래 시간표 생성 (30분 단위, 144개 구간)
        last_time = df_ml.index[-1]
        future_index = pd.date_range(start=last_time + pd.Timedelta(minutes=30), periods=144, freq='30min')
        
        # A. NeuralProphet 미래 예측
        df_recent = df_ml.reset_index().rename(columns={'logged_at': 'ds', 'Close': 'y'})[['ds', 'y', 'GPT_Score']]
        
        forecast_steps = 144
        future_reg_values = [0.0] * forecast_steps
        if df_notices is not None and not df_notices.empty:
            for i, f_time in enumerate(future_index):
                for _, row in df_notices.iterrows():
                    n_date = pd.to_datetime(row['notice_date'])
                    s = n_date.replace(hour=10, minute=0, second=0)
                    e = (n_date + pd.Timedelta(days=7)).replace(hour=6, minute=0, second=0)
                    if s <= f_time < e:
                        future_reg_values[i] = row['gpt_score']
        
        future_regressors = pd.DataFrame({'GPT_Score': future_reg_values})

        # 미래 데이터프레임 생성 및 예측
        future = self.np_model.make_future_dataframe(
            df_recent,
            periods=forecast_steps,
            n_historic_predictions=False,
            regressors_df=future_regressors
        )
        # 1. NeuralProphet 예측 실행
        # (중요: make_future_dataframe으로 만든 데이터는
        #  학습된 n_forecasts와 상관없이 시퀀스 전체를 담고 있습니다.)
        forecast_np_raw = self.np_model.predict(future)
        
        # 2. NP 결과에서 yhat 추출 (ValueError 방지를 위한 길이 체크 로직)
        # 미래 행들만 필터링
        future_rows = forecast_np_raw[forecast_np_raw['y'].isnull()].copy()
        
        valid_preds_np = []
        for i in range(1, len(future_rows) + 1):
            # 모델 설정에 따라 yhat1만 나올 수도, yhat1~144가 나올 수도 있습니다.
            # 가장 안전하게 yhat1을 우선적으로 가져오되, 없으면 가능한 값을 가져옵니다.
            col_name = f'yhat{i}'
            if col_name in future_rows.columns:
                val = future_rows.iloc[i-1][col_name]
            else:
                # yhat{i}가 없으면 가장 기본인 yhat1을 사용합니다.
                val = future_rows.iloc[i-1]['yhat1']
            valid_preds_np.append(val)
        
        preds_np = np.array(valid_preds_np)

        # [추가 체크] 만약 preds_np 길이가 144보다 작다면 부족한 만큼 마지막 값으로 채움
        if len(preds_np) < 144:
            last_val = preds_np[-1] if len(preds_np) > 0 else 0
            padding = [last_val] * (144 - len(preds_np))
            preds_np = np.concatenate([preds_np, padding])
        
        # B. ML 모델(LGBM, XGB) 미래 예측
        future_ml = pd.DataFrame(index=future_index)
        future_ml['Hour'] = future_ml.index.hour
        future_ml['DayOfWeek'] = future_ml.index.dayofweek

        for col in ['MA_5', 'MA_48', 'Std_20', 'RSI', 'Close_Lag1', 'Close_Lag2']:
            future_ml[col] = df_ml[col].iloc[-1]
        future_ml['GPT_Lag1'] = future_reg_values

        preds_lgbm = self.lgbm.predict(future_ml[FEATURES])
        preds_xgb = self.xgb.predict(future_ml[FEATURES])
        
        # C. 3대장 앙상블
        final_forecast = (preds_lgbm * 0.55) + (preds_xgb * 0.35) + (preds_np * 0.10)
        
        # 결과 정리
        result_df = pd.DataFrame({
            'ds': future_index,
            'forecast': final_forecast.astype(int),
            'lgbm': preds_lgbm,
            'xgb': preds_xgb,
            'np': preds_np
        })
        print("미래 3일 예측 완료\n")
        
        return result_df
    