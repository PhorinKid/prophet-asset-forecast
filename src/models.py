import os
import joblib
import pandas as pd
import numpy as np
import shutil
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from neuralprophet import NeuralProphet, set_random_seed

FEATURES = ['MA_5', 'MA_48', 'Std_20', 'RSI', 'Close_Lag1', 'Close_Lag2', 'GPT_Lag1', 'Hour', 'DayOfWeek']

class ModelFactory:
    def __init__(self, base_path="models"):
        self.base_path = base_path
        self.lgbm = None
        self.xgb = None
        self.np_model = None

    def _get_model_path(self, item_id):
        return os.path.join(self.base_path, str(item_id))
    
    def train_all(self, df_ml):
        print("Model training started...")
        
        X = df_ml[FEATURES]
        y = df_ml['Target']
        
        # 1. LightGBM (Memory Safe Mode)
        self.lgbm = LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.01,
            max_depth=5,
            num_leaves=31,
            random_state=42,
            n_jobs=1,  # CPU 과부하 방지
            verbose=-1
        )
        self.lgbm.fit(X, y)
        
        # 2. XGBoost (Memory Safe Mode)
        self.xgb = XGBRegressor(
            n_estimators=2000,
            learning_rate=0.01,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=1   # CPU 과부하 방지
        )
        self.xgb.fit(X, y, verbose=False)
        
        # 3. NeuralProphet
        df_np = df_ml.reset_index().rename(columns={'logged_at': 'ds', 'Close': 'y'})[['ds', 'y', 'GPT_Score']]
        df_np['ds'] = pd.to_datetime(df_np['ds'])
        
        set_random_seed(42)
        self.np_model = NeuralProphet(
            n_forecasts=144,
            n_lags=240,
            weekly_seasonality=True,
            daily_seasonality=True,
            yearly_seasonality=False,
            learning_rate=0.01,
            batch_size=64,
            growth='off',
            trainer_config={
                "enable_checkpointing": False,
                "logger": False,
                "enable_progress_bar": False
            }
        )
        self.np_model.add_future_regressor("GPT_Score")
        self.np_model.fit(df_np, freq="30min", progress=None, num_workers=0)

        if os.path.exists("lightning_logs"):
            shutil.rmtree("lightning_logs")
        
        print("Training completed.")
    
    def save_models(self, item_id):
        save_path = self._get_model_path(item_id)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        joblib.dump(self.lgbm, f"{save_path}/lgbm_model.pkl")
        joblib.dump(self.xgb, f"{save_path}/xgb_model.pkl")
        joblib.dump(self.np_model, f"{save_path}/np_model.pkl")
        print(f"Models saved to {save_path}")

    def load_models(self, item_id):
        load_path = self._get_model_path(item_id)
        if os.path.exists(load_path):
            self.lgbm = joblib.load(f"{load_path}/lgbm_model.pkl")
            self.xgb = joblib.load(f"{load_path}/xgb_model.pkl")
            self.np_model = joblib.load(f"{load_path}/np_model.pkl")
        else:
            raise FileNotFoundError(f"No model found for Item ID: {item_id}")

    def predict_future_3days(self, df_ml, item_id, df_notices):
        last_time = df_ml.index[-1]
        future_index = pd.date_range(start=last_time + pd.Timedelta(minutes=30), periods=144, freq='30min')
        
        # NeuralProphet Prediction
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

        future = self.np_model.make_future_dataframe(
            df_recent,
            periods=forecast_steps,
            n_historic_predictions=False,
            regressors_df=future_regressors
        )
        
        forecast_np_raw = self.np_model.predict(future)
        future_rows = forecast_np_raw[forecast_np_raw['y'].isnull()].copy()
        
        valid_preds_np = []
        for i in range(1, len(future_rows) + 1):
            col_name = f'yhat{i}'
            val = future_rows.iloc[i-1][col_name] if col_name in future_rows.columns else future_rows.iloc[i-1]['yhat1']
            valid_preds_np.append(val)
        
        preds_np = np.array(valid_preds_np)
        if len(preds_np) < 144:
            last_val = preds_np[-1] if len(preds_np) > 0 else 0
            padding = [last_val] * (144 - len(preds_np))
            preds_np = np.concatenate([preds_np, padding])
        
        # ML Recursive Prediction
        preds_lgbm = self._recursive_predict(self.lgbm, df_ml, steps=144)
        preds_xgb = self._recursive_predict(self.xgb, df_ml, steps=144)
        
        # Ensemble (Weight: LGBM 55%, XGB 15%, NP 30%)
        final_forecast = (preds_lgbm * 0.4) + (preds_xgb * 0.4) + (preds_np * 0.2)
        
        result_df = pd.DataFrame({
            'ds': future_index,
            'forecast': final_forecast.astype(int),
            'lgbm': preds_lgbm,
            'xgb': preds_xgb,
            'neuralprophet': preds_np
        })
        
        return result_df
    
    def _recursive_predict(self, model, df_initial, steps=144):
        future_preds = []
        history_prices = df_initial['Close'].iloc[-200:].tolist()
        
        last_row = df_initial.iloc[[-1]].copy()
        current_time = last_row.index[0]
        
        for i in range(steps):
            pred_price = model.predict(last_row[FEATURES])[0]
            future_preds.append(pred_price)
            
            next_time = current_time + pd.Timedelta(minutes=30)
            
            history_prices.append(pred_price)
            if len(history_prices) > 200:
                history_prices.pop(0)
            
            s_history = pd.Series(history_prices)
            
            last_row['MA_5'] = s_history.rolling(5).mean().iloc[-1]
            last_row['MA_48'] = s_history.rolling(48).mean().iloc[-1]
            last_row['Std_20'] = s_history.rolling(20).std().iloc[-1]
            
            delta = s_history.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean().iloc[-1]
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
            
            if loss == 0:
                last_row['RSI'] = 100
            else:
                last_row['RSI'] = 100 - (100 / (1 + (gain / loss)))
            
            last_row['Close_Lag2'] = last_row['Close_Lag1']
            last_row['Close_Lag1'] = pred_price
            
            last_row['Hour'] = next_time.hour
            last_row['DayOfWeek'] = next_time.dayofweek
            last_row['GPT_Lag1'] = 0.0
            
            current_time = next_time
            
        return np.array(future_preds)
    