import warnings
import logging
import os

# 1. 파이썬 기본 경고(FutureWarning 등) 무시
warnings.filterwarnings("ignore")

# 2. NeuralProphet 및 PyTorch Lightning 로그 끄기
logging.getLogger("NP.df_utils").setLevel(logging.ERROR)
logging.getLogger("NP.forecaster").setLevel(logging.ERROR)

# 3. 환경 변수로 텐서플로우/토치 관련 로그 최소화
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from src.db import load_data
from src.processing import preprocess_data
from src.models import ModelFactory
import os


ITEM_NAME = "운명의 파괴석"

# --------------------
# 1. 예측
# --------------------
def main():
    # 1. 데이터 로드 (ID 포함)
    df_prices, df_notices, item_id = load_data(ITEM_NAME)
    if item_id is None: return
    
    factory = ModelFactory()
    model_dir = factory._get_model_path(item_id)

    # 2. 모델 존재 여부 확인
    if not os.path.exists(os.path.join(model_dir, "lgbm_model.pkl")):
        print(f"\n{ITEM_NAME} 모델이 없습니다. 학습을 시작")

        # [학습 단계] 2월 4일 이전 데이터로 학습 진행
        from datetime import datetime
        TRAIN_CUTOFF = "2026-02-04 06:00:00"
        df_prices_train = df_prices[df_prices['logged_at'] < TRAIN_CUTOFF].copy()
        
        df_ml_train = preprocess_data(df_prices_train, df_notices)
        factory.train_all(df_ml_train)
        factory.save_models(item_id)
        print(f"학습 및 저장 완료")

    print(f"\n{ITEM_NAME} 예측 모드 진입")
    factory.load_models(item_id)
    
    df_ml_now = preprocess_data(df_prices, df_notices)
    # 미래 3일 예측 수행
    forecast = factory.predict_future_3days(df_ml_now, item_id, df_notices)
    
    print(forecast.head())

if __name__ == "__main__":
    main()
