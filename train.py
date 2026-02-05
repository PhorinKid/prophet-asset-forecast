from src.db import load_data
from src.processing import preprocess_data
from src.models import ModelFactory

ITEM_NAME = "운명의 파괴석"

# --------------------
# 1. 데이터 로드
# --------------------
def main():
    # 1. 데이터 가져오기 (이제 ID도 받아옵니다)
    df_prices, df_notices, item_id = load_data(ITEM_NAME)
    
    if item_id is None:
        return

    # 2. 전처리
    df_ml = preprocess_data(df_prices, df_notices)

    # 3. 모델 학습
    factory = ModelFactory()
    factory.train_all(df_ml)
    
    # 4. 저장
    factory.save_models(item_id)
    
    print(f"{ITEM_NAME}(ID:{item_id}) 모델 업데이트 완료\n")

if __name__ == "__main__":
    main()
