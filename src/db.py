import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# --------------------
# 1. DB 엔진 생성
# --------------------
def get_db_engine():
    """
    SQLAlchemy 엔진을 생성하여 반환합니다.
    """
    db_config = {
        "host": os.getenv("DB_HOST"),
        "port": int(os.getenv("DB_PORT", 3306)),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "db": os.getenv("DB_NAME", "projectl")
    }
    
    # DB URL 생성
    url = f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['db']}"
    
    return create_engine(url)

# --------------------
# 2. 데이터 로드 (가격, 공지사항)
# --------------------
def load_data(item_name, item_grade):
    """
    특정 아이템의 가격 로그와 GPT 공지 점수를 DB에서 가져옵니다.
    """
    engine = get_db_engine()
    
    try:
        # with 구문을 사용하여 자동으로 커넥션을 닫습니다 (Auto-close)
        with engine.connect() as conn:
            # 1. 아이템 ID 조회
            item_sql = text("SELECT id FROM market_items WHERE name = :name AND grade = :grade")
            item_id = conn.execute(item_sql, {"name": item_name, "grade": item_grade}).scalar()
            
            if not item_id:
                return None, None, None

            # 2. 가격 로그 조회
            price_sql = text("""
                SELECT logged_at, current_min_price
                FROM market_price_logs
                WHERE item_id = :item_id
                ORDER BY logged_at ASC
            """)
            df_prices = pd.read_sql(price_sql, conn, params={"item_id": item_id})
            
            # 3. GPT 공지 점수 조회
            notice_sql = text("""
                SELECT r.notice_date, i.gpt_score
                FROM item_notice_impacts i
                JOIN raw_notices r ON i.notice_id = r.id
                WHERE i.item_id = :item_id
                ORDER BY r.notice_date ASC
            """)
            df_notices = pd.read_sql(notice_sql, conn, params={"item_id": item_id})

            return df_prices, df_notices, item_id
            
    except Exception as e:
        print(f"[DB Error] load_data 실패: {e}")
        return None, None, None

# --------------------
# 3. 아이템 목록 로드 (검색용)
# --------------------
def load_merged_data():
    """
    검색 사이드바 구성을 위한 아이템 이름 및 등급 목록을 가져옵니다.
    """
    engine = get_db_engine()
    
    try:
        with engine.connect() as conn:
            sql = text("SELECT DISTINCT name, grade FROM market_items ORDER BY name, grade")
            df = pd.read_sql(sql, conn)
            return df
            
    except Exception as e:
        print(f"[DB Error] load_merged_data 실패: {e}")
        return pd.DataFrame(columns=['name', 'grade'])
