import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# --------------------
# 상위 폴더의 .env 로드
# --------------------
load_dotenv()

# --------------------
# rds db 연결
# --------------------
def get_db_engine():
    # print("\ndb 연결 시작")

    db_config = {
        "host": os.getenv("DB_HOST"),
        "port": int(os.getenv("DB_PORT", 3306)),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "db": os.getenv("DB_NAME", "projectl")
    }
    url = f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['db']}"

    print("db 연결 성공\n")
    return create_engine(url)

# --------------------
# 2. 아이템 (이름, 등급) 으로 id, 가격 로그, 공지사항, gpt점수 수집
# --------------------
def load_data(item_name, item_grade):
    # print("\n데이터 검색 시작")

    engine = get_db_engine()
    conn = engine.connect()
    
    try:
        # 1. 아이템 ID 찾기
        item_sql = text("SELECT id FROM market_items WHERE name = :name AND grade = :grade")
        params = {"name": item_name, "grade": item_grade}

        item_id = conn.execute(item_sql, params).scalar()
        
        if not item_id:
            print(f"'{item_name}' [{item_grade}] 아이템을 DB에서 찾을 수 없습니다.")
            return None, None, None

        # 2. 가격 로그
        price_sql = text("SELECT logged_at, current_min_price FROM market_price_logs WHERE item_id = :item_id ORDER BY logged_at ASC")
        df_prices = pd.read_sql(price_sql, conn, params={"item_id": item_id})
        
        # 3. GPT 공지 점수
        notice_sql = text("""
            SELECT r.notice_date, i.gpt_score
            FROM item_notice_impacts i
            JOIN raw_notices r ON i.notice_id = r.id
            WHERE i.item_id = :item_id ORDER BY r.notice_date ASC
        """)
        df_notices = pd.read_sql(notice_sql, conn, params={"item_id": item_id})

        print("데이터 검색 성공\n")
        return df_prices, df_notices, item_id
        
    except Exception as e:
        print(f"[DB Error] 데이터 상세 로드 중 오류: {e}")
        return None, None, None

    finally:
        conn.close()

# --------------------
# 3. 이름, 등급 리스트
# --------------------
def load_merged_data():
    engine = get_db_engine()
    conn = engine.connect()
    
    try:
        sql = text("SELECT DISTINCT name, grade FROM market_items ORDER BY name, grade")
        df = pd.read_sql(sql, conn)
        return df
        
    except Exception as e:
        print(f"[DB Error] 아이템 목록 로드 실패: {e}")
        return pd.DataFrame(columns=['name', 'grade'])
        
    finally:
        conn.close()

# --------------------
# 4. 공지사항 가져오기
# --------------------
def load_gpt_scores():
    """
    전체 공지사항의 GPT 분석 점수를 가져옵니다.
    (필요 없다면 이 함수는 생략해도 됩니다)
    """
    return pd.DataFrame()
