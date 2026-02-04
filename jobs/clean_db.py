from sqlalchemy import create_engine, text
import pymysql
import os
from dotenv import load_dotenv

# 1. 환경변수 로드 (로컬: .env 읽음 / EC2: 시스템 환경변수 읽음)
load_dotenv()

# 2. DB 접속 정보 설정 (config.py 없이 직접 구성)
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT", 3306)),
    "user": os.getenv("DB_USER", "admin"),
    "password": os.getenv("DB_PASSWORD"),
    "db": os.getenv("DB_NAME", "projectl")
}

# 3. DB 엔진 생성
db_url = f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['db']}"
engine = create_engine(db_url)

def clean_dead_items(days=3):
    print(f"\n[DB Cleaner] 최근 {days}일간 업데이트 없는 '유령 아이템' 삭제 시작...")
    
    try:
        with engine.connect() as conn:
            # 1. 대상 확인
            check_sql = text(f"SELECT count(*) FROM market_items WHERE updated_at < DATE_SUB(NOW(), INTERVAL {days} DAY)")
            count = conn.execute(check_sql).scalar()
            
            if count == 0:
                print("삭제할 유령 아이템이 없습니다. DB가 깨끗합니다!")
                return

            print(f"발견된 유령 아이템: {count}개")

            # 2. 로그 삭제 (자식 데이터 먼저 삭제 - 외래키 제약 때문)
            print("   - 가격/뉴스 로그 삭제 중...")
            
            # (1) 가격 로그 삭제
            del_price_logs = text(f"""
                DELETE FROM market_price_logs 
                WHERE item_id IN (
                    SELECT id FROM market_items WHERE updated_at < DATE_SUB(NOW(), INTERVAL {days} DAY)
                )
            """)
            conn.execute(del_price_logs)
            
            # (2) 뉴스 영향 로그 삭제 (이것도 지워줘야 깔끔함)
            del_impact_logs = text(f"""
                DELETE FROM item_notice_impacts
                WHERE item_id IN (
                    SELECT id FROM market_items WHERE updated_at < DATE_SUB(NOW(), INTERVAL {days} DAY)
                )
            """)
            conn.execute(del_impact_logs)
            
            # 3. 아이템 삭제 (부모 데이터)
            print("   - 아이템 정보 삭제 중...")
            del_items_sql = text(f"DELETE FROM market_items WHERE updated_at < DATE_SUB(NOW(), INTERVAL {days} DAY)")
            conn.execute(del_items_sql)
            
            conn.commit()
            print("청소 완료!")
            
    except Exception as e:
        print(f"에러 발생: {e}")

if __name__ == "__main__":
    # 3일 이상 업데이트 안 된 아이템 삭제
    clean_dead_items(3)
    