import requests
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Date, Text, Float, TIMESTAMP, ForeignKey, func, text
from sqlalchemy.orm import declarative_base, sessionmaker
from bs4 import BeautifulSoup
import openai
import time
import json
from datetime import date
import os
import sys
from dotenv import load_dotenv

# ==============================================================================
# [1] 설정 로드 (Unified Config)
# ==============================================================================

load_dotenv()

print("✅ 환경변수 설정을 로드합니다.")

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT", 3306)),
    "user": os.getenv("DB_USER", "admin"),
    "password": os.getenv("DB_PASSWORD"),
    "db": os.getenv("DB_NAME", "projectl")
}

LOSTARK_API_TOKEN = os.getenv("LOSTARK_API_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TARGET_CATEGORIES = [50000, 60000, 40000]

# 12월 17일부터 수집
TARGET_START_DATE = date(2025, 12, 17)

# ==============================================================================
# [2] 초기화
# ==============================================================================

db_url = f'mysql+pymysql://{DB_CONFIG["user"]}:{DB_CONFIG["password"]}@{DB_CONFIG["host"]}:{DB_CONFIG["port"]}/{DB_CONFIG["db"]}'
engine = create_engine(db_url)
Session = sessionmaker(bind=engine)
Base = declarative_base()

class RawNotice(Base):
    __tablename__ = 'raw_notices'
    id = Column(Integer, primary_key=True, autoincrement=True)
    notice_date = Column(Date, nullable=False)
    title = Column(String(255), nullable=False)
    link = Column(String(500), nullable=False)
    content = Column(Text, nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now())

class ItemNoticeImpact(Base):
    __tablename__ = 'item_notice_impacts'
    id = Column(Integer, primary_key=True, autoincrement=True)
    notice_id = Column(Integer, ForeignKey('raw_notices.id'), nullable=False)
    item_id = Column(Integer, nullable=False)
    gpt_score = Column(Float, default=0.0)
    demand_pressure = Column(Float, default=0.0)
    supply_pressure = Column(Float, default=0.0)
    behavior_pressure = Column(Float, default=0.0)
    impact_days = Column(Integer, default=0)
    analyzed_at = Column(TIMESTAMP, server_default=func.now())

client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ==============================================================================
# [3] 로직
# ==============================================================================

def scrape_notice_content(url):
    """ 본문 전체 수집 (60,000자) """
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            content_div = soup.find('div', class_='fr-view')
            if not content_div:
                content_div = soup.find('div', class_='article-content')
            
            if content_div:
                text = content_div.get_text(separator='\n')
                clean_text = '\n'.join([line.strip() for line in text.splitlines() if line.strip()])
                return clean_text[:60000]
        return ""
    except Exception as e:
        print(f"크롤링 에러: {e}")
        return ""

def get_gpt_score(title, content, item_name):
    prompt = f"""
    You are an economic signal generator for a time-series prediction model.

    Your task is NOT to summarize patch notes,
    but to convert game announcements into quantitative market signals
    that can be used as exogenous variables in ML/DL models.

    Target item: {item_name}

    Time horizon:
    - Short-term impact only (1 to 7 days)
    - Ignore long-term meta effects

    Based on the announcement below, estimate the following signals:

    1. Demand pressure on the item
    2. Supply pressure on the item
    3. Behavioral / sentiment pressure (panic buy, sell-off, hoarding)

    Scoring rules:
    - Each signal must be a float between -1.0 and 1.0
    - Positive = price upward pressure
    - Negative = price downward pressure
    - 0.0 = no meaningful impact

    If the announcement is mainly bug fixes or maintenance:
    - All signals must be 0.0

    Also estimate:
    4. Impact duration (number of days, integer 0–7)

    Announcement title:
    {title}

    Announcement content:
    {content[:60000]}

    Output JSON ONLY in the following format:
    {{
        "demand_pressure": <float>,
        "supply_pressure": <float>,
        "behavior_pressure": <float>,
        "impact_days": <int>,
        "total_score": <float>,
        "reason": "brief causal explanation (Must be written in KOREAN)"
    }}

    Rules:
    - total_score must be a weighted result of the three pressures
    - Do NOT speculate beyond the announcement text
    - If uncertain, reduce magnitude toward 0
    - IMPORTANT: The 'reason' field MUST be written in Korean.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        clean_json = response.choices[0].message.content.strip().replace("```json", "").replace("```", "")
        return json.loads(clean_json)
    except Exception as e:
        print(f"GPT Error: {e}")
        return {
            "demand_pressure": 0.0, "supply_pressure": 0.0, "behavior_pressure": 0.0,
            "impact_days": 0, "total_score": 0.0, "reason": "Error"
        }

def load_history():
    print(f"\n[History Loader] 경제 시그널 생성 모드 시작 (12/17 ~ )")
    session = Session()
    page = 1
    is_running = True

    try:
        while is_running:
            print(f"\nPage {page} 검색 중...")
            api_url = f'https://developer-lostark.game.onstove.com/news/notices?type=공지&searchText=업데이트&page={page}'
            headers = {'authorization': f'bearer {LOSTARK_API_TOKEN}'}
            res = requests.get(api_url, headers=headers)
            
            notices = res.json()
            if not notices:
                break

            for notice_data in notices:
                n_date = pd.to_datetime(notice_data['Date']).date()
                n_title = notice_data['Title']
                
                if n_date < TARGET_START_DATE:
                    print(f"-- 기준 날짜 도달 ({n_date}). 종료합니다.")
                    is_running = False
                    break
                
                # 1. 공지 저장
                existing = session.query(RawNotice).filter_by(title=n_title, notice_date=n_date).first()
                notice_id = None
                content_text = ""

                if not existing:
                    content_text = scrape_notice_content(notice_data['Link'])
                    new_notice = RawNotice(notice_date=n_date, title=n_title, link=notice_data['Link'], content=content_text)
                    session.add(new_notice)
                    session.commit()
                    notice_id = new_notice.id
                    print(f"[{n_date}] {n_title} (신규 저장)")
                else:
                    print(f"[{n_date}] {n_title} (이미 있음)")
                    notice_id = existing.id
                    content_text = existing.content

                # 2. 아이템 분석
                items_sql = text("SELECT id, name FROM market_items WHERE category_code IN :cats")
                target_items = session.execute(items_sql, {"cats": tuple(TARGET_CATEGORIES)}).fetchall()
                
                analyze_count = 0
                for row in target_items:
                    item_id = row[0]
                    item_name = row[1]
                    
                    # 이미 분석했는지 확인
                    score_exists = session.query(ItemNoticeImpact).filter_by(notice_id=notice_id, item_id=item_id).first()
                    
                    if not score_exists:
                        # [핵심] 5가지 데이터 받아오기
                        result = get_gpt_score(n_title, content_text, item_name)
                        
                        # DB 저장 (컬럼 맵핑)
                        impact = ItemNoticeImpact(
                            notice_id=notice_id,
                            item_id=item_id,
                            gpt_score=result['total_score'],       # 종합 점수
                            demand_pressure=result['demand_pressure'], # 수요
                            supply_pressure=result['supply_pressure'], # 공급
                            behavior_pressure=result['behavior_pressure'], # 심리
                            impact_days=result['impact_days']      # 지속 기간
                        )
                        session.add(impact)
                        session.commit()
                        analyze_count += 1
                        
                        # 로그 출력 (0점이 아니면 자세히)
                        if result['total_score'] != 0.0:
                            print(f"{item_name}: {result['total_score']}")
                            print(f"D(수요):{result['demand_pressure']} | S(공급):{result['supply_pressure']} | B(심리):{result['behavior_pressure']}")
                            print(f"지속:{result['impact_days']}일 | {result['reason']}")
                        
                        time.sleep(0.05)
                
                if analyze_count > 0:
                    print(f"-- {analyze_count}개 아이템 분석 완료")

            page += 1

    except Exception as e:
        print(f"에러 발생: {e}")
    finally:
        session.close()
        print("작업 종료")

if __name__ == "__main__":
    load_history()
