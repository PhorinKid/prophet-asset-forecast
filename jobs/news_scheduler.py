import requests
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Date, Text, Float, TIMESTAMP, ForeignKey, func, text
from sqlalchemy.orm import declarative_base, sessionmaker
from bs4 import BeautifulSoup
import openai
import time
import json
from datetime import datetime
import config
from apscheduler.schedulers.blocking import BlockingScheduler
import os
import sys

try:
    # 1. 로컬 환경: config.py가 같은 폴더에 있다면 불러옵니다.
    import config
    print("✅ 로컬 설정 파일(config.py)을 발견하고 로드했습니다.")
    
    DB_CONFIG = config.DB_CONFIG
    LOSTARK_API_TOKEN = config.LOSTARK_API_TOKEN
    OPENAI_API_KEY = config.OPENAI_API_KEY
    TARGET_CATEGORIES = config.TARGET_CATEGORIES

except ImportError:
    # 2. 도커/EC2 환경: config.py가 없으므로 환경변수에서 직접 만듭니다.
    print("⚠️ config.py가 없습니다. 시스템 환경변수를 직접 사용합니다.")
    
    # .env 파일에서 읽어온 값들을 사용
    DB_CONFIG = {
        "host": os.getenv("DB_HOST"),
        "port": int(os.getenv("DB_PORT", 3306)), # 없으면 기본값 3306
        "user": os.getenv("DB_USER", "admin"),   # config.py에 있던 하드코딩 값 반영
        "password": os.getenv("DB_PASSWORD"),
        "db": os.getenv("DB_NAME", "projectl")   # config.py에 있던 하드코딩 값 반영
    }
    
    LOSTARK_API_TOKEN = os.getenv("LOSTARK_API_TOKEN")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # 카테고리는 보통 콤마(,)로 구분된 문자열로 관리하거나 기본값 사용
    # 도커 실행 시 TARGET_CATEGORIES 환경변수가 없으면 기본값([50000, 60000, 40000]) 사용
    cats_env = os.getenv("TARGET_CATEGORIES")
    if cats_env:
        TARGET_CATEGORIES = [int(x.strip()) for x in cats_env.split(',')]
    else:
        TARGET_CATEGORIES = [50000, 60000, 40000] # 포린님 config 파일의 기본값

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
    # [Expert ML Signal Prompt] 원본 프롬프트 적용
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

def run_scheduler():
    print(f"\n[Scheduler Job Started] {datetime.now()}")
    session = Session()

    try:
        api_url = 'https://developer-lostark.game.onstove.com/news/notices?type=공지&searchText=업데이트'
        headers = {'authorization': f'bearer {LOSTARK_API_TOKEN}'}
        res = requests.get(api_url, headers=headers)
        
        if res.status_code != 200:
            print("API 호출 실패")
            return

        notices = res.json()
        target_notices = notices[:3]

        for notice_data in target_notices:
            n_date = pd.to_datetime(notice_data['Date']).date()
            n_title = notice_data['Title']
            n_link = notice_data['Link']

            existing = session.query(RawNotice).filter_by(title=n_title, notice_date=n_date).first()
            notice_id = None
            content_text = ""

            if not existing:
                print(f"신규 공지: {n_title}")
                content_text = scrape_notice_content(n_link)
                new_notice = RawNotice(notice_date=n_date, title=n_title, link=n_link, content=content_text)
                session.add(new_notice)
                session.commit()
                notice_id = new_notice.id
            else:
                print(f"기존 공지: {n_title}")
                notice_id = existing.id
                content_text = existing.content

            # 아이템 분석
            items_sql = text("SELECT id, name FROM market_items WHERE category_code IN :cats")
            target_items = session.execute(items_sql, {"cats": tuple(TARGET_CATEGORIES)}).fetchall()

            print(f"-- 아이템 {len(target_items)}개 분석 체크...")
            
            analyze_count = 0
            for row in target_items:
                item_id = row[0]
                item_name = row[1]

                score_exists = session.query(ItemNoticeImpact).filter_by(notice_id=notice_id, item_id=item_id).first()
                
                if not score_exists:
                    result = get_gpt_score(n_title, content_text, item_name)
                    
                    impact = ItemNoticeImpact(
                        notice_id=notice_id,
                        item_id=item_id,
                        gpt_score=result['total_score'],
                        demand_pressure=result['demand_pressure'],
                        supply_pressure=result['supply_pressure'],
                        behavior_pressure=result['behavior_pressure'],
                        impact_days=result['impact_days']
                    )
                    session.add(impact)
                    session.commit()
                    analyze_count += 1
                    
                    if result['total_score'] != 0.0:
                        print(f"      ✨ {item_name}: {result['total_score']} ({result['reason']})")
                    
                    time.sleep(0.05)

            if analyze_count > 0:
                print(f"-- {analyze_count}개 신규 분석 완료")

    except Exception as e:
        print(f"에러: {e}")
    finally:
        session.close()
        print("스케줄러 종료")

if __name__ == "__main__":
    # 타임존 설정 (아주 잘하셨습니다!)
    scheduler = BlockingScheduler(timezone='Asia/Seoul')
    
    # 스케줄 등록: 매주 수요일 10시 01분
    scheduler.add_job(run_scheduler, 'cron', day_of_week='wed', hour=10, minute=1)
    
    print("⏰ 파이썬 스케줄러 설정 완료 (매주 수요일 10:01)")

    # ---------------------------------------------------------
    # [추가] 배포 직후 잘 돌아가는지 '지금 당장' 한 번 실행해보기
    # ---------------------------------------------------------
    print("🚀 [Self-Test] 서버 시작 직후 최초 1회 실행 중...")
    try:
        run_scheduler() # 여기서 에러 나면 바로 알 수 있음!
    except Exception as e:
        print(f"❌ 초기 실행 실패: {e}")

    # 스케줄러 시작 (Blocking)
    print("✅ 초기 실행 완료. 스케줄러 대기 모드 진입...")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        pass
