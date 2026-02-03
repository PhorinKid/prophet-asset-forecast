# 1. 베이스 이미지: 파이썬 3.10 버전 (가볍고 안정적)
FROM python:3.10-slim

# 2. 작업 폴더 설정
WORKDIR /app

# 3. 필수 시스템 패키지 설치 (XGBoost, LightGBM 등을 위해 필요)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# 4. 한국 시간(KST) 설정 (이거 안 하면 시간이 9시간 차이 남!)
RUN ln -snf /usr/share/zoneinfo/Asia/Seoul /etc/localtime && echo Asia/Seoul > /etc/timezone

# 5. 라이브러리 설치 (requirements.txt 먼저 복사해서 캐시 활용)
COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# 6. 나머지 모든 코드 복사
COPY . .

# 7. 포트 열기 (Streamlit 기본 포트)
EXPOSE 8501

# 8. 실행 명령어: Streamlit 앱 실행
# (주의: 스케줄러가 아니라 웹 대시보드를 메인으로 띄웁니다)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
