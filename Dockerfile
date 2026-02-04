# 1. 베이스 이미지: 파이썬 3.10 버전 (가볍고 안정적)
FROM python:3.10-slim

# 2. 작업 폴더 설정
WORKDIR /app

# 3. 필수 시스템 패키지 설치
# [수정됨] procps 추가 -> 이제 컨테이너 안에서 'ps aux' 명령어가 먹힙니다!
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    git \
    procps \
    && rm -rf /var/lib/apt/lists/*

# 4. 한국 시간(KST) 설정
RUN ln -snf /usr/share/zoneinfo/Asia/Seoul /etc/localtime && echo Asia/Seoul > /etc/timezone

# 5. 라이브러리 설치 (캐시 활용)
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# 6. 나머지 모든 코드 복사 (run.sh 포함됨)
COPY . .

# [추가됨] run.sh 실행 권한 부여 (이거 없으면 'Permission denied' 에러 남)
RUN chmod +x run.sh

# 7. 포트 열기
EXPOSE 8501

# 8. 실행 명령어 변경
# [수정됨] 이제 streamlit을 직접 실행하는 게 아니라, run.sh 스크립트를 실행합니다.
CMD ["./run.sh"]
