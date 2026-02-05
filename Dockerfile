FROM python:3.10-slim

WORKDIR /app

# 1. 필수 패키지 설치 (git, procps 등 꼭 필요한 것만 남김)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    procps \
    && rm -rf /var/lib/apt/lists/*

# 한국 시간 설정
RUN ln -snf /usr/share/zoneinfo/Asia/Seoul /etc/localtime && echo Asia/Seoul > /etc/timezone

# 3. 라이브러리 설치 (CPU 전용 torch로 용량 반토막)
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

# 4. 전체 코드 복사
COPY . .

# 5. 실행 권한 부여 (반드시 필요!)
RUN chmod +x run.sh

EXPOSE 8501

# 실행!
CMD ["./run.sh"]
