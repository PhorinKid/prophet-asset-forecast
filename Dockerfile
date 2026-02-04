FROM python:3.10-slim

WORKDIR /app

# procps 추가 (ps 명령어용)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    git \
    procps \
    && rm -rf /var/lib/apt/lists/*

# 한국 시간 설정
RUN ln -snf /usr/share/zoneinfo/Asia/Seoul /etc/localtime && echo Asia/Seoul > /etc/timezone

# 라이브러리 설치
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# 코드 복사
COPY . .

# 실행 권한 부여
RUN chmod +x run.sh

EXPOSE 8501

CMD ["./run.sh"]
