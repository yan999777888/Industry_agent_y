FROM python:3.11-slim

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 项目代码
COPY src/ src/
COPY scripts/ scripts/
COPY Knowledge_base/ Knowledge_base/
COPY data/processed/kb/ data/processed/kb/
COPY pyproject.toml .

# 环境变量（比赛时在 docker run 时传入）
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["uvicorn", "industry_agent.api.app:create_app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "2", \
     "--timeout-keep-alive", "35"]
