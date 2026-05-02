# Chạy API & Docker (đơn giản)

## Env

Chỉ cần một file **`.env`** ở thư mục gốc repo (xem `.env.example`).

## Chạy local

```bash
.med_venv\Scripts\python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

- Chat UI: http://localhost:8000/chat-ui/
- Docs: http://localhost:8000/docs

## Docker (image API)

```bash
docker build -t medagent-api .
docker run --rm -p 8000:8000 --env-file .env medagent-api
```

Compose có sẵn backend Next.js riêng — xem `docker-compose.yml`.

## Pytest

```bash
.med_venv\Scripts\python -m pytest tests/api/test_api_v1.py -q
```

Smoke (API đang chạy sẵn):

```bash
set RUN_LIVE_SMOKE=1
.med_venv\Scripts\python -m pytest tests/smoke/test_optional_live.py -q
```

## Ghi chú

- **`DOTENV_PATH`**: đường dẫn file env (mặc định `.env`). Tests đặt sẵn qua `tests/conftest.py`.
- HTTPS / reverse proxy: khi deploy thật mới cần (nginx, Caddy, …).
