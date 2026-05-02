# Chạy API & Docker (đơn giản)

## Env

Chỉ cần một file **`.env`** ở thư mục gốc repo (xem `.env.example`).

## Chạy local (Python)

```powershell
.med_venv\Scripts\python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

- Chat UI: http://localhost:8000/chat-ui/
- Docs: http://localhost:8000/docs

## Docker

**Điều kiện:** [Docker Desktop](https://docs.docker.com/desktop/) đang chạy.

**Bước 1 — có file `.env`:** copy mẫu rồi điền biến (đặc biết là khóa API nếu dùng LLM/Pinecone):

```powershell
copy .env.example .env
```

**Bước 2 — chạy bằng Compose (khuyến nghị):**

```powershell
docker compose up --build
```

Ở terminal khác hoặc trình duyệt:

- Chat UI: http://localhost:8000/chat-ui/
- Swagger: http://localhost:8000/docs

**Đổi cổng host:** trong `.env` đặt `APP_PORT=8080`, sau đó `docker compose up --build` — container vẫn lắng nghe cổng 8000 bên trong; Compose map `8080:8000`.

**Chạy image một lần (không Compose):**

```powershell
docker build -t medagent-api .
docker run --rm -p 8000:8000 --env-file .env medagent-api
```

**Lưu ý build:** `requirements.txt` có PyTorch / sentence-transformers — lần build đầu có thể lâu và image khá nặng. `.dockerignore` giúp giảm context gửi lên daemon.

**Google Calendar (`credentials.json`):** trong container là Linux — đường dẫn kiểu `C:\...` trong `.env` sẽ không tồn tại trong image. Khi chạy Docker, đặt ví dụ `GOOGLE_CREDENTIALS_PATH=/app/credentials.json` và mount file:

```powershell
docker run --rm -p 8000:8000 --env-file .env -v "${PWD}/credentials.json:/app/credentials.json:ro" medagent-api
```

(Tương tự có thể thêm khối `volumes` cho service `api` trong `docker-compose.yml` nếu bạn cần.)

**Next.js trong `frontend/`:** Dockerfile nằm ở `deploy/docker/frontend.Dockerfile`; hiện chưa gắn vào `docker-compose.yml` vì project UI Next cần hoàn thiện `package.json` / build. UI tĩnh dùng qua API `/chat-ui/` là đủ cho demo Docker.

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
