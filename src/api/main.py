from fastapi import FastAPI

app = FastAPI(title="MedAgent RAG API")


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}
