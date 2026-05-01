from fastapi import FastAPI

from src.api.routes.appointment import router as appointment_router
from src.api.routes.chat import router as chat_router
from src.api.routes.system import router as system_router

app = FastAPI(title="MedAgent RAG API")
app.include_router(system_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")
app.include_router(appointment_router, prefix="/api/v1")

