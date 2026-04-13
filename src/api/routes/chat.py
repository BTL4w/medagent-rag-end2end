from fastapi import APIRouter

router = APIRouter()


@router.post("/chat")
def chat() -> dict:
    return {"message": "todo"}
