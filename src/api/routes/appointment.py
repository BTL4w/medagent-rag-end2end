from fastapi import APIRouter

router = APIRouter()


@router.post("/appointment")
def appointment() -> dict:
    return {"message": "todo"}
