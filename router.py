from fastapi import APIRouter
from model import BouquetRequest, BouquetResponse
from engine import recommend_bouquet_colors

router = APIRouter()

@router.post("/recommend", response_model=BouquetResponse)
def recommend_bouquet(request: BouquetRequest):
    """
    텍스트 입력을 기반으로 감정, 색상 팔레트, 메시지를 추천합니다.
    """
    result = recommend_bouquet_colors(
        when_text=request.when_text,
        actor=request.actor,
        recipient=request.recipient,
        relationship=request.relationship or "",
        history=request.history or "",
        recipient_gender=request.recipient_gender or "",
        rgb_target=request.rgb_target or 5
    )
    return result
