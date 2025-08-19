from fastapi import FastAPI
from router import router

app = FastAPI(
    title="Text to Color API",
    description="텍스트를 기반으로 감정, 색상, 메시지를 추천하는 API",
    version="1.0.0",
)

# 라우터 등록
app.include_router(router, prefix="/api/v1", tags=["Bouquet"])
