from engine import recommend_bouquet_colors

def generate_bouquet_response(data: dict) -> dict:
    """engine.py의 recommend_bouquet_colors 호출 후 결과 반환"""
    return recommend_bouquet_colors(
        when_text=data.get("when_text", ""),
        actor=data.get("actor", ""),
        recipient=data.get("recipient", ""),
        relationship=data.get("relationship", ""),
        history=data.get("history", ""),
        recipient_gender=data.get("recipient_gender", ""),
        rgb_target=data.get("rgb_target", 5)
    )
