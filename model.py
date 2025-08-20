from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class BouquetRequest(BaseModel):
    when_text: str
    actor: str
    recipient: str
    relationship: Optional[str] = None
    history: Optional[str] = None
    recipient_gender: Optional[str] = None
    rgb_target: Optional[int] = 5

class BouquetResponse(BaseModel):
    #emotion: str
    # allowed_emotions: List[str]
    # relation: Dict[str, str]
    # base_colors: List[str]
    # accent_colors: List[str]
    # hex: Dict[str, List[str]]
    # rgb: Dict[str, List[List[int]]]
    rgb_selected: List[List[int]]
    # rgb_compact: str
    # avoid: List[str]
    # rationale: str
    message: str
    # raw: Dict[str, Any]
