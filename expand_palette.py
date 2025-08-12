# expand_palette.py  — Colormind 중심 (보색 제거), engine.py 수정 불필요
from functools import lru_cache
from typing import List, Dict, Any
from name2hex import to_hex
from palette_fetchers import get_palette_from_colormind, get_scheme_from_thecolorapi
from color_utils import unique_palette, is_grayish

# 무채색 seed일 때나 외부 실패 시, 계절감 유지용 안전 팔레트
NEUTRAL_SEASON_ACCENTS = {
    "봄": ["#FFF3B0", "#FFD1DC", "#B0C4DE"],    # 파스텔 옐로, 핑크, 라이트 스틸 블루
    "여름": ["#FFFFFF", "#87CEEB", "#98FF98"],  # 화이트, 스카이블루, 민트
    "가을": ["#F4A460", "#8D6E63", "#D4AF37"],  # 샌디브라운, 웜브라운, 머스터드/골드
    "겨울": ["#8B0000", "#01796F", "#2C3E50"],  # 딥레드, 파인그린, 네이비
}

def _upper_set(xs: List[str]) -> set:
    return {x.upper() for x in xs if x}

def _safe_unique(hexes: List[str], tol: int = 18) -> List[str]:
    try:
        return unique_palette(hexes, tol=tol)
    except Exception:
        seen, out = set(), []
        for h in hexes:
            u = (h or "").upper()
            if u and u not in seen:
                seen.add(u); out.append(h)
        return out

@lru_cache(maxsize=256)
def _cached_colormind(seed: str) -> tuple:
    try:
        res = get_palette_from_colormind(seed)
        return tuple(res or [])
    except Exception:
        return tuple()

@lru_cache(maxsize=256)
def _cached_analogous_from_api(seed: str) -> tuple:
    """Colormind가 비었을 때 최소한의 조화 팔레트(Analogous) 보조용."""
    try:
        res = get_scheme_from_thecolorapi(seed, mode="analogic", count=5)
        return tuple(res or [])
    except Exception:
        return tuple()

def expand_colors_from_external(seasonal_colors: List[str], season: str, use_colormind: bool = True) -> Dict[str, Any]:
    """
    엔진과 호환되는 반환:
      {"base_hex": [...], "complement": [], "analogous": [...], "colormind": [...]}
    - 보색(complement)은 항상 비움(꽃 추천용 조화 중심)
    - analogous에는 Colormind 결과를 채워 엔진의 로컬 하모니 fallback을 방지
    """
    # 1) 입력 정규화 (이름→HEX, 불량 스킵)
    base_hex: List[str] = []
    for c in seasonal_colors or []:
        try:
            hx = to_hex(c)
            if hx:
                base_hex.append(hx)
        except Exception:
            continue

    out: Dict[str, Any] = {"base_hex": base_hex, "complement": [], "analogous": [], "colormind": []}
    if not base_hex:
        return out

    seed = base_hex[0]  # Colormind는 하나만 seed로 사용
    seeds_upper = _upper_set([seed])

    # 2) Colormind 기반 팔레트 (1순위)
    cm = []
    if use_colormind:
        cm = list(_cached_colormind(seed))

    # 3) Colormind 실패/빈 결과 시 보조: 무채색이면 계절 액센트, 아니면 analogic API
    if not cm:
        if is_grayish(seed):
            cm = NEUTRAL_SEASON_ACCENTS.get(season, [])
        else:
            cm = list(_cached_analogous_from_api(seed))

    # 4) seed 자체 제외 + 근접 중복 제거
    cm = [h for h in cm if h and h.upper() not in seeds_upper]
    cm_u = _safe_unique(cm, tol=18)

    # 5) 반환 구성
    out["colormind"] = cm_u                         # 레퍼런스를 위해 원본 소스도 전달
    out["analogous"] = cm_u[:3] or cm_u             # 엔진 fallback 방지용으로 analogous 채움
    out["complement"] = []                          # 보색 완전 제거

    return out
