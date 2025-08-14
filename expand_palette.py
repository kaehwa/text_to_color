# expand_palette.py  — Colormind 중심 (보색 제거), engine.py 수정 불필요
from functools import lru_cache
from typing import List, Dict, Any
from HEXnRGB import to_hex, hex_to_rgb
import requests
from typing import Optional



NEUTRAL_SEASON_ACCENTS = {
    "봄": ["#FFF3B0", "#FFD1DC", "#B0C4DE"],    # 파스텔 옐로, 핑크, 라이트 스틸 블루
    "여름": ["#FFFFFF", "#87CEEB", "#98FF98"],  # 화이트, 스카이블루, 민트
    "가을": ["#F4A460", "#8D6E63", "#D4AF37"],  # 샌디브라운, 웜브라운, 머스터드/골드
    "겨울": ["#8B0000", "#01796F", "#2C3E50"],  # 딥레드, 파인그린, 네이비
}


###유틸
def rgb_dist(a, b):
    return sum((x-y)*(x-y) for x,y in zip(a,b))**0.5

def unique_palette(hex_list, tol=18):
    uniq = []
    for hx in hex_list:
        ra = hex_to_rgb(hx)
        if all(rgb_dist(ra, hex_to_rgb(u)) > tol for u in uniq):
            uniq.append(hx)
    return uniq

def is_grayish(hx, sat_tol=12):
    r,g,b = hex_to_rgb(hx)
    return max(r,g,b) - min(r,g,b) <= sat_tol

##colormind

THE_COLOR_API = "https://www.thecolorapi.com/scheme"
COLORMIND_API = "http://colormind.io/api/"

def get_scheme_from_thecolorapi(hex_seed: str, mode: str = "complement", count: int = 4) -> list[str]:
    """
    hex_seed: '#733DCB' or '733DCB'
    mode: 'complement', 'analogic', 'triad', 'tetrad', 'monochrome', ...
    return: list of HEX strings like ['#733DCB', '#CB943D', ...]
    """
    seed = hex_seed.lstrip("#")
    params = {"hex": seed, "mode": mode, "count": count}
    r = requests.get(THE_COLOR_API, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    return [c["hex"]["value"] for c in data.get("colors", [])]

def get_palette_from_colormind(seed_hex: Optional[str] = None) -> list[str]:
    """
    If seed_hex provided, lock first slot and ask Colormind for the rest.
    return: 5 HEX colors
    """
    payload = {"model": "default"}
    if seed_hex:
        s = seed_hex.lstrip("#")
        rgb = [int(s[i:i+2], 16) for i in (0,2,4)]
        payload["input"] = [rgb, "N", "N", "N", "N"]

    r = requests.post(COLORMIND_API, json=payload, timeout=10)
    r.raise_for_status()
    data = r.json()
    colors = data.get("result", [])
    return [f"#{r:02X}{g:02X}{b:02X}" for r, g, b in colors]

## 매핑

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
    try:
        res = get_scheme_from_thecolorapi(seed, mode="analogic", count=5)
        return tuple(res or [])
    except Exception:
        return tuple()

def expand_colors_from_external(seasonal_colors: List[str], season: str, use_colormind: bool = True) -> Dict[str, Any]:
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

    # Colormind 기반 팔레트
    cm = []
    if use_colormind:
        cm = list(_cached_colormind(seed))

    # Colormind 실패/빈 결과 시 보조: 무채색이면 계절 액센트, 아니면 analogic API
    if not cm:
        if is_grayish(seed):
            cm = NEUTRAL_SEASON_ACCENTS.get(season, [])
        else:
            cm = list(_cached_analogous_from_api(seed))

    # seed 자체 제외 + 근접 중복 제거
    cm = [h for h in cm if h and h.upper() not in seeds_upper]
    cm_u = _safe_unique(cm, tol=18)
    # 원본 소스, 관련색, 보색제거
    out["colormind"] = cm_u                         # 레퍼런스를 위해 원본 소스도 전달
    out["analogous"] = cm_u[:3] or cm_u             # 엔진 fallback 방지용으로 analogous 채움
    out["complement"] = []                          # 보색 완전 제거

    return out
