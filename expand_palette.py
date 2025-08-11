from name2hex import to_hex
from palette_fetchers import get_scheme_from_thecolorapi, get_palette_from_colormind
from color_utils import unique_palette, is_grayish

NEUTRAL_SEASON_ACCENTS = {
    "봄": ["#FFF3B0", "#FFD1DC", "#B0C4DE"],    # 파스텔 옐로, 핑크, 라이트 스틸 블루
    "여름": ["#FFFFFF", "#87CEEB", "#98FF98"],  # 화이트, 스카이블루, 민트
    "가을": ["#F4A460", "#8D6E63", "#D4AF37"],  # 샌디브라운, 웜브라운, 머스터드/골드
    "겨울": ["#8B0000", "#01796F", "#2C3E50"],  # 딥레드, 파인그린, 네이비
}

def expand_colors_from_external(seasonal_colors: list[str], season: str, use_colormind: bool = False) -> dict:
    base_hex = [to_hex(c) for c in seasonal_colors if c]
    out = {"base_hex": base_hex, "complement": [], "analogous": [], "colormind": []}
    if not base_hex:
        return out

    seeds = base_hex[:2]  
    comps, anags = [], []

    for seed in seeds:
        if is_grayish(seed):
            comps += NEUTRAL_SEASON_ACCENTS.get(season, [])
            anags += NEUTRAL_SEASON_ACCENTS.get(season, [])
        else:
            comps += get_scheme_from_thecolorapi(seed, mode="complement", count=4)
            anags += get_scheme_from_thecolorapi(seed, mode="analogic", count=5)

    all_comp = [h for h in comps if h.upper() not in {s.upper() for s in seeds}]
    all_anag = [h for h in anags if h.upper() not in {s.upper() for s in seeds}]
    out["complement"] = unique_palette(all_comp, tol=18)[:3]
    out["analogous"]  = unique_palette(all_anag, tol=18)[:3]

    if use_colormind:
        out["colormind"] = unique_palette(get_palette_from_colormind(seeds[0]), tol=18)
    return out
