# palette_fetchers.py
import requests
from typing import Optional

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
