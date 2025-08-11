# color_vectors.py
import numpy as np
import colorsys
from name2hex import to_hex 

def ensure_hex_list(items):
    out = []
    for x in items:
        xs = str(x)
        if xs.startswith("#") and len(xs) in (4,7):
            out.append(xs)
        else:
            out.append(to_hex(xs))
    return out

#뽑아온 hex 색상을 RGB 정수 형태로 변환(0~255)
def hex_to_rgb(hexstr: str) -> tuple[int, int, int]:
    h = hexstr.lstrip("#")
    if len(h) == 3:  # 만약 hex가 3자리라면 다음과 같이 2배로 늘려줌
        h = "".join(c*2 for c in h)
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))