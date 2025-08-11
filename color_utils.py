# color_utils.py
def hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0,2,4))

def rgb_to_hex(rgb):
    r,g,b = rgb
    return f"#{r:02X}{g:02X}{b:02X}"

def rgb_dist(a, b):
    return sum((x-y)*(x-y) for x,y in zip(a,b))**0.5

def unique_palette(hex_list, tol=18):
    """RGB 거리 기반으로 비슷한 색 제거(기본 tol=18). 순서를 유지."""
    uniq = []
    for hx in hex_list:
        ra = hex_to_rgb(hx)
        if all(rgb_dist(ra, hex_to_rgb(u)) > tol for u in uniq):
            uniq.append(hx)
    return uniq

def is_grayish(hx, sat_tol=12):
    r,g,b = hex_to_rgb(hx)
    return max(r,g,b) - min(r,g,b) <= sat_tol
