# engine.py
import os, json, re, io
import requests
import unicodedata
from contextlib import redirect_stdout
from datetime import datetime, timezone, timedelta, time, date as dt_date
from typing import Optional, List, Dict, Any, Tuple
from dotenv import load_dotenv
from HEXnRGB import ensure_hex_list, hex_to_rgb
from expand_palette import expand_colors_from_external

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

LC_LLM = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.2,
    max_output_tokens=4,
    google_api_key=GOOGLE_API_KEY,
)

# llm 감정 추출 프롬프트
EMOTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "너는 이벤트/기념일 설명을 읽고 대표 감정을 한국어 단어 1개로만 답한다."
     "가능한 답: 기쁨, 사랑, 감사, 슬픔, 설렘. 다른 말/문장/기호/마침표 금지."),
    ("human", "기념일: {event_text}\n출력:")
])
_str_parser = StrOutputParser()

# 출력된 감정상태 정제
def _clean_emotion(text: str) -> Optional[str]:
    if not text:
        return None
    ans = re.sub(r"[^\w가-힣]", "", text).strip()
    return ans if ans in {"기쁨","사랑","감사","슬픔","설렘"} else None

_clean_emotion_runnable = RunnableLambda(_clean_emotion)
emotion_chain = EMOTION_PROMPT | LC_LLM | _str_parser | _clean_emotion_runnable

# 컬러 DB 직접 제작한 json
with open("coloremotion.json", encoding="utf-8") as f:
    COLOR_DB: Dict[str, Dict[str, Any]] = json.load(f)

# 이벤트-감정 룰
EVENT_EMOTION_MAP: Dict[str, str] = {
    "졸업": "기쁨","생일": "사랑","결혼": "사랑","기념일": "사랑","발렌타인": "사랑",
    "어버이날": "감사","스승의날": "감사","추석": "감사","설날": "기쁨","크리스마스": "기쁨",
    "장례": "슬픔","제사": "슬픔","추모": "슬픔",
}

# assoc 토큰 파싱 만약 앞에 카테고리가 없으면 plain으로 카테고리를 임의로 만들어준다. 처리하기 편하게 ㅎㅎ
def _split_assoc_token(tok: str) -> Tuple[str, str]:
    if ":" in tok:
        cat, val = tok.split(":", 1)
        return cat.strip(), val.strip()
    return "plain", tok.strip()

def _build_assoc_index(db: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, set], set]:
    by_cat: Dict[str, set] = {}
    all_vals: set = set()
    for _, meta in db.items():
        for tok in meta.get("associations", []) or []:
            cat, val = _split_assoc_token(tok)
            if not val:
                continue
            by_cat.setdefault(cat, set()).add(val)
            all_vals.add(val)
    return by_cat, all_vals

ASSOC_BY_CAT, ASSOC_ALL_VALS = _build_assoc_index(COLOR_DB)

# 전처리
def _normalize_ko(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "")

SYNONYM_MAP: Dict[str, str] = {
    # 상황
    "결혼": "결혼식", "웨딩": "결혼식",
    "추모식": "장례", "부고": "장례",
    "프레젠테이션": "프레젠테이션", "PT": "프레젠테이션",
    "네트워킹": "네트워킹", "오프닝": "오프닝",
    "세일": "세일", "행사": "이벤트",
    # 무드
    "고요": "차분", "평온함": "평온", "신뢰감": "신뢰",
    "럭셔리": "럭셔리", "미니멀리즘": "미니멀",
    # 주의
    "눈부심": "눈부심", "공격적": "공격적", "무거운": "무거움", "유치한": "유치함",
}

def _apply_synonyms(text: str) -> str:
    t = text
    for k, v in SYNONYM_MAP.items():
        t = t.replace(k, v)
    return t

# 감정 브릿지
EMOTION_BRIDGE: Dict[str, set] = {
    "감사": {"안도"},
    "설렘": {"기쁨", "사랑"},
    "슬픔": {"슬픔"},
    "사랑": {"사랑"},
    "기쁨": {"기쁨"},
}
def _emotion_match(meta_emotions: List[str], emo: str) -> bool:
    if emo in meta_emotions:
        return True
    return any(e in meta_emotions for e in EMOTION_BRIDGE.get(emo, set()))

# 이벤트 텍스트 자체와 히트하는지, 이벤트 date와 계절과 얼마나 히트하는지 판단한다.
def _assoc_hits_from_text(event_text: str, event_date: datetime) -> Dict[str, set]:
    t = _apply_synonyms(_normalize_ko(event_text))
    hits: Dict[str, set] = {cat: set() for cat in set(ASSOC_BY_CAT.keys()) | {"plain"}}

    for cat, vals in ASSOC_BY_CAT.items():
        for v in vals:
            if v and v in t:
                hits[cat].add(v)

    for v in (ASSOC_BY_CAT.get("plain") or []):
        if v and v in t:
            hits["plain"].add(v)

    season = get_season(event_date)
    if "계절" in ASSOC_BY_CAT and any(season == v for v in ASSOC_BY_CAT["계절"]):
        hits.setdefault("계절", set()).add(season)

    return hits

# 가중치 그냥 내가 설정함.
CATEGORY_WEIGHTS: Dict[str, float] = {
    "상황": 1.0, "무드": 0.5, "계절": 0.8, "온도": 0.7,
    "톤": 0.3, "팔레트": 0.2, "대비": 0.1, "plain": 0.4,
}
W_PERCENT: float = 0.35 #대중성 가중치
W_EMOTION: float = 0.40 #감정에 가중치
W_ASSOC: float = 0.35 #관련 연관성 텍스트 가중치
W_CAUTION: float = 0.20

# 주의 트리거 목록이고, json 안에 있는 주의라는 카테고리에 대한 벨류 예시를 보여준다. 적절히 튜닝이 필요하다고 생각된다. 사용방향성이 다양한데 제약되어있는 느낌임
CAUTION_TRIGGERS = {"눈부심", "공격적", "무거움", "유치함", "과장감", "장시간", "격식↓"}

def _caution_hits(event_text: str, color_meta: Dict[str, Any]) -> int:
    t = _apply_synonyms(_normalize_ko(event_text))
    caution_vals = set()
    for tok in color_meta.get("associations", []) or []:
        cat, val = _split_assoc_token(tok)
        if cat == "주의":
            caution_vals.add(val)
    return sum(1 for v in caution_vals if v in t or v in CAUTION_TRIGGERS)

# 어쨌든 색상별 association 스코어링을 통해서, 긍정점수 - 부정 점수를 하여 핵심 색상을 추출해주는 함수이다. 
def recommend_colors_scored_v2(event_text: str, emotion: str, event_date: datetime, top_k: int = 2) -> List[str]:
    hits = _assoc_hits_from_text(event_text, event_date)
    sum_w = sum(CATEGORY_WEIGHTS.values()) or 1.0

    scored: List[Tuple[float, str]] = []
    for color, meta in COLOR_DB.items():
        pct = float(meta.get("percentage", 0) or 0) / 100.0
        emo_bonus = 1.0 if _emotion_match(list(meta.get("emotion", [])), emotion) else 0.0
        assoc_raw = 0.0
        color_by_cat: Dict[str, set] = {}
        for tok in meta.get("associations", []) or []:
            cat, val = _split_assoc_token(tok)
            color_by_cat.setdefault(cat, set()).add(val)

        for cat, wcat in CATEGORY_WEIGHTS.items():
            if cat in color_by_cat:
                overlap = len(color_by_cat[cat] & hits.get(cat, set()))
                if overlap > 0:
                    assoc_raw += wcat

        assoc_norm = min(1.0, assoc_raw / sum_w)
        cautions = _caution_hits(event_text, meta)
        caution_norm = min(1.0, cautions / 3.0)

        score = (W_PERCENT * pct) + (W_EMOTION * emo_bonus) + (W_ASSOC * assoc_norm) - (W_CAUTION * caution_norm)
        scored.append((score, color))

    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        top = sorted(COLOR_DB.items(), key=lambda kv: float(kv[1].get("percentage", 0) or 0), reverse=True)
        return [c for c, _ in top[:top_k]]
    return [c for _, c in scored[:top_k]]

# 감정 추출
def extract_emotion_rule(event_text: str) -> Optional[str]:
    for kw, emo in EVENT_EMOTION_MAP.items():
        if kw in event_text:
            return emo
    return None

def extract_emotion_langchain(event_text: str) -> Optional[str]:
    try:
        return emotion_chain.invoke({"event_text": event_text})
    except Exception:
        return None

def extract_emotion(event_text: str) -> str:
    return extract_emotion_rule(event_text) or extract_emotion_langchain(event_text) or "기쁨"

# 계절/표준화
SEASON_TONE_MAP: Dict[str, Dict[str, str]] = {
    "봄": {"yellow": "pastel yellow", "orange": "pastel orange", "red": "pastel red", "pink": "pastel pink"},
    "여름": {"blue": "sky blue", "white": "pure white", "green": "mint green"},
    "가을": {"orange": "burnt orange", "brown": "warm brown", "yellow": "mustard yellow"},
    "겨울": {"red": "deep red", "green": "pine green", "black": "charcoal black", "white": "snow white"},
}
CANONICAL_COLOR_MAP = {
    "pastel yellow": "yellow","pastel orange": "orange","pastel red": "red","pastel pink": "pink",
    "sky blue": "blue","pure white": "white","mint green": "green","burnt orange": "orange",
    "warm brown": "brown","mustard yellow": "yellow","deep red": "red","pine green": "green",
    "charcoal black": "black","snow white": "white",
}

def get_season(dt: datetime) -> str:
    m = dt.month
    if 3 <= m <= 5:  return "봄"
    if 6 <= m <= 8:  return "여름"
    if 9 <= m <= 11: return "가을"
    return "겨울"

def adjust_colors_for_season(colors: List[str], event_date: datetime) -> List[str]:
    season = get_season(event_date)
    tone_map = SEASON_TONE_MAP.get(season, {})
    return [tone_map.get(c, c) for c in colors]

def canonicalize_colors(colors: List[str]) -> List[str]:
    out = []
    for c in colors:
        c2 = CANONICAL_COLOR_MAP.get(c.strip().lower(), c.strip().lower())
        out.append(c2)
    return out

# 메인 추천 (보색 제거 / Colormind 중심)
def recommend_colors_for_event(event_text: str, event_date: datetime, top_k_base: int = 2, expand: bool = True) -> Dict[str, Any]:
    emotion = extract_emotion(event_text)

    # base color 스코어링
    base_colors = recommend_colors_scored_v2(event_text, emotion, event_date, top_k=top_k_base)

    # 계절 톤 적용 + 표준화
    seasonal = adjust_colors_for_season(base_colors, event_date)
    seasonal_norm = canonicalize_colors(seasonal)
    season = get_season(event_date)

    # 외부 확장(Colormind 중심) — expand_palette.py가 보색 없이 analogous만 채움
    try:
        seed_bundle = expand_colors_from_external(seasonal_norm, season, use_colormind=True)
    except Exception:
        seed_bundle = {"base_hex": ensure_hex_list(seasonal_norm),
                       "complement": [], "analogous": [], "colormind": []}

    base_hex = ensure_hex_list(seed_bundle.get("base_hex") or seasonal_norm)
    analog = seed_bundle.get("analogous") or []

    # extra는 아날로그(=Colormind 결과)만 사용
    extra = analog[:3] if (expand and analog) else []
    extra_hex = ensure_hex_list(extra or [])

    base_rgb  = [hex_to_rgb(h) for h in base_hex]
    extra_rgb = [hex_to_rgb(h) for h in extra_hex]
    return {
        "emotion": emotion,
        "base_colors": base_colors,
        "seasonal_colors": seasonal,
        "extra_colors": extra_hex,
        "rgb": {"base": base_rgb, "extra": extra_rgb}
    }

# 출력 포맷
def format_rgb_compact(rgb_list: List[Tuple[int, int, int]]) -> str:
    return "rgb(" + ",".join(f"({r},{g},{b})" for (r, g, b) in rgb_list) + ")"

# ------------------------------
# Google Calendar 연동
# ------------------------------
KST = timezone(timedelta(hours=9))
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REFRESH_TOKEN = os.getenv("GOOGLE_REFRESH_TOKEN")

def _to_rfc3339(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=KST)
    return dt.isoformat()

def _parse_gcal_start(ev: Dict[str, Any]) -> Optional[datetime]:
    start = ev.get("start", {})
    if "dateTime" in start and start["dateTime"]:
        s = start["dateTime"]
        if isinstance(s, str) and s.endswith("Z"):
            s = s.replace("Z", "+00:00")
        try:
            dtv = datetime.fromisoformat(s)
        except Exception:
            return None
        return dtv.astimezone(KST)
    if "date" in start and start["date"]:
        d = dt_date.fromisoformat(start["date"])
        return datetime.combine(d, time(0, 0, 0), KST)
    return None

def _refresh_access_token() -> str:
    if not (GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET and GOOGLE_REFRESH_TOKEN):
        raise RuntimeError("환경변수 GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET / GOOGLE_REFRESH_TOKEN 설정 필요")
    data = {
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "grant_type": "refresh_token",
        "refresh_token": GOOGLE_REFRESH_TOKEN,
    }
    resp = requests.post("https://oauth2.googleapis.com/token", data=data, timeout=30)
    resp.raise_for_status()
    return resp.json()["access_token"]

def fetch_upcoming_events_from_calendar(
    calendar_id: str = "primary",
    max_results: int = 10,
    days_ahead: int = 7,
    time_min: Optional[datetime] = None,
    time_max: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    access_token = _refresh_access_token()
    now_kst = datetime.now(KST)
    tmin = time_min or now_kst
    tmax = time_max or (now_kst + timedelta(days=days_ahead))
    params = {
        "singleEvents": "true",
        "orderBy": "startTime",
        "maxResults": str(max_results),
        "timeMin": _to_rfc3339(tmin),
        "timeMax": _to_rfc3339(tmax),
        "fields": "items(id,summary,description,location,start,end,htmlLink)",
    }
    url = f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events"
    headers = {"Authorization": f"Bearer {access_token}"}
    r = requests.get(url, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("items", [])

def recommend_for_calendar_events(
    calendar_id: str = "primary",
    max_results: int = 10,
    days_ahead: int = 7,
    top_k_base: int = 2,
    expand: bool = True,
) -> List[Dict[str, Any]]:
    events = fetch_upcoming_events_from_calendar(calendar_id, max_results, days_ahead)
    results = []
    for ev in events:
        start_dt = _parse_gcal_start(ev)
        if not start_dt:
            continue
        title = (ev.get("summary") or "").strip()
        desc  = (ev.get("description") or "").strip()
        loc   = (ev.get("location") or "").strip()
        parts = [p for p in [title, desc, loc] if p]
        event_text = " / ".join(parts) if parts else "일정"
        with io.StringIO() as buf, redirect_stdout(buf):
            rec = recommend_colors_for_event(event_text, start_dt, top_k_base=top_k_base, expand=expand)
        results.append({"event_text": event_text, "event_date": start_dt, "rec": rec})
    return results

if __name__ == "__main__":
    try:
        items = recommend_for_calendar_events(
            calendar_id="primary",
            max_results=50,
            days_ahead=7,
            top_k_base=2,
            expand=True,
        )
        for it in items:
            rec = it["rec"]
            rgb_all: List[Tuple[int,int,int]] = [tuple(map(int, t)) for t in (rec["rgb"]["base"] + rec["rgb"]["extra"])]
            print(format_rgb_compact(rgb_all))
    except Exception:
        pass
