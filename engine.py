# bouquet_prompt_engine.py — prompt-first color recommender (no calendar)
# v4: 관계 추론 + 감정 고정(LLM) + DB-가이드레일(prefer/avoid) + **감정 허용 집합을 COLOR_DB에 맞게 동적 확장**

from __future__ import annotations

import os, json, re
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

from HEXnRGB import ensure_hex_list, hex_to_rgb
from expand_palette import expand_colors_from_external
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ----------------------------------------------------------------------------
# CONFIG & CONSTANTS
# ----------------------------------------------------------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

with open("coloremotion.json", encoding="utf-8") as f:
    COLOR_DB: Dict[str, Dict[str, Any]] = json.load(f)

ALLOWED_COLORS = list(COLOR_DB.keys())

# 이름 변형 정규화 (LLM이 토닉/딥/파스텔 등 변형명을 낼 때 안전장치)
CANONICAL = {
    "navy": "blue", "sky blue": "blue",
    "mint": "green", "mint green": "green",
    "deep red": "red", "pastel pink": "pink",
    "pastel yellow": "yellow", "burnt orange": "orange",
    "charcoal": "black",
}

# COLOR_DB에서 감정 라벨을 동적으로 수집 + 기본 5종을 포함해 허용 감정 집합 생성
def _collect_db_emotions() -> List[str]:
    s: set[str] = set()
    for meta in COLOR_DB.values():
        for e in (meta.get("emotion") or []):
            ee = (e or "").strip()
            if ee:
                s.add(ee)
    # 기본 5종은 항상 포함
    s |= {"기쁨", "사랑", "감사", "슬픔", "설렘"}
    return sorted(s)

ALLOWED_EMOTIONS: List[str] = _collect_db_emotions()
EMOTION_SET: set[str] = set(ALLOWED_EMOTIONS)

# 슬픔(장례/추모) 상황에서 피할 색상(고채도/따뜻 계열)
FORBID_FOR_SORROW = {"red", "orange", "yellow", "pink"}
PREFER_FOR_SORROW = ["white", "blue", "green", "black"]

# DB 가이드레일 키워드
FUNERAL_KEYS = {"장례", "추모", "부고"}
FORMAL_RISK = {"과장감", "공격적", "눈부심", "가벼움", "유치함", "장시간", "격식↓", "피로", "거리감", "탁함", "차가움", "구식"}

# ----------------------------------------------------------------------------
# LLMs
# ----------------------------------------------------------------------------
JSON_LLM = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro", temperature=0.4, max_output_tokens=600, google_api_key=GOOGLE_API_KEY
)
MSG_LLM = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro", temperature=0.7, max_output_tokens=700, google_api_key=GOOGLE_API_KEY
)
AUX_LLM = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro", temperature=0.2, max_output_tokens=120, google_api_key=GOOGLE_API_KEY
)
JSON_PARSER = StrOutputParser()
TXT_PARSER = StrOutputParser()

# ----------------------------------------------------------------------------
# PROMPTS — 관계 추론 / 감정 추론 / 팔레트 / 메시지
# ----------------------------------------------------------------------------
RELATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """다음 정보를 보고 '보내는 사람'이 '받는 사람'과 맺은 관계를 한글 한 단어로 추론하고, 호칭(recipient_call)과 말투(politeness)를 제안하라.
출력은 JSON 하나만. 필드는 relationship, recipient_call, politeness.

규칙:
- '받는 사람' 입력값에 '어머니/엄마/아버지/아빠' 같은 호칭이 포함되어 있어도 **실제 관계/세대에 맞는 호칭으로 재선정**한다.
  예) 보내는 사람=할머니, 받는 사람=어머니  → recipient_call='딸'
- 세대 판단을 반영한다: 조부모/부모 세대 → 자녀 호칭(딸/아들), 반대로 자녀 세대 → 부모 호칭(어머니/아버지).
- 성별 단서가 없으면 중립형(친애하는 너/얘야 등)보다 일반적 가족 호칭(딸/아들)을 우선한다.
- politeness는 관계와 상황에 자연스럽게 맞추되, 한국어 표준 어법을 따른다.

설명 금지, JSON만 출력."""
    ),
    (
        "human",
        """보내는 사람: {actor}
받는 사람: {recipient}
초기 관계 힌트(있다면): {relationship_hint}
수신자 성별(있다면): {recipient_gender}
JSON만 출력:"""
    ),
])


EMOTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "입력을 읽고 대표 감정을 한국어 단어 하나로만 선택하라. 허용: {allowed_emotions}. 다른 단어/기호/문장 금지."),
    (
        "human",
        "언제/상황: {when_text}\n관계요약: {relationship}\n히스토리: {history}\n출력:"),
])

SCHEMA = r'''{
  "emotion": "<허용 감정 중 1개>",
  "base_colors": ["..."],
  "accent_colors": ["..."],
  "avoid": ["피해야 할 색/특성"],
  "rationale": "선택 이유 요약"
}'''

PALETTE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "너는 플로리스트 겸 색채 심리 컨설턴트다.\n"
        "감정은 이미 확정되었고 반드시 {forced_emotion} 로 출력해야 한다.\n"
        "'허용 색상 목록' 범위에서 기본색 2~3, 악센트 0~2를 고르고 JSON 하나만 출력하라. 설명 문장 금지.\n"
        "규칙:\n"
        "- allowed_colors 안에서만 고른다 (소문자 단수형).\n"
        "- 장례/추모/부고(=슬픔) 맥락: 비비드/고채도 금지, white/green/blue 중심, red/orange/yellow/pink 회피.\n"
        "- 사랑/연애/기념일: red/pink/white 중심, 과하지 않게 green으로 안정.\n"
        "- 비즈니스/프레젠테이션/수험: blue/white 중심, 과장감 주의.\n"
        "- 고연령 수신자(60+ 추정)나 격식 높은 자리: 고채도·과장감 지양, 대비 과하면 주의.\n"
        "- base_colors 2~3, accent_colors 0~2. 총합 3~5 색. 중복 금지.\n"
        "- 다음 색은 반드시 사용 금지: {must_avoid}. base_colors, accent_colors에 포함하지 마라.\n"
        "- 가능하면 다음 색을 우선 고려: {prefer_list}. 규칙을 해치지 않는 범위에서 반영하라.\n"
        "JSON 스키마:\n{schema}\n"),
    (
        "human",
        "허용 색상 목록: {allowed}\n확정 감정: {forced_emotion}\n관계 추론: {relation_json}\n컨텍스트:\n- 언제: {when_text}\n- 히스토리: {history}\n- 성별(수신자): {recipient_gender}\nJSON만 출력:",
    ),
])

MESSAGE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """너는 사람의 마음을 움직이는 감성적인 편지글 전문가이자, 따뜻하고 섬세한 감정을 글로 풀어내는 한국어 카피라이터이다. 받는 사람에게 깊은 감동과 위로를 전할 수 있도록, 핵심 메시지를 3~5문장 안에 담아낸다. 상대방의 마음을 어루만져 주는 위로와 공감의 톤으로, 진심이 느껴지는 따뜻한 문장을 쓴다. 이때, 비유와 은유를 적절히 활용하여 서정적인 분위기를 연출해야 한다.

기본 규칙
1) 이모지/해시태그/과도한 특수문자 금지(마침표와 쉼표만 사용).
2) 색 이름/HEX 언급 금지(정서는 은근히).
3) {politeness} 톤 준수. 문장은 수신자 호칭 '{recipient_call}'로 시작.
4) 수신자와 친근함을 나타내야한다면 성을 뺀 '누구누구야'로 시작.
5) 메시지는 **항상 보내는 사람(actor)의 1인칭 시점**으로, 수신자에게 직접 말을 건넨다.
   - actor_is_deceased 여부와 무관하게 ‘내가 너에게 말한다’ 관점을 유지.
   - 노골적 영적 표현(하늘에서 지켜보고, 보고 있겠죠, 평안하세요 등) 금지.
6) 맥락 일치: 축하/기쁨에 추모 어휘 금지, 슬픔 맥락에 과도한 축하 어휘 금지.
7) 관계/히스토리는 1문장 이내로만 암시한다.
시적 장치 (감성 강화)
8) **감각 이미지 1~2개**를 쓴다(냄새/소리/온기/빛/감촉). 예: 미역국의 김, 새벽 부엌의 물끓는 소리, 손등의 온기.
9) **구체 장면 1개**를 짧게 불러온다(시장 골목, 창턱의 햇살, 젖은 앞치마 등).
10) 물음표/느낌표 금지. 단정적 어조로 잔잔한 호흡 유지.
11) 종결은 **약속/응원/축원** 중 하나로 맺고, ‘사랑한다’는 최대 1회만 자연스럽게 배치한다(보통 마지막 또는 끝에서 두 번째).

출력 형식
- 한 문단, 줄바꿈 없이 문장만 출력한다."""
    ),
    (
        "human",
        """상황:
- 언제: {when_text}
- 보내는 사람(표현 관점): {actor_effective}
- 받는 사람: {recipient} ({recipient_gender})
- 관계(추론): {inferred_relationship}
- 히스토리: {history}
- 감정: {emotion}
- actor_is_deceased: {actor_is_deceased}
- actor_original: {actor_original}

문장만 출력:"""
    ),
])



# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------

def _invoke(prompt, model, parser, payload) -> str:
    return (prompt | model | parser).invoke(payload)


def _try_parse_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    m = re.search(r"\{[\s\S]*\}\s*$", text)
    raw = m.group(0) if m else text
    for repl in (raw, raw.replace("'", '"')):
        try:
            return json.loads(repl)
        except Exception:
            continue
    return {}


def _canonicalize_names(names: List[str]) -> List[str]:
    out = []
    for n in names or []:
        k = CANONICAL.get((n or "").strip().lower(), (n or "").strip().lower())
        if k in ALLOWED_COLORS and k not in out:
            out.append(k)
    return out


def _to_hex_list(color_names: List[str]) -> List[str]:
    try:
        hx = ensure_hex_list(color_names)
        return [h for h in hx if isinstance(h, str) and h.startswith("#")]
    except Exception:
        return []


def _select_rgb_values(rgb_base: List[Tuple[int,int,int]],
                       rgb_accent: List[Tuple[int,int,int]],
                       rgb_extra: List[Tuple[int,int,int]],
                       min_n: int = 3,
                       max_n: int = 4) -> List[Tuple[int,int,int]]:
    seen = set(); ordered: List[Tuple[int,int,int]] = []
    for lst in (rgb_base, rgb_accent, rgb_extra):
        for t in lst:
            if isinstance(t, (list, tuple)) and len(t) == 3:
                key = tuple(int(x) for x in t)
                if key not in seen:
                    seen.add(key); ordered.append(key)
    while len(ordered) < min_n and ordered:
        ordered.extend(ordered[: min_n - len(ordered)])
    return ordered[:max_n] if ordered else []


def _format_rgb_compact(rgb_list: List[Tuple[int,int,int]]) -> str:
    return "rgb(" + ",".join(f"({r},{g},{b})" for (r, g, b) in rgb_list) + ")"

# --- DB 가이드레일 도출 ------------------------------------------------------

def _assoc_by_cat(assoc_list: List[str]) -> Dict[str, set]:
    by: Dict[str, set] = {}
    for tok in assoc_list or []:
        if ":" in tok:
            cat, val = tok.split(":", 1)
            by.setdefault(cat.strip(), set()).add(val.strip())
        else:
            by.setdefault("plain", set()).add(tok.strip())
    return by


def _derive_prefer_avoid_from_db(emotion: str, when_text: str) -> Tuple[List[str], List[str]]:
    t = (when_text or "")
    is_funeral = any(k in t for k in FUNERAL_KEYS)

    prefer, avoid = set(), set()
    for color, meta in COLOR_DB.items():
        emos = set(meta.get("emotion") or [])
        assoc = _assoc_by_cat(meta.get("associations") or [])

        # caution 수집: '주의:값' + '(주의)' 표기 처리
        cautions = set()
        for tok in (meta.get("associations") or []):
            if ":" in tok:
                cat, val = tok.split(":", 1)
                v = val.strip()
                if cat.strip() == "주의" and v:
                    cautions.add(v)
                if "(주의)" in v:
                    base = v.replace("(주의)", "").strip()
                    if base:
                        cautions.add(base)
        cautions |= (assoc.get("주의") or set())

        # prefer: 감정 매칭 or 장례 상황에 해당 색이 상황:장례에 매핑되어 있으면
        funeral_hit = bool((assoc.get("상황") or set()) & FUNERAL_KEYS)
        if (emotion in emos) or (is_funeral and funeral_hit):
            prefer.add(color)

        # avoid: 주의어 교집합 + 장례 warm vivid
        if (cautions & FORMAL_RISK):
            avoid.add(color)
        if is_funeral and color in FORBID_FOR_SORROW:
            avoid.add(color)

    prefer_list = [c for c in prefer if c in ALLOWED_COLORS]
    avoid_list  = [c for c in avoid  if c in ALLOWED_COLORS]
    return prefer_list, avoid_list

# ----------------------------------------------------------------------------
# INFERENCE — 관계 / 감정
# ----------------------------------------------------------------------------

def infer_relation(
    actor: str,
    recipient: str,
    relationship_hint: str = "",
    recipient_gender: str = "",
) -> Dict[str, str]:
    payload = {
        "actor": actor.strip() or "",
        "recipient": recipient.strip() or "",
        "relationship_hint": relationship_hint.strip() or "없음",
        "recipient_gender": (recipient_gender or "").strip(),
    }
    txt = _invoke(RELATION_PROMPT, AUX_LLM, TXT_PARSER, payload)
    info = _try_parse_json(txt)

    rel  = (info.get("relationship") or relationship_hint or "지인").strip()
    call = (info.get("recipient_call") or "").strip()
    politeness = (info.get("politeness") or "존댓말").strip()

    # 🔧 (1) 과거 '어머니/엄마' 강제 보호 로직 제거 (오동작 원인)
    #     -> 더 이상 recipient 문자열만으로 '어머니'를 고정하지 않음

    # 🔧 (2) 세대 보정: 보내는 사람이 '상위 세대'인 경우, 잘못 추정된 부모 호칭을 자녀 호칭으로 교정
    elder_markers = ("할머니","할아버지","외할머니","외할아버지","어머니","아버지","엄마","아빠","부모","고모","이모","삼촌","큰엄마","큰아버지")
    parent_like_calls = {"어머니","어머님","엄마","아버지","아빠","부모님"}

    if any(k in actor for k in elder_markers):
        if (not call) or (call in parent_like_calls):
            g = (recipient_gender or "").strip()
            if g.startswith("여"):
                call = "딸"
            elif g.startswith("남"):
                call = "아들"
            else:
                call = "딸"  # 젠더 불명시 기본값

        # 관계도 자녀 축으로 정규화
        if rel in {"지인","동료","친구"}:
            rel = "부모"  # 상위세대 → '부모' 관계로 보정
    if not call:
        call = recipient.strip() or "친애하는 너"

    return {"relationship": rel, "recipient_call": call, "politeness": politeness}

def _rule_based_emotion(when_text: str) -> Optional[str]:
    t = when_text or ""
    if any(k in t for k in ["장례", "부고", "추모", "영결", "발인"]):
        return "슬픔"
    if any(k in t for k in ["결혼", "연애", "프로포즈", "기념일", "발렌타인"]):
        return "사랑"
    if any(k in t for k in ["생일", "축하", "졸업", "승진", "합격"]):
        return "기쁨"
    if "감사" in t:
        return "감사"
    return None


def infer_emotion(when_text: str, relationship: str, history: str) -> str:
    # 규칙 우선
    rb = _rule_based_emotion(when_text)
    if rb in EMOTION_SET:
        return rb
    # LLM에 허용 감정 집합을 명시
    txt = _invoke(EMOTION_PROMPT, AUX_LLM, TXT_PARSER, {
        "allowed_emotions": ", ".join(ALLOWED_EMOTIONS),
        "when_text": when_text,
        "relationship": relationship,
        "history": history or "없음",
    })
    emo = re.sub(r"[^가-힣A-Za-z]", "", txt or "").strip()
    return emo if emo in EMOTION_SET else "기쁨"

# ----------------------------------------------------------------------------
# CORE API
# ----------------------------------------------------------------------------

def recommend_bouquet_colors(*,
    when_text: str,
    actor: str,
    recipient: str,
    relationship: str,
    history: str,
    recipient_gender: str,
    expand_analogous: bool = True,
    rgb_target: int = 4,
) -> Dict[str, Any]:
    """관계 추론 → 감정 확정(허용 감정=COLOR_DB 기반) → DB-가이드레일 → 팔레트(JSON) → HEX/RGB → 메시지."""

    # 1) 관계 추론 (배경: actor가 고인일 수 있음)
    def _is_deceased(txt: str) -> bool:
        t = (txt or "")
        return any(k in t for k in ["돌아가신", "고인", "故", "하늘", "별세", "영면", "타계"])

    actor_is_deceased = _is_deceased(actor)
    actor_for_relation = actor
    rel_info = infer_relation(actor_for_relation, recipient, relationship)

    # 2) 감정 확정(룰 기반 우선, 없으면 LLM)
    emotion = infer_emotion(when_text, rel_info["relationship"], history)

    # 3) DB 가이드레일 산출
    prefer, avoid = _derive_prefer_avoid_from_db(emotion, when_text)

    # 4) 팔레트 JSON (감정 강제 + prefer/avoid 주입)
    relation_json = json.dumps(rel_info, ensure_ascii=False)
    palette_payload = {
        "allowed": ", ".join(ALLOWED_COLORS),
        "forced_emotion": emotion,
        "relation_json": relation_json,
        "when_text": when_text.strip(),
        "history": history.strip(),
        "recipient_gender": recipient_gender.strip(),
        "schema": SCHEMA,
        "prefer_list": ", ".join(prefer) if prefer else "없음",
        "must_avoid": ", ".join(avoid) if avoid else "없음",
    }
    out = _try_parse_json(_invoke(PALETTE_PROMPT, JSON_LLM, JSON_PARSER, palette_payload))

    # 5) 후처리 — 감정 기반 강제 규칙 적용(슬픔 시 금지색 제거 + 기본색 보충)
    base_names = _canonicalize_names(out.get("base_colors", []))
    acc_names  = _canonicalize_names(out.get("accent_colors", []))

    if emotion == "슬픔":
        base_names = [c for c in base_names if c not in FORBID_FOR_SORROW]
        acc_names  = [c for c in acc_names  if c not in FORBID_FOR_SORROW]
        if len(base_names) < 2:
            for c in PREFER_FOR_SORROW:
                if c in ALLOWED_COLORS and c not in base_names:
                    base_names.append(c)
                if len(base_names) >= 2:
                    break

    # Fallback: 최소 2색 확보
    names = base_names + [c for c in acc_names if c not in base_names]
    if len(names) < 2:
        ranked = sorted(ALLOWED_COLORS, key=lambda c: COLOR_DB[c].get("percentage", 0), reverse=True)
        for c in ranked:
            if c not in names:
                names.append(c)
            if len(names) >= 2:
                break
        base_names = names[:2]
        acc_names = names[2:3]

    # 6) HEX 변환 + 유사색 확장(analogous만)
    base_hex = _to_hex_list(base_names)
    acc_hex  = _to_hex_list(acc_names)

    extra_hex: List[str] = []
    if expand_analogous:
        try:
            seed = expand_colors_from_external(base_names, None, use_colormind=True) or {}
            extra_hex = [h for h in (seed.get("analogous") or []) if isinstance(h, str)][:3]
        except Exception:
            extra_hex = []

    # 7) RGB 변환 + 3~4개 발췌
    rgb_base   = [hex_to_rgb(h) for h in base_hex]
    rgb_accent = [hex_to_rgb(h) for h in acc_hex]
    rgb_extra  = [hex_to_rgb(h) for h in extra_hex]
    rgb_selected = _select_rgb_values(rgb_base, rgb_accent, rgb_extra, min_n=3, max_n=max(3, min(4, rgb_target)))

    # 8) 메시지 생성(관계/호칭/말투 + 고인 시점 방지 반영)
    msg_text = _invoke(MESSAGE_PROMPT, MSG_LLM, TXT_PARSER, {
        "when_text": when_text,
        "actor_effective": actor,          # <-- actor 그대로
        "actor_original": actor,
        "actor_is_deceased": "예" if actor_is_deceased else "아니오",
        "recipient": recipient,
        "recipient_gender": recipient_gender,
        "inferred_relationship": rel_info["relationship"],
        "recipient_call": rel_info["recipient_call"],   # <-- 여기서 '딸' 전달됨
        "politeness": rel_info["politeness"],
        "history": history or "",
        "emotion": emotion,
    })
    message = re.sub(r"[ \t]+", " ", msg_text or "").strip()

    return {
        "emotion": emotion,
        "allowed_emotions": ALLOWED_EMOTIONS,
        "relation": rel_info,
        "base_colors": base_names,
        "accent_colors": acc_names,
        "hex": {"base": base_hex, "accent": acc_hex, "extra": extra_hex},
        "rgb": {"base": rgb_base, "accent": rgb_accent, "extra": rgb_extra},
        "rgb_selected": rgb_selected,
        "rgb_compact": _format_rgb_compact(rgb_selected) if rgb_selected else "",
        "avoid": out.get("avoid", []),
        "rationale": out.get("rationale", ""),
        "message": message,
        "raw": out,
    }

# ----------------------------------------------------------------------------
# CLI demo
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    demo = recommend_bouquet_colors(
        when_text="엄마 생일",
        actor="돌아가신 할아버지",
        recipient="울 엄마",
        relationship="딸",  # 힌트가 있어도 LLM이 재추론함
        history="엄마는 할아버지랑 오래 같이 살았었어",
        recipient_gender="여자",
        rgb_target=4,
    )
    print(demo.get("allowed_emotions"))
    print(demo.get("rgb_compact"))
    print("메시지:\n" + demo.get("message", ""))
