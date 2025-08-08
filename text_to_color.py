import os
import json
import re
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))  # 여기에 실제 키 입력

with open("coloremotion.json", encoding="utf-8") as f:
    db = json.load(f)

emb_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
chroma = chromadb.Client()
try:
    col = chroma.get_collection("color_psych")
except:
    col = chroma.create_collection("color_psych")

try:
    col.delete()
    col = chroma.create_collection("color_psych")
except:
    pass

docs = []
for color, meta in db.items():
    text = f"{color} 색은 {', '.join(meta['emotion'])} 감정을 주며 " \
           f"{', '.join(meta['associations'])} 느낌이 있다."
    docs.append({"id": color, "text": text})

for d in docs:
    emb = emb_model.encode(d["text"]).tolist()
    col.add(ids=[d["id"]], embeddings=[emb], documents=[d["text"]])

def extract_explicit_colors(text: str):
    lower = text.lower()
    return [c for c in db.keys() if c in lower]

def recommend_colors_gemini(user_text: str, top_k: int = 3) -> str:
    explicit = extract_explicit_colors(user_text)
    if explicit:
        return ", ".join(explicit[:2])
    q_emb = emb_model.encode(user_text).tolist()
    hits = col.query(query_embeddings=[q_emb], n_results=top_k)
    context = "\n".join(hits["documents"][0])

    system_msg = (
        "너는 색상 심리학 기반 꽃다발 색상 추천 전문가야. "
        "아래 색상 심리 정보를 참고하고, 사용자 요청에 어울리는 색상 두 가지만 "
        "한국어 단답형으로 추천해줘. 쉼표로 구분하고 이유는 쓰지 마."
    )
    prompt = f"색상 심리 정보:\n{context}\n\n사용자 요청: \"{user_text}\"\n출력:"

    model = genai.GenerativeModel("gemini-1.5-pro") 
    response = model.generate_content(
        [system_msg, prompt],
        generation_config=genai.types.GenerationConfig(
            temperature=0.4,
            max_output_tokens=32
        )
    )

    return re.sub(r"[.\n]+", "", response.text).strip()

if __name__ == "__main__":
    samples = [
        "나 지금 멘탈 나가서 꽃이 받고 싶어",
        "친구 곧 졸업이라 축하해주고 싶어",
        "고려대 졸업식에 어울리는 꽃다발"
    ]
    for text in samples:
        print(f"입력: {text}")
        print(f"추천: {recommend_colors_gemini(text)}\n")
