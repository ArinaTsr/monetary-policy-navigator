import streamlit as st
import os
import re
import json
from dotenv import load_dotenv
from rapidfuzz import process, fuzz
from pinecone import Pinecone

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Monetary Policy Navigator",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main > div { padding-top: 2rem; }

.mpn-header {
    text-align: center;
    margin-bottom: 2.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid #e5e7eb;
}
.mpn-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    font-weight: 400;
    color: #0f172a;
    margin: 0 0 0.4rem;
    letter-spacing: -0.5px;
}
.mpn-header p {
    font-size: 0.95rem;
    color: #64748b;
    margin: 0;
    font-weight: 300;
}

.chip-container { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 1.5rem; justify-content: center; }
.chip {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 0.82rem;
    color: #475569;
    font-family: 'DM Sans', sans-serif;
}

.route-badge {
    display: inline-block;
    font-size: 0.72rem;
    padding: 2px 10px;
    border-radius: 20px;
    font-weight: 500;
    margin-bottom: 8px;
    letter-spacing: 0.03em;
}
.badge-full_mix                  { background: #EEEDFE; color: #3C3489; }
.badge-temporal_comparative      { background: #FAEEDA; color: #633806; }
.badge-institutional_comparative { background: #E1F5EE; color: #085041; }
.badge-temporal                  { background: #E6F1FB; color: #0C447C; }
.badge-institution               { background: #F1EFE8; color: #444441; }
.badge-factual                   { background: #F1EFE8; color: #444441; }

.answer-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    font-size: 0.95rem;
    line-height: 1.75;
    color: #1e293b;
    margin-bottom: 1rem;
}

.source-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid #f1f5f9;
    font-size: 0.83rem;
}
.source-item:last-child { border-bottom: none; }
.source-author { font-weight: 500; color: #0f172a; }
.source-meta { color: #94a3b8; font-size: 0.78rem; }
.source-score {
    background: #f1f5f9;
    border-radius: 4px;
    padding: 2px 7px;
    font-size: 0.75rem;
    color: #64748b;
    font-weight: 500;
}

.user-msg {
    background: #1e3a5f;
    color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 10px 16px;
    font-size: 0.92rem;
    margin-left: auto;
    max-width: 80%;
    margin-bottom: 1.5rem;
    width: fit-content;
}

.stTextInput input {
    border-radius: 12px !important;
    border: 1.5px solid #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.75rem 1rem !important;
}
.stTextInput input:focus {
    border-color: #1e3a5f !important;
    box-shadow: 0 0 0 3px rgba(30,58,95,0.08) !important;
}

.stButton button {
    background: #1e3a5f !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.5rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    width: 100% !important;
}
.stButton button:hover { background: #1e293b !important; }
.answer-divider { border: none; border-top: 1px solid #f1f5f9; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Load environment ──────────────────────────────────────────────────
load_dotenv()

# ── Load models & connections ─────────────────────────────────────────
@st.cache_resource
def load_resources():
    from llama_index.llms.google_genai import GoogleGenAI
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.core import Settings

    # API keys — from .env locally, from st.secrets on Streamlit Cloud
    try:
        google_key  = st.secrets["GOOGLE_API_KEY"]
        pinecone_key = st.secrets["PINECONE_API_KEY"]
    except:
        google_key  = os.getenv("GOOGLE_API_KEY")
        pinecone_key = os.getenv("PINECONE_API_KEY")

    # LLM
    llm = GoogleGenAI(
        model="gemini-2.5-flash",
        api_key=google_key,
        system_prompt="You are a concise central bank analyst. Keep answers to 3-5 sentences. No preamble, no step headers."
    )

    # Embedding model — same as used for indexing
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    Settings.llm = llm
    Settings.embed_model = embed_model

    # Pinecone
    pc = Pinecone(api_key=pinecone_key)
    index_pinecone = pc.Index("bis-speeches")

    # Author list for fuzzy matching
    with open('authors.json', 'r') as f:
        author_list = json.load(f)

    return llm, embed_model, index_pinecone, author_list

llm, embed_model, index_pinecone, AUTHOR_LIST = load_resources()

# ── Routing constants ─────────────────────────────────────────────────
TOP_K = {
    "temporal": 4, "institution": 4,
    "institutional_comparative": 8, "temporal_comparative": 8,
    "full_mix": 12, "factual": 4,
}

QUERY_TO_METADATA = {
    "ecb": "ECB", "european central bank": "ECB",
    "fed": "Federal Reserve", "federal reserve": "Federal Reserve", "fomc": "Federal Reserve",
    "bank of england": "Bank of England", "boe": "Bank of England",
    "bank of japan": "Bank of Japan", "boj": "Bank of Japan",
    "bundesbank": "Deutsche Bundesbank", "deutsche bundesbank": "Deutsche Bundesbank",
    "bank of canada": "Bank of Canada",
    "reserve bank of australia": "Reserve Bank of Australia", "rba": "Reserve Bank of Australia",
    "reserve bank of new zealand": "Reserve Bank of New Zealand", "rbnz": "Reserve Bank of New Zealand",
    "reserve bank of india": "Reserve Bank of India", "rbi": "Reserve Bank of India",
    "people's bank of china": "People's Bank of China", "pboc": "People's Bank of China",
    "central bank of china": "People's Bank of China", "bank of china": "People's Bank of China",
    "chinese central bank": "People's Bank of China",
    "swiss national bank": "Swiss National Bank", "snb": "Swiss National Bank",
    "riksbank": "Sveriges Riksbank", "sveriges riksbank": "Sveriges Riksbank",
    "norges bank": "Norges Bank",
    "bis": "Bank for International Settlements",
    "bank for international settlements": "Bank for International Settlements",
    "imf": "IMF", "international monetary fund": "IMF",
    "banque de france": "Banque de France", "bank of france": "Banque de France",
    "banca d'italia": "Banca d'Italia", "bank of italy": "Banca d'Italia",
    "banco de españa": "Banco de España", "bank of spain": "Banco de España",
    "south african reserve bank": "South African Reserve Bank", "sarb": "South African Reserve Bank",
    "bank of korea": "Bank of Korea",
    "monetary authority of singapore": "Monetary Authority of Singapore", "mas": "Monetary Authority of Singapore",
}

# ── Helpers ───────────────────────────────────────────────────────────

def extract_years(question):
    return re.findall(r'\b(19|20)\d{2}\b', question)

def extract_institutions(question):
    prompt = f"""Extract any central bank or monetary institution names from this question.
Map them to their official canonical name.
Return ONLY a valid JSON list like ["ECB", "Federal Reserve", "People's Bank of China"].
If none found, return [].
Do not include any other text.

Question: {question}"""
    try:
        response = llm.complete(prompt)
        text = response.text.strip().replace("```json","").replace("```","").strip()
        institutions = json.loads(text)
        result = {}
        for inst in institutions:
            inst_lower = inst.lower()
            matched = False
            for alias, canonical in QUERY_TO_METADATA.items():
                if alias in inst_lower or inst_lower in alias:
                    result[canonical] = True
                    matched = True
                    break
            if not matched:
                result[inst] = True
        return list(result.keys())
    except:
        q = question.lower()
        found = {}
        for alias, canonical in QUERY_TO_METADATA.items():
            if alias in q:
                found[canonical] = True
        return list(found.keys())

def extract_author(question):
    prompt = f"""If this question asks about a specific central bank official or speaker, return their full name or last name only.
If no specific person is mentioned, return null.
Return ONLY the name or null, no other text.

Question: {question}"""
    try:
        response = llm.complete(prompt)
        extracted = response.text.strip()
        if extracted.lower() in ["null", "none", ""]:
            return None
        match, score, _ = process.extractOne(
            extracted, AUTHOR_LIST, scorer=fuzz.partial_ratio
        )
        if score >= 70:
            return match
        return None
    except:
        return None

def route_query(question):
    q = question.lower()
    years        = extract_years(q)
    institutions = extract_institutions(question)
    author       = extract_author(question)
    has_comparison = any(w in q for w in ["compare","differ","difference","versus","vs","contrast"])
    has_temporal   = any(w in q for w in ["over time","evolved","trend","changed","history","before","after","since","crisis"])
    has_two_years  = len(years) >= 2
    has_one_year   = len(years) == 1
    has_two_inst   = len(institutions) >= 2
    has_one_inst   = len(institutions) == 1

    if has_two_inst and has_two_years:
        return {"route":"full_mix","year_1":years[0],"year_2":years[1],
                "institution_1":institutions[0],"institution_2":institutions[1],"author":author}
    if has_two_years:
        return {"route":"temporal_comparative","year_1":years[0],"year_2":years[1],"author":author}
    if has_two_inst or (has_comparison and has_one_inst):
        return {"route":"institutional_comparative",
                "institution_1":institutions[0] if len(institutions)>=1 else "institution 1",
                "institution_2":institutions[1] if len(institutions)>=2 else "institution 2",
                "author":author}
    if has_temporal or has_one_year:
        return {"route":"temporal","pivot_year":years[0] if years else "the reference period","author":author}
    if has_one_inst:
        return {"route":"institution","institution":institutions[0],"author":author}
    return {"route":"factual","author":author}

# ── Retrieval ─────────────────────────────────────────────────────────

def retrieve_unique(query, top_k=4, institutions=None, author=None):
    seen_titles = set()

    if institutions and len(institutions) >= 2:
        per_inst = max(2, top_k // len(institutions))
        unique_nodes = []

        for inst in institutions:
    # Remove other institution names from query to avoid embedding bias
            inst_query = f"{inst} {inst_query}"
            inst_vector = embed_model.get_text_embedding(inst_query)

            results = index_pinecone.query(
                vector=inst_vector,
                top_k=per_inst * 40,
                include_metadata=True
            )

            inst_chunks = []
            for match in results['matches']:
                title = match['metadata'].get('title','')
                if title in seen_titles: continue
                node_inst = match['metadata'].get('institution','').lower()
                if inst.lower() in node_inst or node_inst in inst.lower():
                    if author and author.lower() not in match['metadata'].get('author','').lower():
                        continue
                    seen_titles.add(title)
                    inst_chunks.append(match)
                if len(inst_chunks) == per_inst: break

            unique_nodes.extend(inst_chunks)
        return unique_nodes[:top_k]

    else:
        search_query = f"{author} {query}" if author else query
        search_vector = embed_model.get_text_embedding(search_query)

        results = index_pinecone.query(
            vector=search_vector,
            top_k=top_k * 40,
            include_metadata=True
        )

        unique_nodes = []
        for match in results['matches']:
            title = match['metadata'].get('title','')
            if title in seen_titles: continue
            if author and author.lower() not in match['metadata'].get('author','').lower():
                continue
            seen_titles.add(title)
            unique_nodes.append(match)
            if len(unique_nodes) == top_k: break

        return unique_nodes

# ── Prompts ───────────────────────────────────────────────────────────

PROMPTS = {
    "institution": "You are analyzing central bank communication.\nBased only on the speeches provided, answer the question about {institution}. Be specific — cite speakers and dates inline.\nOutput a concise answer of 3-5 sentences. No headers, no steps, no preamble.\n\nQuestion: {question}\n\nSpeeches:\n{chunks}\n\nAnswer:",
    "temporal": "You are analyzing how central bank communication evolved over time around {pivot_year}.\nInternally consider: what was the position before and after, what changed and why.\nOutput a concise answer of 3-5 sentences. No headers, no steps, no preamble. Name banks and speakers inline.\n\nQuestion: {question}\n\nSpeeches:\n{chunks}\n\nAnswer:",
    "institutional_comparative": "You are analyzing central bank communication across {institution_1} and {institution_2}.\nInternally consider: each institution's position and how they differ.\nOutput a concise answer of 3-5 sentences. No headers, no steps, no preamble. Cite speakers and dates inline.\n\nQuestion: {question}\n\nSpeeches:\n{chunks}\n\nAnswer:",
    "temporal_comparative": "You are analyzing how central bank positions evolved between {year_1} and {year_2}.\nInternally consider: dominant position in each period, what changed, whether banks diverged.\nOutput a concise answer of 3-5 sentences. No headers, no steps, no preamble. Name banks and speakers inline.\n\nQuestion: {question}\n\nSpeeches:\n{chunks}\n\nAnswer:",
    "full_mix": "You are analyzing central bank communication across {institution_1} and {institution_2} in {year_1} and {year_2}.\nInternally consider: each institution's position in each period, what changed, what stayed the same.\nOutput a concise answer of 3-5 sentences. No headers, no steps, no preamble. Cite speakers and dates inline.\n\nQuestion: {question}\n\nSpeeches:\n{chunks}\n\nAnswer:",
    "factual": "You are analyzing central bank speeches.\nAnswer the question based only on the speeches provided.\nOutput a concise answer of 3-5 sentences. No headers, no steps, no preamble. Cite speakers and dates inline.\n\nQuestion: {question}\n\nSpeeches:\n{chunks}\n\nAnswer:",
}

def build_chunks_text(matches):
    return "\n\n---\n\n".join([
        f"[{m['metadata']['author']} | {m['metadata'].get('institution','?')} | {m['metadata']['date'][:10]}]\n{m['metadata']['text']}"
        for m in matches
    ])

def build_prompt(question, route_info, matches):
    chunks = build_chunks_text(matches)
    route  = route_info["route"]
    tmpl   = PROMPTS.get(route, PROMPTS["factual"])
    params = {k:v for k,v in route_info.items() if k not in ["route","author"]}
    return tmpl.format(chunks=chunks, question=question, **params)

def smart_query(question):
    route_info   = route_query(question)
    k            = TOP_K[route_info["route"]]
    author       = route_info.get("author")
    institutions = None
    if route_info["route"] in ["institutional_comparative","full_mix"]:
        institutions = [route_info.get("institution_1",""), route_info.get("institution_2","")]
        institutions = [i for i in institutions if i]
    matches  = retrieve_unique(question, top_k=k, institutions=institutions, author=author)
    prompt   = build_prompt(question, route_info, matches)
    response = llm.complete(prompt)
    return response.text, route_info, matches

# ── Session state ─────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Header ────────────────────────────────────────────────────────────
st.markdown("""
<div class="mpn-header">
    <h1>Monetary Policy Navigator</h1>
    <p>20,000+ central bank speeches from 130 institutions · 1997–2024<br>Compare institutions, track policy evolution, explore what leaders said</p>
</div>
""", unsafe_allow_html=True)

# ── Example chips ─────────────────────────────────────────────────────
st.markdown("""
<div class="chip-container">
    <span class="chip">How did ECB and Fed differ on inflation in 2008 vs 2020?</span>
    <span class="chip">What did Draghi say about quantitative easing?</span>
    <span class="chip">How has forward guidance evolved since 2008?</span>
    <span class="chip">What is the ECB's view on digital currency?</span>
</div>
""", unsafe_allow_html=True)

# ── Chat history ──────────────────────────────────────────────────────
for item in st.session_state.history:
    st.markdown(f'<div class="user-msg">{item["question"]}</div>', unsafe_allow_html=True)

    route          = item["route_info"]["route"]
    author_detected = item["route_info"].get("author")
    badge_label    = route.replace("_"," ")
    if author_detected:
        badge_label += f" · {author_detected}"

    st.markdown(f'<span class="route-badge badge-{route}">{badge_label}</span>', unsafe_allow_html=True)
    st.markdown(f'<div class="answer-box">{item["answer"]}</div>', unsafe_allow_html=True)

    st.markdown("**Sources**")
    for match in item["matches"]:
        author      = match['metadata'].get("author","Unknown")
        institution = match['metadata'].get("institution","?")
        date        = match['metadata'].get("date","")[:10]
        score       = f"{match['score']:.3f}"
        st.markdown(f"""
        <div class="source-item">
            <div>
                <span class="source-author">{author}</span>
                <span class="source-meta"> · {institution} · {date}</span>
            </div>
            <span class="source-score">{score}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="answer-divider">', unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────
with st.form("query_form", clear_on_submit=True):
    question = st.text_input(
        label="question",
        label_visibility="collapsed",
        placeholder="Ask about central bank speeches..."
    )
    submitted = st.form_submit_button("Ask")

if submitted and question.strip():
    with st.spinner("Searching speeches..."):
        answer, route_info, matches = smart_query(question)
    st.session_state.history.append({
        "question":   question,
        "answer":     answer,
        "route_info": route_info,
        "matches":    matches
    })
    st.rerun()
