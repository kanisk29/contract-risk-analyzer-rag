import warnings
warnings.filterwarnings("ignore")

import os
import re
import time
from groq import Groq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_classic.retrievers import BM25Retriever,EnsembleRetriever
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
load_dotenv()


def get_client():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("GROQ_API_KEY not found")
    return Groq(api_key=key)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_vectorstore():
    return Chroma(
        persist_directory="db/chroma_db",
        embedding_function=embeddings
    )

vecstore = get_vectorstore()
raw = vecstore.get()
retriever = vecstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 10,
        "lambda_mult": 0.7
    }
)
docs = []

if raw and raw.get("documents"):
    docs = [
        Document(page_content=txt, metadata={"id": i})
        for i, txt in enumerate(raw["documents"], 1)
    ]

if docs:
    bm_25 = BM25Retriever.from_documents(docs)
    bm_25.k = 5
    hybrid_retriever = EnsembleRetriever(
        retrievers=[retriever, bm_25],
        weights=[0.4, 0.6]
    )
else:
    print("DB EMPTY → fallback to vector only")
    hybrid_retriever = retriever

def groq_llm(prompt, retries=3, delay=8):
    for attempt in range(retries):
        try:
            client = get_client()
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are an Indian contract risk analysis assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            err = str(e)
            print(f"GROQ ERROR (attempt {attempt+1}): {err}")
            if "rate_limit" in err and attempt < retries - 1:
                wait_match = re.search(r'try again in (\d+)m', err)
                wait = int(wait_match.group(1)) * 60 + 10 if wait_match else delay
                print(f"Rate limited. Waiting {min(wait, 30)}s...")
                time.sleep(min(wait, 30))
            else:
                return f"Error Analyzing Clause: {err}"
    return "Error Analyzing Clause: Max retries exceeded"

def split_clauses(text):
    numbered = re.split(
        r'\n(?=\s*(?:clause|article|section|schedule)?\s*\d+[\.\)]\s)',
        text,
        flags=re.IGNORECASE
    )
    if len(numbered) > 2:
        clauses = [c.strip() for c in numbered]
    else:
        clauses = [c.strip() for c in re.split(r'\n{2,}', text)]

    if len(clauses) <= 2:
        clauses = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

    merged = []
    for c in clauses:
        c = c.strip()
        if not c:
            continue
        if len(c) < 60 and merged:
            merged[-1] = merged[-1] + " " + c
        else:
            merged.append(c)

    return [c for c in merged if len(c) >= 40]

def _words(text):
    return set(re.sub(r'[^\w\s]', '', text.lower()).split())

def unique_clauses(clauses, threshold=0.8):
    seen, result = [], []
    for c in clauses:
        wc = _words(c)
        is_dup = any(
            len(wc & _words(s)) / max(len(wc | _words(s)), 1) >= threshold
            for s in seen
        )
        if not is_dup:
            seen.append(c)
            result.append(c)
    return result

RISK_KEYWORDS = [
    "perpetual", "unlimited", "exclusive", "irrevocable", "sole discretion",
    "non-refundable", "indemnify", "waive", "forfeit", "penalt",
    "liquidated damages", "unilateral", "terminate without notice"
]

def highlight(text):
    for k in RISK_KEYWORDS:
        text = re.sub(k, f"**{k}**", text, flags=re.IGNORECASE)
    return text

def parse_risk(output):
    match = re.search(r'Risk Level\s*:\s*(VERY HIGH|MEDIUM[\s\-]?HIGH|HIGH|MEDIUM|LOW)', output, re.IGNORECASE)
    if match:
        val = match.group(1).upper()
        if "VERY" in val or ("MEDIUM" in val and "HIGH" in val):
            return "HIGH"
        if "HIGH" in val:
            return "HIGH"
        if "MEDIUM" in val:
            return "MEDIUM"
    return "LOW"

def compute_score(risks):
    if not risks:
        return 0.0
    high   = risks.count("HIGH")
    medium = risks.count("MEDIUM")
    total  = len(risks)
    raw    = (high * 3 + medium * 2 + (total - high - medium)) / total
    base   = round((raw - 1) / 2 * 10, 2)
    penalty = min(high * 0.5, 2.0)
    return min(round(base + penalty, 2), 10.0)

def interpret_score(score):
    if score >= 8: return "CRITICAL RISK"
    if score >= 6: return "HIGH RISK"
    if score >= 4: return "MODERATE RISK"
    return "LOW RISK"

def summary(score, risk_count):
    h, m = risk_count["HIGH"], risk_count["MEDIUM"]
    if h >= 3: return f"Contract is highly risky — {h} critical clauses detected"
    if h >= 2: return "Contract contains multiple high-risk clauses"
    if h == 1: return "Contract has at least one serious risk"
    if m >= 2: return "Contract has several medium-risk clauses worth negotiating"
    return "Contract is relatively safe"

def analyze_clause(clause):
    retrieved_docs = hybrid_retriever.invoke(clause)
    context = ""
    if retrieved_docs:
        context = "\n\n".join([doc.page_content for doc in retrieved_docs[:3]])

    prompt = f"""You are an Indian contract risk analysis assistant.

    Input Clause:
    {clause}

    Reference Knowledge (supporting context):
    {context}

    Instructions:
    - First understand the clause itself. Do NOT rely on reference knowledge to determine clause type.
    - Clause Type must be inferred strictly from the clause content.
    - Use reference knowledge only to support reasoning.
    - Risk Level must be exactly one of: LOW, MEDIUM, HIGH.
    - Only mark HIGH if there is clear unfairness, illegality, or strong imbalance.
    - If clause is standard and reasonable → return LOW.
    - Do NOT force legal principles if not applicable.

    Output strictly in this format:

    Clause Type: <type>

    Risk Level: <LOW / MEDIUM / HIGH>

    Why Risky:
    - <point 1>
    - <point 2>

    Legal Basis:
    - <relevant principle OR None>
    """

    return groq_llm(prompt)

def analyze_contract(text):
    clauses = split_clauses(text)
    clauses = unique_clauses(clauses)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = [None] * len(clauses)
    risks = [None] * len(clauses)
    highlighted_clauses = [None] * len(clauses)

    def process(i, clause):
        h_clause = highlight(clause)
        out = analyze_clause(h_clause)
        return i, h_clause, out, parse_risk(out)

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process, i, clause) for i, clause in enumerate(clauses)]

        for future in as_completed(futures):
            i, h_clause, out, risk = future.result()
            results[i] = out
            risks[i] = risk
            highlighted_clauses[i] = h_clause

    score = compute_score(risks)
    risk_count = {
        "HIGH": risks.count("HIGH"),
        "MEDIUM": risks.count("MEDIUM"),
        "LOW": risks.count("LOW")
    }

    return {
        "results": results,
        "clauses": highlighted_clauses,
        "score": score,
        "risk_level": interpret_score(score),
        "summary": summary(score, risk_count),
        "risk_count": risk_count
    }