import warnings
warnings.filterwarnings("ignore")

import os
from dotenv import load_dotenv
from groq import Groq

import re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_cohere import CohereRerank

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

def get_vectorstore():
    
    return Chroma(
        persist_directory="db/chroma_db",
        embedding_function=embeddings
    )

vecstore = get_vectorstore()


retriever = vecstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 10,
        "fetch_k": 25,
        "lambda_mult": 0.7
    }
)

reranker = CohereRerank(
    model="rerank-english-v3.0",
    top_n=5,
    cohere_api_key=os.getenv("COHERE_API_KEY")
)

def groq_llm(prompt):
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an Indian contract risk analysis assistant"},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except:
        return "Error Analyzing Clause"


def split_clauses(text):
    return [c.strip() for c in re.split(r'\n|\.', text) if len(c.strip()) > 30]


def highlight(text):
    keywords = ["perpetual","unlimited","exclusive","irrevocable","sole discretion"]
    for k in keywords:
        text = re.sub(k, f"**{k}**", text, flags=re.IGNORECASE)
    return text

def parse_risk(output):
    text = output.upper()

    if "VERY HIGH" in text:
        return "HIGH"
    if "MEDIUM TO HIGH" in text or "MEDIUM-HIGH" in text:
        return "HIGH"
    if "HIGH" in text:
        return "HIGH"
    if "MEDIUM" in text:
        return "MEDIUM"
    if "LOW" in text:
        return "LOW"

    return "LOW"


def compute_score(results):
    score_map = {"LOW":1, "MEDIUM":2, "HIGH":3}
    total = 0
    count = 0
    for r in results:
        for k,v in score_map.items():
            if f"Risk Level: {k}" in r:
                total += v
                count += 1
                break
    return round((total / max(count,1)) * 3.3, 2)

def interpret_score(score):
    if score >= 8:
        return "CRITICAL RISK"
    elif score >= 6:
        return "HIGH RISK"
    elif score >= 4:
        return "MODERATE RISK"
    else:
        return "LOW RISK"


def summary(score, risk_count):
    if risk_count["HIGH"] >= 2:
        return "Contract contains multiple high-risk clauses"
    if risk_count["HIGH"] == 1:
        return "Contract has at least one serious risk"
    return "Contract is relatively safe"

def unique_clauses(clauses):
    seen = set()
    res = []
    for c in clauses:
        key = c[:60]
        if key not in seen:
            seen.add(key)
            res.append(c)
    return res


def analyze_clause(clause):
    retrieved_docs = retriever.invoke(clause)

    if not retrieved_docs:
         return "Risk Level: LOW\nWhy Risky\n: No relevant context found"
    
    reranked_docs = reranker.compress_documents(retrieved_docs, query=clause)

    context = "\n\n".join([doc.page_content for doc in reranked_docs])

    prompt = f"""
                You are an Indian contract risk analysis assistant.

                Input Clause:
                {clause}

                Reference Knowledge (authoritative context):
                {context}

                Instructions:
                - Base your answer ONLY on the reference knowledge.
                - Risk Level must be exactly one of: LOW, MEDIUM, HIGH (no other variants)
                - Apply principles from Indian contract law.
                - Be specific, not generic.
                - If no significant risk, return LOW.
                - Do not hallucinate beyond context

                Output strictly in this format and NOTHING ELSE:

                Clause Type: <type>

                Risk Level: <LOW / MEDIUM / HIGH>

                Why Risky:
                - <write in points>

                Legal Basis:
                - <relevant principle>

                Suggested Fix:
                - <rewrite>

                Keep it concise.
                """
    return groq_llm(prompt)


def analyze_contract(text):
    clauses = split_clauses(text)
    clauses = unique_clauses(clauses)

    results = []
    risks = []

    for clause in clauses[:7]:
        clause = highlight(clause)
        out = analyze_clause(clause)
        results.append(out)
        risks.append(parse_risk(out))

    score = compute_score(results)

    risk_count = {
        "HIGH": risks.count("HIGH"),
        "MEDIUM": risks.count("MEDIUM"),
        "LOW": risks.count("LOW")
    }

    return {
        "results": results,
        "score": score,
        "risk_level": interpret_score(score),
        "summary": summary(score, risk_count),
        "risk_count": risk_count
    }


