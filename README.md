# Contract Risk Analyzer (RAG)

An automated system for analyzing legal contracts and identifying high-risk clauses under the framework of Indian Contract Law. The system uses a Retrieval-Augmented Generation pipeline with hybrid search and LLM reasoning to produce structured risk assessments.

## Features

- Clause-level risk detection across contract text  
- Risk classification into LOW, MEDIUM, and HIGH  
- Legal grounding using principles from the Indian Contract Act, 1872  
- Mitigation suggestions with safer clause rewrites  
- Hybrid retrieval combining vector search, keyword search, and reranking  

## Tech Stack

| Component          | Technology        |
|-------------------|------------------|
| Framework         | FastAPI          |
| Orchestration     | LangChain        |
| Vector Database   | ChromaDB         |
| LLM               | Groq (LLaMA 3)   |
| Reranking         | Cohere Rerank    |
| Embeddings        | HuggingFace      |

## Analysis Example:

### Input Clause:

The employee shall not work with any competitor for 3 years after termination.

### Output Assessment

```json
{
  "results": [
    "Clause Type: Non-Compete (India)\n\nRisk Level: HIGH\n\nWhy Risky:\n- Restricts working after contract ends\n- Broad scope (industry-wide)\n- Long duration\n\nLegal Basis:\n- Indian Contract Act (Restraint of Trade), which generally renders post-contract non-compete clauses unenforceable\n\nSuggested Fix:\n- Use non-solicitation instead\n- Limit restriction during contract only"
  ],
  "score": 9.9,
  "risk_level": "CRITICAL RISK",
  "summary": "Contract has at least one serious risk",
  "risk_count": {
    "HIGH": 1,
    "MEDIUM": 0,
    "LOW": 0
  }
```
