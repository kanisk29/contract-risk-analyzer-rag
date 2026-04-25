from fastapi import FastAPI
from pydantic import BaseModel
from core.retrieval_pipeline import analyze_contract

app = FastAPI()

class ContractRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Contract Risk Analyzer API"}

@app.post("/analyze")
def analyze(req: ContractRequest):
    return analyze_contract(req.text)