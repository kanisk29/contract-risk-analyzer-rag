from fastapi import FastAPI
from pydantic import BaseModel
from core.retrieval_pipeline import analyze_contract
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class ContractRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Contract Risk Analyzer API"}

@app.post("/analyze")
def analyze(req: ContractRequest):
    return analyze_contract(req.text)