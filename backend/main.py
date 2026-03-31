"""
FastAPI backend — Next Word Prediction
Run: uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import os

# ── App setup ──────────────────────────────────────────────────────────────────

app = FastAPI(title="Next Word Predictor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model loading ──────────────────────────────────────────────────────────────

MODEL_PATH = os.getenv("MODEL_PATH", "gpt2")   # swap for ./gpt2-finetuned after training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer: GPT2TokenizerFast = None
model: GPT2LMHeadModel = None


@app.on_event("startup")
async def load_model():
    global tokenizer, model
    print(f"Loading model from '{MODEL_PATH}' on {device}...")
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_PATH)
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH).to(device)
    model.eval()
    print("Model ready.")


# ── Schemas ────────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str
    top_k: int = 5          # number of suggestions to return
    max_new_tokens: int = 1  # tokens to generate per suggestion


class Suggestion(BaseModel):
    word: str
    score: float            # normalised probability


class PredictResponse(BaseModel):
    suggestions: list[Suggestion]


# ── Endpoint ───────────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    inputs = tokenizer(req.text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    if input_ids.shape[1] > 1024:
        # GPT-2 max context is 1024 tokens — truncate from the left
        input_ids = input_ids[:, -1024:]

    with torch.no_grad():
        outputs = model(input_ids)
        # logits for the very next token position
        next_token_logits = outputs.logits[0, -1, :]

    # Top-K probabilities
    top_k = min(req.top_k, next_token_logits.size(-1))
    probs = torch.softmax(next_token_logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, top_k)

    suggestions = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        token = tokenizer.decode([idx]).strip()
        if token:  # skip empty / whitespace-only tokens
            suggestions.append(Suggestion(word=token, score=round(prob, 4)))

    return PredictResponse(suggestions=suggestions)


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_PATH, "device": str(device)}