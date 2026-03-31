# NextWord — AI Next Word Prediction

A full-stack next word prediction app powered by GPT-2 and PyTorch.

```
next-word-predictor/
├── model/
│   └── train.py          # Fine-tune GPT-2 on your own text
├── backend/
│   ├── main.py           # FastAPI prediction server
│   └── requirements.txt
└── frontend/
    └── index.html        # Drop-in web UI (no build step needed)
```

---

## 1 — Backend Setup

```bash
cd backend
pip install -r requirements.txt

# Start the server (uses base GPT-2 by default)
uvicorn main:app --reload --port 8000
```

Visit `http://localhost:8000/health` to verify it's running.

---

## 2 — (Optional) Fine-tune on your own data

```bash
cd model
pip install transformers torch tqdm

# Train — replace sample.txt with your own corpus
python train.py --data sample.txt --epochs 3 --output ../gpt2-finetuned
```

Then point the backend at your fine-tuned model:

```bash
MODEL_PATH=../gpt2-finetuned uvicorn main:app --reload --port 8000
```

---

## 3 — Frontend

Just open `frontend/index.html` in your browser — no build step required.

- Set the backend URL (default `http://localhost:8000`)
- Click **ping** to verify the connection (green dot = ready)
- Type text and hit **Predict →** or press `Ctrl+Enter`
- Click any suggestion chip to append the word and auto-predict

---

## API

**POST** `/predict`
```json
{ "text": "The future of AI", "top_k": 5 }
```
Returns:
```json
{ "suggestions": [{ "word": "is", "score": 0.1832 }, ...] }
```

**GET** `/health` — server & model status
