"""
Fine-tune GPT-2 for Next Word Prediction
Usage: python train.py --data_path your_text.txt --output_dir ./model
"""

import argparse
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm


# ── Dataset ───────────────────────────────────────────────────────────────────

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.examples = []
        for text in texts:
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            self.examples.append({
                "input_ids": encoded["input_ids"].squeeze(),
                "attention_mask": encoded["attention_mask"].squeeze(),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def load_texts(data_path):
    """Load text file and split into overlapping chunks."""
    with open(data_path, "r", encoding="utf-8") as f:
        content = f.read()
    chunks = [content[i:i+512] for i in range(0, len(content), 256)]
    return [c.strip() for c in chunks if len(c.strip()) > 20]


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer & model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.to(device)

    # Dataset
    if args.data_path and os.path.exists(args.data_path):
        texts = load_texts(args.data_path)
        print(f"Loaded {len(texts)} text chunks from {args.data_path}")
    else:
        # Built-in demo data
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Natural language processing enables computers to understand human language.",
            "Deep learning models require large amounts of training data.",
            "Python is the most popular programming language for data science.",
            "Neural networks are inspired by the structure of the human brain.",
            "Transformers have revolutionized the field of natural language processing.",
            "The attention mechanism allows models to focus on relevant parts of input.",
            "GPT-2 is a large language model trained on a diverse range of internet text.",
            "Fine-tuning adapts a pretrained model to a specific task or domain.",
        ] * 100
        print("No data_path provided — using built-in demo sentences.")

    dataset = TextDataset(texts, tokenizer, max_length=args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    # Training loop
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch in progress:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Labels = input_ids (standard causal LM objective)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} complete — avg loss: {avg_loss:.4f}")

    # Save fine-tuned model
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nModel saved to '{args.output_dir}'")
    print("Point your backend to this directory with: MODEL_PATH=" + args.output_dir)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 for next word prediction")
    parser.add_argument("--data_path",     type=str,   default=None,      help="Path to .txt training file (optional)")
    parser.add_argument("--output_dir",    type=str,   default="./model", help="Directory to save the fine-tuned model")
    parser.add_argument("--epochs",        type=int,   default=3,         help="Number of training epochs")
    parser.add_argument("--batch_size",    type=int,   default=4,         help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5,      help="Learning rate")
    parser.add_argument("--max_length",    type=int,   default=128,       help="Max token length per chunk")
    args = parser.parse_args()
    train(args)