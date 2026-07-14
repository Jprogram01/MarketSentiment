"""Fine-tune FinBERT on harvested StockTwits labels — SKELETON.

The engineering story here is class imbalance (StockTwits skews heavily bullish), the
same shape as the ant queen/worker problem: the interesting work is the data pipeline +
weighted loss, not the model. This is a runnable-shaped skeleton with the imbalance
handling stubbed at the TODO. Needs the [finbert] extra:  pip install -e ".[finbert]"

    python -m marketsentiment.scripts.train_finbert --data data/labeled.csv --out models/finbert-st
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune FinBERT on StockTwits labels.")
    parser.add_argument("--data", default="data/labeled.csv")
    parser.add_argument("--out", default="models/finbert-st")
    parser.add_argument("--base-model", default="ProsusAI/finbert")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    import numpy as np
    import pandas as pd
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight

    df = pd.read_csv(args.data)
    labels = sorted(df["label"].unique())
    label2id = {lab: i for i, lab in enumerate(labels)}
    df["y"] = df["label"].map(label2id)

    train_df, eval_df = train_test_split(
        df, test_size=0.2, stratify=df["y"], random_state=42
    )

    # --- The imbalance handling: class weights inversely proportional to frequency. ---
    class_weights = compute_class_weight(
        "balanced", classes=np.array(sorted(label2id.values())), y=train_df["y"].to_numpy()
    )
    print(f"Classes: {label2id}\nClass weights: {dict(zip(labels, class_weights.round(3)))}")

    # TODO: fine-tune with a weighted loss. Sketch:
    #   from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
    #                             Trainer, TrainingArguments)
    #   tok = AutoTokenizer.from_pretrained(args.base_model)
    #   model = AutoModelForSequenceClassification.from_pretrained(
    #       args.base_model, num_labels=len(labels))
    #   Subclass Trainer.compute_loss to apply
    #       nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float))
    #   Tokenize train/eval, run Trainer(...).train(), then model.save_pretrained(args.out).
    #   Point MS_FINBERT_MODEL at args.out to serve the fine-tuned checkpoint.
    #
    #   Also report queen/worker-style per-class recall — the minority (bearish) class
    #   recall is the number that proves the imbalance work paid off.

    print(
        f"\n[skeleton] Would fine-tune {args.base_model} for {args.epochs} epochs "
        f"→ {args.out}\nFill in the Trainer block above, then evaluate:\n"
    )
    print(classification_report.__doc__.splitlines()[0])  # nudge: report per-class recall


if __name__ == "__main__":
    main()
