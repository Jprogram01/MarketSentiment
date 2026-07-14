"""Fine-tune FinBERT on harvested StockTwits labels, with class-weighted loss.

The engineering story is **class imbalance**: StockTwits chatter skews heavily bullish,
so a naive model learns "predict bullish" and posts great accuracy while missing the
bearish signal entirely. The fix — the same shape as a queen/worker imbalance problem —
is a data pipeline plus a weighted loss so the minority class isn't drowned out. The
number that proves it worked is **minority-class recall**, which this script reports.

Needs the [finbert] extra:  pip install -e ".[finbert]"

    python -m marketsentiment.scripts.harvest_labels --out data/labeled.csv
    python -m marketsentiment.scripts.train_finbert --data data/labeled.csv --out models/finbert-st
    export MS_FINBERT_MODEL=models/finbert-st   # serve the fine-tuned checkpoint
"""

from __future__ import annotations

import argparse
import inspect


def _training_args(**kwargs):
    """Build TrainingArguments, tolerating the eval_strategy/evaluation_strategy rename."""
    from transformers import TrainingArguments

    params = inspect.signature(TrainingArguments.__init__).parameters
    strategy = kwargs.pop("eval_strategy", "epoch")
    key = "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"
    kwargs[key] = strategy
    if "save_strategy" in params:
        kwargs.setdefault("save_strategy", strategy)
    return TrainingArguments(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune FinBERT on StockTwits labels.")
    parser.add_argument("--data", default="data/labeled.csv")
    parser.add_argument("--out", default="models/finbert-st")
    parser.add_argument("--base-model", default="ProsusAI/finbert")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import numpy as np
    import pandas as pd
    import torch
    from datasets import Dataset
    from sklearn.metrics import classification_report, precision_recall_fscore_support
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
    )

    # ---- 1. Load + encode labels ----
    df = pd.read_csv(args.data).dropna(subset=["text", "label"])
    labels = sorted(df["label"].unique())
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    df["y"] = df["label"].map(label2id)
    n_labels = len(labels)
    print(f"Loaded {len(df)} rows. Classes: {label2id}")
    print("Class distribution:\n" + df["label"].value_counts().to_string())

    # ---- 2. Stratified split ----
    stratify = df["y"] if df["y"].value_counts().min() >= 2 else None
    train_df, eval_df = train_test_split(
        df, test_size=args.test_size, stratify=stratify, random_state=args.seed
    )

    # ---- 3. Class weights: inverse-frequency, the core imbalance handling ----
    class_weights = compute_class_weight(
        "balanced",
        classes=np.arange(n_labels),
        y=train_df["y"].to_numpy(),
    )
    weight_tensor = torch.tensor(class_weights, dtype=torch.float)
    print("Class weights: " + ", ".join(f"{labels[i]}={w:.3f}" for i, w in enumerate(class_weights)))

    minority_label = train_df["label"].value_counts().idxmin()
    print(f"Minority class: {minority_label!r} — its recall is the headline metric.\n")

    # ---- 4. Tokenize ----
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    def to_ds(frame):
        ds = Dataset.from_dict({"text": frame["text"].tolist(), "labels": frame["y"].tolist()})
        return ds.map(tokenize, batched=True, remove_columns=["text"])

    train_ds, eval_ds = to_ds(train_df), to_ds(eval_df)

    # ---- 5. Model (re-init head; base FinBERT has a 3-way head that may not match) ----
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=n_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    # ---- 6. Weighted-loss Trainer ----
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels_ = inputs.pop("labels")
            outputs = model(**inputs)
            loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor.to(outputs.logits.device))
            loss = loss_fn(outputs.logits.view(-1, n_labels), labels_.view(-1))
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(eval_pred):
        logits, y_true = eval_pred
        y_pred = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=np.arange(n_labels), zero_division=0
        )
        metrics = {"macro_f1": float(f1.mean()), "accuracy": float((y_pred == y_true).mean())}
        for i, lab in enumerate(labels):
            metrics[f"recall_{lab}"] = float(recall[i])
        return metrics

    training_args = _training_args(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=50,
        seed=args.seed,
        report_to="none",
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # ---- 7. Report + save ----
    preds = trainer.predict(eval_ds)
    y_pred = np.argmax(preds.predictions, axis=-1)
    print("\n" + classification_report(preds.label_ids, y_pred, target_names=labels, digits=3))

    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"\nSaved fine-tuned model to {args.out}")
    print(f"Serve it with:  export MS_FINBERT_MODEL={args.out}")


if __name__ == "__main__":
    main()
