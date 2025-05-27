import os
import random
import logging
import importlib

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, Dataset
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed,
)
import argparse


def load_data_from_hf() -> pd.DataFrame:
    ds = load_dataset("JavaneseHonorifics/Unggah-Ungguh", "translation", split="train")
    df = ds.to_pandas()
    df = df.rename(columns={
        "javanese sentence": "text",
        "label": "label",
        "group": "group"
    })
    df["label"] = df["label"].astype(int)
    df["group"] = df["group"].astype(str)
    return df


def group_split(df: pd.DataFrame, group_col: str, test_size: float, random_state: int):
    splitter = GroupShuffleSplit(test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=df[group_col]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def compute_metrics(pred):
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(pred.label_ids, preds, average='weighted')
    accuracy = accuracy_score(pred.label_ids, preds)
    return {'accuracy': accuracy, 'f1': f1, 'precision': precision, 'recall': recall}


def cross_validate(model_name, df_train, group_col, text_col, label_col, args) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    gkf = GroupKFold(n_splits=args.k_folds)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(df_train, groups=df_train[group_col])):
        logging.info(f"Fold {fold + 1}/{args.k_folds}")
        train_df = df_train.iloc[train_idx]
        val_df = df_train.iloc[val_idx]

        train_ds = Dataset.from_pandas(train_df)
        val_ds = Dataset.from_pandas(val_df)

        def tokenize(batch):
            return tokenizer(batch[text_col], truncation=True, padding='max_length', max_length=args.max_length)

        train_ds = train_ds.map(tokenize, batched=True)
        val_ds = val_ds.map(tokenize, batched=True)

        data_collator = DataCollatorWithPadding(tokenizer)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=args.num_labels, ignore_mismatched_sizes=True
        )

        output_fold = args.output_dir / model_name.replace("/", "_") / f"fold_{fold}"
        output_fold.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(output_fold),
            seed=args.random_state,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.num_epochs,
            weight_decay=args.weight_decay,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir=str(output_fold / "logs"),
            logging_steps=args.logging_steps,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        metrics = trainer.predict(val_ds)
        fold_metrics.append(compute_metrics(metrics))

    avg = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
    return {'Model': model_name, **avg}


def train_and_evaluate(model_name, df_train, df_test, group_col, text_col, label_col, args) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = Dataset.from_pandas(df_train)
    test_ds = Dataset.from_pandas(df_test)

    def tokenize(batch):
        return tokenizer(batch[text_col], truncation=True, padding='max_length', max_length=args.max_length)

    train_ds = train_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=args.num_labels, ignore_mismatched_sizes=True
    )

    output_final = args.output_dir / model_name.replace("/", "_") / "final"
    output_final.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_final),
        seed=args.random_state,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=str(output_final / "logs"),
        logging_steps=args.logging_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    test_pred = trainer.predict(test_ds)
    test_metrics = compute_metrics(test_pred)
    return {'Model': model_name, **test_metrics}

def group_cross_validation(args):
    seed = args.random_state
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed)

    logging.basicConfig(level=logging.INFO)
    logging.info(f"Transformers version: {importlib.import_module('transformers').__version__}")
    logging.info(f"Datasets version: {importlib.import_module('datasets').__version__}")
    logging.info(f"PyTorch version: {torch.__version__}")

    df = load_data_from_hf()
    df_train, df_test = group_split(df, "group", args.test_size, args.random_state)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(args.output_dir / "df_train.csv", index=False)
    df_test.to_csv(args.output_dir / "df_test.csv", index=False)

    cv_results = []
    for model in args.model_names:
        result = cross_validate(model, df_train, group_col="group", text_col="text", label_col="label", args=args)
        cv_results.append(result)
    pd.DataFrame(cv_results).to_csv(args.output_dir / "cross_val_results.csv", index=False)
    logging.info("Cross-validation completed.")

    best_models = pd.DataFrame(cv_results).sort_values("f1", ascending=False).head(3)["Model"].tolist()

    test_results = []
    for model in best_models:
        metrics = train_and_evaluate(model, df_train, df_test, group_col="group", text_col="text", label_col="label", args=args)
        test_results.append(metrics)
    pd.DataFrame(test_results).to_csv(args.output_dir / "test_results.csv", index=False)
    logging.info("Final evaluation completed.")