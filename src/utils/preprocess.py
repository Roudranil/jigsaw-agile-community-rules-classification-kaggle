import heapq
import logging
import re
from functools import partial
from typing import Optional

import markdown2
import pandas as pd
from bs4 import BeautifulSoup
from datasets import Dataset
from unidecode import unidecode


def sanitize_comment(comment):
    # Convert markdown to HTML, then extract the text (HTML tags removed)
    html = markdown2.markdown(comment)
    text = BeautifulSoup(html, features="html.parser").get_text()

    # Convert markdown links [text](url) to just "text"
    # Must be done on original comment, but here we do it on extracted text to be safe
    # To be sure, you can do it before markdown conversion, but here kept as is for simplicity

    # The markdown2 conversion often converts markdown links into HTML anchors,
    # so links should already have URL removed by BeautifulSoup.get_text().
    # However, just in case, let's remove leftover markdown links from original comment first:
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", comment)
    # Then re-run markdown2 and extract text again to clean up
    html = markdown2.markdown(text)
    text = BeautifulSoup(html, features="html.parser").get_text()

    # Replace URLs with the url itself as plain text
    # Extract URLs and replace markdown-style inline URLs [text](url) with url only is already handled,
    # but explicit URLs just in text should be replaced with the URL string, not removed.
    # For example: "visit http://example.com for more" should keep "http://example.com" as is.
    # So to convert URL markdown to plain URLs requires us to find URLs and keep them as text.

    # Here, let's find URLs and replace any markdown link forms to plain URLs if any missed:
    # But since markdown2 and BeautifulSoup stripped them to plain text, raw URLs remain intact.

    # So no need to remove URLs, but ensure any URLs embedded in text like "https://..." remain
    # We can optionally extract and re-insert URLs if you want, but seems not required.

    # Just to be sure, let's convert all URL-like substrings to themselves surrounded by spaces (to separate)
    # This helps if URLs are concatenated with other text.
    url_pattern = re.compile(r"((?:http|https)://[^\s]+|www\.[^\s]+)", re.IGNORECASE)
    text = url_pattern.sub(lambda m: m.group(0), text)

    # Convert non-unicode characters to unicode (ASCII compatible)
    text = unidecode(text)

    # Normalize whitespace
    text = " ".join(text.split()).lower()

    return text


def create_master_dataset(
    train: pd.DataFrame, test: pd.DataFrame, logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    if logger:
        logger.info("Starting master dataset creation")
        logger.info(f"Input - Train: {len(train)} rows, Test: {len(test)} rows")

    # 1. From train: use body
    if logger:
        logger.debug("Extracting main training data from 'body' column")
    train_main = train[["body", "rule", "subreddit", "rule_violation"]].copy()
    train_main = train_main.rename(
        columns={"body": "comment", "rule_violation": "violation"}
    )
    if logger:
        logger.debug(f"Main training data: {len(train_main)} records")

    # 2. From train AND test: from positive/negative examples

    # Helper to melt examples from a single dataframe
    def extract_examples(
        df, prefix_pos="positive_example_", prefix_neg="negative_example_"
    ):
        records = []

        # For positive examples
        for i in [1, 2]:
            col = f"{prefix_pos}{i}"
            # Ensure column exists and drop NA
            if col in df.columns:
                subdf = df[["rule", "subreddit", col]].dropna(subset=[col])
                if logger:
                    logger.debug(
                        f"Extracting {len(subdf)} positive examples from {col}"
                    )
                for _, row in subdf.iterrows():
                    records.append(
                        {
                            "comment": row[col],
                            "rule": row["rule"],
                            "subreddit": row["subreddit"],
                            "violation": 1,
                        }
                    )

        # For negative examples
        for i in [1, 2]:
            col = f"{prefix_neg}{i}"
            if col in df.columns:
                subdf = df[["rule", "subreddit", col]].dropna(subset=[col])
                if logger:
                    logger.debug(
                        f"Extracting {len(subdf)} negative examples from {col}"
                    )
                for _, row in subdf.iterrows():
                    records.append(
                        {
                            "comment": row[col],
                            "rule": row["rule"],
                            "subreddit": row["subreddit"],
                            "violation": 0,
                        }
                    )

        return pd.DataFrame(records)

    if logger:
        logger.debug("Extracting examples from train dataset")
    train_examples = extract_examples(train)
    if logger:
        logger.debug(f"Train examples extracted: {len(train_examples)} records")

    if logger:
        logger.debug("Extracting examples from test dataset")
    test_examples = extract_examples(test)
    if logger:
        logger.debug(f"Test examples extracted: {len(test_examples)} records")

    # Concatenate all parts
    if logger:
        logger.info("Concatenating all dataset parts")
    master_df = pd.concat(
        [train_main, train_examples, test_examples], ignore_index=True
    )

    # Optional: drop rows with empty or null comment if any sneaked in
    initial_size = len(master_df)
    master_df = master_df.dropna(subset=["comment"])
    if logger and len(master_df) < initial_size:
        logger.warning(
            f"Dropped {initial_size - len(master_df)} rows with null comments"
        )

    # Reset index for cleanliness
    master_df = master_df.reset_index(drop=True)

    if logger:
        logger.info(
            f"Master dataset created successfully: {len(master_df)} total records"
        )
        logger.info(
            f"Violation distribution: {master_df['violation'].value_counts().to_dict()}"
        )

    return master_df


def predict_labels(
    batch,  # ← comes from a HF Dataset
    vectorizer,
    label_vecs,
    column: str,
    labels: list,
    threshold: float = 0.5,
    heap_nlargest: int = 2,
    logger: Optional[logging.Logger] = None,
):
    if logger:
        logger.info(f"Starting prediction for column '{column}'")

    # encode
    texts = batch[column] if isinstance(batch[column], list) else [batch[column]]
    if logger:
        logger.info(f"Processing {len(texts)} texts from batch")

    rule_vecs = vectorizer.transform(texts)
    if logger:
        logger.debug(f"Vectorized texts: shape {rule_vecs.shape}")

    sims = (rule_vecs @ label_vecs.T).toarray()  # cosine similarity (same as sklearn's)
    if logger:
        logger.debug(f"Computed similarity matrix: shape {sims.shape}")

    out = []
    for i, sim_row in enumerate(sims):
        pairs = list(zip(labels, sim_row))
        top2 = heapq.nlargest(heap_nlargest, pairs, key=lambda x: x[1])
        above = [(l, s) for l, s in top2 if s >= threshold]
        chosen = above if above else [max(pairs, key=lambda x: x[1])]

        if logger:
            best_scores = [f"{l}:{s:.3f}" for l, s in top2[:3]]  # Top 3 for debug
            logger.info(
                f"Sample {i}: top scores {best_scores}, "
                f"above threshold: {len(above)}, chosen: {len(chosen)}"
            )

        out.append(", ".join(f"{l}" for l, s in chosen))

    if logger:
        logger.info(f"Prediction completed for {len(out)} samples on column '{column}'")
        logger.info(f"Sample predictions: {out[:3] if len(out) >= 3 else out}")

    return {f"predicted_{column}_feature": out}


def predict_labels_nli(
    batch,
    classifier,
    column: str,
    labels: list,
    threshold: float = 0.8,
    lower_threshold: float = 0.5,  # NEW PARAM ─ default 0.5
    logger: Optional[logging.Logger] = None,
):
    if logger:
        logger.info(f"Starting NLI prediction for column '{column}'")

    texts = batch[column] if isinstance(batch[column], list) else [batch[column]]
    if logger:
        logger.info(f"Processing {len(texts)} texts with NLI classifier")

    outputs = classifier(texts, labels, multi_label=True)
    if isinstance(outputs, dict):  # single-example edge case
        outputs = [outputs]
        if logger:
            logger.debug("Converted single output dict to list format")

    if logger:
        logger.debug(f"NLI classifier returned {len(outputs)} predictions")

    out = []
    fallback_count = 0
    for i, o in enumerate(outputs):
        pairs = list(zip(o["labels"], o["scores"]))  # (label, score)
        top2 = heapq.nlargest(2, pairs, key=lambda x: x[1])
        above = [(l, s) for l, s in top2 if s >= threshold]

        if above:
            chosen = above  # 1 or 2 labels
            if logger:
                logger.debug(
                    f"Sample {i}: {len(above)} labels above threshold {threshold}"
                )
        else:
            top1 = max(pairs, key=lambda x: x[1])  # best label overall
            if top1[1] < lower_threshold:
                chosen = [("general or other", 1.0)]
                fallback_count += 1
                if logger:
                    logger.info(
                        f"Sample {i}: best score {top1[1]:.3f} < {lower_threshold}, using fallback"
                    )
            else:
                chosen = [top1]
                if logger:
                    logger.info(
                        f"Sample {i}: using best label {top1[0]} with score {top1[1]:.3f}"
                    )

        out.append(", ".join(f"{l}" for l, s in sorted(chosen, key=lambda x: -x[1])))

    if logger:
        logger.info(
            f"NLI prediction completed for {len(out)} samples on column '{column}'"
        )
        if fallback_count > 0:
            logger.info(
                f"Used fallback 'general or other' for {fallback_count} samples"
            )
        logger.info(f"Sample predictions: {out[:3] if len(out) >= 3 else out}")

    return {f"predicted_{column}_feature": out}


def _calculate_column_lookup(dataset, column, map_fn, **map_kwargs):
    unique_vals = list(set(dataset[column]))  # unique strings
    mini_ds = Dataset.from_dict({column: unique_vals})  # tiny dataset
    scored_ds = mini_ds.map(  # classify once each
        partial(map_fn, column=column, **map_kwargs), batched=True, batch_size=128
    )
    return dict(zip(unique_vals, scored_ds[f"predicted_{column}_feature"]))


def _build_lookups(
    df,
    rule_vectorizer,
    rule_vecs,
    features,
    subreddit_classifier,
    subreddits,
    logger=None,
):
    rule_lookup = _calculate_column_lookup(
        df,  # or val_hf_small
        column="rule",
        map_fn=predict_labels,
        vectorizer=rule_vectorizer,
        label_vecs=rule_vecs,
        labels=features,
        threshold=0.3,
        heap_nlargest=4,
        logger=logger,
    )

    subreddit_lookup = _calculate_column_lookup(
        df,
        column="subreddit",
        map_fn=predict_labels_nli,
        classifier=subreddit_classifier,
        labels=subreddits,
        threshold=0.8,
        logger=logger,
    )
    return rule_lookup, subreddit_lookup


def _map_lookup(example, rule_lookup, subreddit_lookup):
    example["predicted_rule_feature"] = rule_lookup[example["rule"]]
    example["predicted_subreddit_feature"] = subreddit_lookup[example["subreddit"]]
    return example


def add_classification_preds_rule_subreddit(
    df, rule_lookup=None, subreddit_lookup=None, logger=None
):
    if rule_lookup is None and subreddit_lookup is None:
        rule_lookup, subreddit_lookup = _build_lookups(df, logger=logger)
    result = df.map(
        partial(
            _map_lookup, rule_lookup=rule_lookup, subreddit_lookup=subreddit_lookup
        ),
        desc="Attach predictions",
    )
    return result, rule_lookup, subreddit_lookup
