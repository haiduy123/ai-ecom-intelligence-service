from __future__ import annotations

from collections import Counter
from typing import Any
import numpy as np
import json

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    SentenceTransformer = None
    cosine_similarity = None

def normalize_aspect(aspect: str) -> str:
    return " ".join(str(aspect).strip().lower().split())

def normalize_sentiment(sentiment: str) -> str:
    return str(sentiment).strip().lower()

def extract_pairs(
        aspects: list[dict[str, Any]],
) -> set[tuple[str, str]]:
    """
    Convert ABSA predictions/labels into:
        {(aspect, sentiment), ...}
    """
    pairs = set()

    for item in aspects:
        if not isinstance(item, dict):
            continue
    
        aspect = item.get("aspect")
        sentiment = item.get("sentiment")

        if aspect is None or sentiment is None:
            continue

        pairs.add(
            (
                normalize_aspect(aspect),
                normalize_sentiment(sentiment)
            )
        )
    
    return pairs

def match_aspects(prediction: dict[str, Any],
                  reference: dict[str, Any],
    ) -> tuple[set[str], set[str], set[str]]:

    pred_map = extract_pairs(
        prediction.get("aspects", [])
    )

    ref_map = extract_pairs(
        reference.get("aspects", [])
    )

    pred_aspects = set(pred_map.keys())
    ref_aspects = set(ref_map.keys())

    matched = pred_aspects & ref_aspects
    false_positive = pred_aspects - ref_aspects
    false_negative = ref_aspects - pred_aspects

    return (
        matched,
        false_positive,
        false_negative,
    )

def one_to_one_matches(similarity, threshold: float):
    if similarity.size == 0:
        return []
    
    candidates = [
        (i, j, float(similarity[i, j]))
        for i in range(similarity.shape[0])
        for j in range(similarity.shape[1])
        if float(similarity[i, j]) >= threshold
    ]

    candidates.sort(key=lambda x: x[2], reverse=True)
    used_gold, used_pred, matches = set(), set(), []
    for i, j, score in candidates:
        if i not in used_gold and j not in used_pred:
            used_gold.add(i)
            used_pred.add(j)
            matches.append((i, j, score))
    return matches

def merge_similar_predictions(ps: list[dict], model: SentenceTransformer, merge_threshold: float = 0.80) -> list[dict]:
    if not ps or len(ps) <= 1:
        return ps
    
    aspect_texts = [str(item.get("aspect", "")).strip().lower() for item in ps]
    embeddings = model.encode(aspect_texts, convert_to_numpy=True, normalize_embeddings=True)
    sim_matrix = cosine_similarity(embeddings, embeddings)

    merged = []
    visited = set()

    for i in range(len(ps)):
        if i in visited:
            continue

        current_group = [ps[i]]
        current_indices = [i]
        visited.add(i)

        base_sentiment = str(ps[i].get("sentiment", "")).strip().lower()

        for j in range(i + 1, len(ps)):
            if j not in visited:
                target_sentiment = str(ps[j].get("sentiment", "")).strip().lower()

                if base_sentiment == target_sentiment and sim_matrix[i, j] >= merge_threshold:
                    current_group.append(ps[j])
                    current_indices.append(j)
                    visited.add(j)
        if len(current_group) > 1:
            sub_sim_matrix = sim_matrix[np.ix_(current_indices, current_indices)]
            mean_similarities = sub_sim_matrix.mean(axis=1)
            best_local_idx = np.argmax(mean_similarities)
            best_representative = current_group[best_local_idx]

            combined_evidence = set()
            for item in current_group:
                ev = item.get("evidence", "")
                if ev:
                    combined_evidence.add(ev)
            
            merged_item = {
                "aspect": best_representative["aspect"],
                "sentiment": base_sentiment,
                "evidence": " | ".join(list(combined_evidence))
            }
            merged.append(merged_item)
        else:
            merged.append(current_group[0])

    return merged

def calculate_aspect_metrics(
        predictions: list[dict[str, Any]],
        references: list[dict[str, Any]]
) -> dict[str, float]:
    """Aspect extraction metrics"""

    tp = 0
    fp = 0
    fn = 0

    for pred, ref in zip(predictions, references):
        pred_aspects = {
            normalize_aspect(x['aspect'])
            for x in pred.get("aspects", [])
            if isinstance(x, dict) and x.get("aspect")
        }

        ref_aspects = {
            normalize_aspect(x["aspect"])
            for x in ref.get("aspects", [])
            if isinstance(x, dict) and x.get("aspect")
        }

        tp += len(pred_aspects & ref_aspects)
        fp += len(pred_aspects - ref_aspects)
        fn += len(ref_aspects - pred_aspects)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "aspect_precision": precision,
        "aspect_recall": recall,
        "aspect_f1": f1,
        "aspect_tp": tp,
        "aspect_fp": fp,
        "aspect_fn": fn,
    }

def calculate_sentiment_metrics(
        predictions: list[dict[str, Any]],
        references: list[dict[str, Any]]
) -> dict[str, Any]:
    """
    Sentiment metrics for correctly matched aspects.

    Sentiment is evaluated only when the predicted aspect exactly matches
    a reference aspect
    """

    y_true = []
    y_pred = []

    for pred, ref in zip(predictions, references):
        pred_map = {}

        for item in pred.get("aspects", []):
            if not isinstance(item, dict):
                continue
        
            aspect = item.get("aspect")
            sentiment = item.get("sentiment")

            if aspect and sentiment:
                pred_map[normalize_aspect(aspect)] = normalize_sentiment(sentiment)

        for item in ref.get("aspects", []):
            if not isinstance(item, dict):
                continue
                
            aspect = item.get("aspect")
            sentiment = item.get("sentiment")

            if not aspect or not sentiment:
                continue

            aspect_norm = normalize_aspect(aspect)

            if aspect_norm in pred_map:
                y_true.append(normalize_sentiment(sentiment))
                y_pred.append(pred_map[aspect_norm])

    if not y_true:
        return {
            "sentiment_accuracy": 0.0,
            "sentiment_macro_f1": 0.0,
            "sentiment_weighted_f1": 0.0,
            "sentiment_samples": 0
        }
    
    labels = sorted(
        set(y_true) | set(y_pred)
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    confusion = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    ).tolist()

    return {
        "sentiment_accuracy": accuracy,
        "sentiment_macro_f1": macro_f1,
        "sentiment_weighted_f1": weighted_f1,
        "sentiment_samples": len(y_true),
        "sentiment_labels": labels,
        "sentiment_confusion_matrix": confusion,
        "sentiment_classification_report": report,
    }
    
def calculate_end_to_end_metrics(
    predictions: list[dict[str, Any]],
    references: list[dict[str, Any]]
) -> dict[str, float]:
    tp = 0
    fp = 0
    fn = 0

    for prediction, reference in zip(predictions, references):
        pred_map = extract_pairs(prediction.get("aspects", []))
        ref_map = extract_pairs(reference.get("aspects", []))

        pred_pairs = pred_map
        ref_pairs = ref_map

        matched_pairs = pred_pairs & ref_pairs

        false_positive_pairs = (
            pred_pairs - ref_pairs
        )

        false_negative_pairs = (
            ref_pairs - pred_pairs
        )

        tp += len(matched_pairs)
        fp += len(false_positive_pairs)
        fn += len(false_negative_pairs)


    precision = tp / (tp + fp) if tp + fp else 0.0

    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "end_to_end_precision": precision,
        "end_to_end_recall": recall,
        "end_to_end_f1": f1,
        "end_to_end_tp": tp,
        "end_to_end_fp": fp,
        "end_to_end_fn": fn,
    }

def calculate_embedding_metrics(
        predictions: list[dict[str, Any]],
        references: list[dict[str, Any]],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        threshold: float = 0.82
) -> dict[str, Any]:
    if not SentenceTransformer:
        print("Warning: sentence-transformers not installed. Skipping embedding metrics.")
        return {}
    
    output_file: str = "evaluated_predictions_merged.json"
    model = SentenceTransformer(model_name)

    all_aspects = []
    reviews = []
    processed_predictions = []

    # Gather all aspects to batch encode
    for pred, ref in zip(predictions, references):
        gs = [x for x in ref.get("aspects", []) if isinstance(x, dict)]

        raw_ps = [x for x in pred.get("aspects", []) if isinstance(x, dict)]
        ps = merge_similar_predictions(raw_ps, model, merge_threshold=0.75)

        pred_copy = pred.copy()
        if "prediction" in pred_copy:
            pred_copy["prediction"]["aspects"] = ps
        else:
            pred_copy["aspects"] = ps
        processed_predictions.append(pred_copy)

        reviews.append((gs, ps))
        for item in gs + ps:
            all_aspects.append(str(item.get("aspect", item.get("name", ""))).strip().lower() or "[missing]")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed_predictions, f, ensure_ascii=False, indent=4)
    print(f"Saved prediction in file: {output_file}")

    aspect_embs = model.encode(all_aspects, convert_to_numpy=True, normalize_embeddings=True) if all_aspects else None

    cursor = 0
    sem_tp = sem_fp = sem_fn = 0
    e2e_tp = e2e_fp = e2e_fn = 0
    y_true, y_pred = [], []
    index = 0

    for gs, ps in reviews:
        index += 1
        ng, npred = len(gs), len(ps)

        gemb = aspect_embs[cursor:cursor + ng] if aspect_embs is not None else None
        cursor += ng
        pemb = aspect_embs[cursor:cursor + npred] if aspect_embs is not None else None
        cursor += npred
        
        if ng and npred and gemb is not None and pemb is not None:
            aspect_sim = cosine_similarity(gemb, pemb)
            matches = one_to_one_matches(aspect_sim, threshold)

            used_g = {i for i, _, _ in matches}
            used_p = {j for _, j, _ in matches}

            sem_tp += len(matches)
            sem_fn += ng - len(used_g)
            sem_fp += npred - len(used_p)

            for i, j, score in matches:
                g_sent = normalize_sentiment(gs[i].get("sentiment", ""))
                p_sent = normalize_sentiment(ps[j].get("sentiment", ""))

                y_true.append(g_sent)
                y_pred.append(p_sent)

                if g_sent == p_sent:
                    e2e_tp += 1
                else:
                    e2e_fp += 1
                    e2e_fn += 1

            e2e_fn += ng - len(used_g)
            e2e_fp += npred - len(used_p)
        else:
            sem_fn += ng
            sem_fp += npred
            e2e_fn += ng
            e2e_fp += npred
    
    asp_p = sem_tp / (sem_tp + sem_fp) if sem_tp + sem_fp else 0.0
    asp_r = sem_tp / (sem_tp + sem_fn) if sem_tp + sem_fn else 0.0
    asp_f1 = 2 * asp_p * asp_r / (asp_p + asp_r) if asp_p + asp_r else 0.0

    e2e_p = e2e_tp / (e2e_tp + e2e_fp) if e2e_tp + e2e_fp else 0.0
    e2e_r = e2e_tp / (e2e_tp + e2e_fn) if e2e_tp + e2e_fn else 0.0
    e2e_f1 = 2 * e2e_p * e2e_r / (e2e_p + e2e_r) if e2e_p + e2e_r else 0.0

    sent_acc = accuracy_score(y_true, y_pred) if y_true else 0.0

    return {
        "emb_aspect_precision": asp_p,
        "emb_aspect_recall": asp_r,
        "emb_aspect_f1": asp_f1,
        "emb_aspect_tp": sem_tp,
        "emb_aspect_fp": sem_fp,
        "emb_aspect_fn": sem_fn,
        "emb_sentiment_accuracy": sent_acc,
        "emb_end_to_end_precision": e2e_p,
        "emb_end_to_end_recall": e2e_r,
        "emb_end_to_end_f1": e2e_f1,
    }

# -------------------
# Combined metrics
# -------------------
def calculate_metrics(
    predictions: list[dict[str, Any]],
    references: list[dict[str, Any]],
    threshold
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}

    metrics.update(calculate_aspect_metrics(predictions, references))
    metrics.update(calculate_sentiment_metrics(predictions, references))

    metrics.update(calculate_end_to_end_metrics(predictions, references))
    metrics.update(calculate_embedding_metrics(predictions, references, model_name='sentence-transformers/all-MiniLM-L6-v2', threshold=threshold))

    return metrics