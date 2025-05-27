from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from datasets import load_dataset
from src.helper.json import _write_json
from src.constants.mapping import LABEL_NAMES
from src.constants.results import DIVERGENCE_RESULT_TEMPLATE

import numpy as np
import pandas as pd
import json

def _compute_divergences(jv_sentences, id_sentences):
    vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    vectorizer.fit(jv_sentences + id_sentences)

    vec_jv = vectorizer.transform(jv_sentences).toarray().sum(axis=0) + 1
    vec_id = vectorizer.transform(id_sentences).toarray().sum(axis=0) + 1

    dist_jv = vec_jv / vec_jv.sum()
    dist_id = vec_id / vec_id.sum()

    kl = 0.5 * (entropy(dist_jv, dist_id) + entropy(dist_id, dist_jv))
    js = jensenshannon(dist_jv, dist_id) ** 2

    jv_vecs = vectorizer.transform(jv_sentences).toarray() + 1
    id_vecs = vectorizer.transform(id_sentences).toarray() + 1
    jv_dists = jv_vecs / jv_vecs.sum(axis=1, keepdims=True)
    id_dists = id_vecs / id_vecs.sum(axis=1, keepdims=True)

    js_scores = [jensenshannon(jv, id_) ** 2 for jv, id_ in zip(jv_dists, id_dists)]
    js_std = np.std(js_scores)

    print(f"Jensen Score STD: {js_std:.6f}")

    return kl, js


def compute_divergence_json(ds, subset, split, output_dir, jv_col="javanese sentence", id_col="indonesian sentence", label_col="label"):
    ds = load_dataset(ds, subset, split=split)
    df = pd.DataFrame(ds)
    result = DIVERGENCE_RESULT_TEMPLATE.copy()

    for label in sorted(df[label_col].unique()):
        sub_df = df[df[label_col] == label]
        kl, js = _compute_divergences(sub_df[jv_col].tolist(), sub_df[id_col].tolist())
        result[LABEL_NAMES[label]] = {
            "jensen-score": float(round(js, 6)),
            "kl-divergence": float(round(kl, 6))
        }

    kl_overall, js_overall = _compute_divergences(df[jv_col].tolist(), df[id_col].tolist())
    result["Jawa-Indonesia"] = {
        "jensen-score": float(round(js_overall, 6)),
        "kl-divergence": float(round(kl_overall, 6))
    }

    _write_json(result, output_dir)