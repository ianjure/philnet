import random
import re
import base64
import copy
from typing import List, Dict, Tuple, Callable, Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.metrics import matthews_corrcoef, precision_recall_curve, auc, confusion_matrix
from sklearn.preprocessing import LabelBinarizer

# Optional imports for models - these must exist in user's environment
try:
    import fasttext
except Exception:
    fasttext = None

try:
    import joblib
except Exception:
    joblib = None

# ----------------------------
# Textual obfuscation utilities
# ----------------------------

HOMOGLYPH_MAP = {
    "a": ["@","4","а"],  # last is Cyrillic a, keep as example (may need font)
    "e": ["3","€","е"],
    "i": ["1","!","і"],
    "o": ["0","О"],
    "s": ["$","5"],
    "t": ["7"],
    "l": ["1","|"],
    "c": ["(", "с"],
}

BENIGN_NOISE_SNIPPETS = [
    "Welcome to our official community page.",
    "Read our privacy policy and updates.",
    "Certified business partner since 2020.",
    "Contact customer support for details.",
    "Please review our terms and conditions.",
    "Follow for more updates!"
]

SUSPICIOUS_WORDS = [
    "login", "verify", "account", "password", "bank", "secure", "update", "confirm",
    "ssn", "credit", "unauthorized", "suspended"
]

def _rand_choice(seq):
    return seq[random.randrange(len(seq))]

def homoglyph_substitute(text: str, prob: float = 0.1) -> str:
    """Replace characters with homoglyphs with probability `prob` per replaceable char."""
    out = []
    for ch in text:
        low = ch.lower()
        if low in HOMOGLYPH_MAP and random.random() < prob:
            repl = _rand_choice(HOMOGLYPH_MAP[low])
            # preserve case if letter
            if ch.isupper():
                repl = repl.upper()
            out.append(repl)
        else:
            out.append(ch)
    return "".join(out)

def insert_benign_noise(text: str, p_insert: float = 0.3) -> str:
    """Randomly insert benign noise sentences into text."""
    if random.random() >= p_insert:
        return text
    noise = " ".join(random.sample(BENIGN_NOISE_SNIPPETS, k=random.randint(1,2)))
    # insert at random position
    words = text.split()
    pos = random.randint(0, max(0, len(words)))
    words[pos:pos] = noise.split()
    return " ".join(words)

def shuffle_words(text: str, p_shuffle: float = 0.1) -> str:
    if random.random() >= p_shuffle:
        return text
    words = text.split()
    if len(words) <= 3:
        return text
    nswap = max(1, int(len(words) * 0.1))
    for _ in range(nswap):
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
    return " ".join(words)

def mask_suspicious_words(text: str, p_mask: float = 0.6) -> str:
    """Replace suspicious words with synonyms or spaced letters."""
    def mask_word(w):
        if random.random() < 0.5:
            return " ".join(list(w))  # space out: "login" -> "l o g i n"
        else:
            # partial mask: lo**n
            if len(w) <= 3:
                return "*" * len(w)
            return w[:2] + "*" * (len(w) - 3) + w[-1]
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, SUSPICIOUS_WORDS)) + r")\b", flags=re.I)
    def repl(m):
        if random.random() < p_mask:
            return mask_word(m.group(0))
        return m.group(0)
    return pattern.sub(repl, text)

def simple_char_substitution(text: str, prob: float = 0.03) -> str:
    """Randomly insert dash/underscore/zero-to-O type edits."""
    out = []
    for ch in text:
        if ch.isalpha() and random.random() < prob:
            out.append(ch + _rand_choice(["-", "_", "."]))
        else:
            out.append(ch)
    return "".join(out)

def obfuscate_text(
    text: str,
    methods: Optional[List[str]] = None,
    seed: Optional[int] = None
) -> str:
    """Run a sequence of obfuscations on visible_text and return new text."""
    if seed is not None:
        random.seed(seed)
    methods = methods or [
        "homoglyph",
        "insert_noise",
        "shuffle",
        "mask_suspicious",
        "char_sub"
    ]
    new_text = str(text) if text is not None else ""
    if "homoglyph" in methods:
        new_text = homoglyph_substitute(new_text, prob=0.08)
    if "insert_noise" in methods:
        new_text = insert_benign_noise(new_text, p_insert=0.25)
    if "shuffle" in methods:
        new_text = shuffle_words(new_text, p_shuffle=0.08)
    if "mask_suspicious" in methods:
        new_text = mask_suspicious_words(new_text, p_mask=0.7)
    if "char_sub" in methods:
        new_text = simple_char_substitution(new_text, prob=0.02)
    # final cleanup
    new_text = re.sub(r"\s{2,}", " ", new_text).strip()
    return new_text

# ----------------------------
# URL / heuristic obfuscations
# ----------------------------

def add_subdomain_to_url(url: str, max_add: int = 2) -> str:
    """Prepend random subdomains to the host part of a URL."""
    try:
        # simple split
        parts = url.split("://", 1)
        if len(parts) == 2:
            scheme, rest = parts
        else:
            scheme, rest = "", parts[0]
        host_and_rest = rest.split("/", 1)
        host = host_and_rest[0]
        rest_suffix = "/" + host_and_rest[1] if len(host_and_rest) == 2 else ""
        # if host contains port or credentials, keep them
        add = ".".join(random.choice(["secure","login","cdn","img","static","api","mail"]) for _ in range(random.randint(1, max_add)))
        new_host = add + "." + host
        return (scheme + "://" if scheme else "") + new_host + rest_suffix
    except Exception:
        return url

def encode_query_in_url(url: str) -> str:
    """Base64-encode the query string portion to create a weird looking URL."""
    if "?" not in url:
        return url
    base, query = url.split("?", 1)
    try:
        enc = base64.urlsafe_b64encode(query.encode()).decode()
        return base + "?q=" + enc
    except Exception:
        return url

def obfuscate_url_and_features(
    url: str,
    features: Dict[str, Any],
    toggle_prob: float = 0.3,
    seed: Optional[int] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Given a URL string and feature dict, produce an obfuscated url and modified features.
    We mutate features in a plausible but adversarial way to try to confuse heuristic model.
    """
    if seed is not None:
        random.seed(seed)
    new_url = str(url) if url is not None else ""
    new_feats = copy.deepcopy(features)

    # Randomly add subdomain(s)
    if random.random() < 0.5:
        new_url = add_subdomain_to_url(new_url, max_add=2)
        # heuristics: increase num_subdomains, url_length
        if "num_subdomains" in new_feats:
            new_feats["num_subdomains"] = min(10, int(new_feats.get("num_subdomains", 0)) + random.randint(1,2))
        if "url_length" in new_feats:
            new_feats["url_length"] = int(new_feats.get("url_length", len(new_url))) + random.randint(1, 10)

    # Toggle uses_https (simulate attacker moving to https)
    if "uses_https" in new_feats and random.random() < toggle_prob:
        new_feats["uses_https"] = not bool(new_feats.get("uses_https", False))

    # Flip has_at_symbol sometimes
    if "has_at_symbol" in new_feats and random.random() < 0.2:
        new_feats["has_at_symbol"] = not bool(new_feats.get("has_at_symbol", False))

    # Mask suspicious TLD by adding extra path or dot
    if "is_suspicious_tld" in new_feats and new_feats.get("is_suspicious_tld", False) and random.random() < 0.6:
        # set to False occasionally
        new_feats["is_suspicious_tld"] = False

    # Add long query encoding occasionally
    if random.random() < 0.3:
        new_url = encode_query_in_url(new_url)
        if "url_has_long_query" in new_feats:
            new_feats["url_has_long_query"] = True

    # Add hyphen or encode hyphen presence
    if random.random() < 0.25 and "has_hyphen" in new_feats:
        new_feats["has_hyphen"] = True

    # Modify counts: num_forms, num_inputs, num_links, etc.
    for k in ["num_forms", "num_inputs", "num_links", "num_password_inputs", "num_hidden_inputs", "num_inline_scripts", "external_js_count", "external_iframe_count", "num_external_domains"]:
        if k in new_feats and random.random() < 0.4:
            # add small noise
            base_val = int(new_feats.get(k, 0))
            new_feats[k] = max(0, base_val + random.randint(-1, 3))

    # Mask suspicious_words presence (if feature exists as boolean or count)
    if "suspicious_words" in new_feats:
        if isinstance(new_feats["suspicious_words"], bool):
            if random.random() < 0.6:
                # flip to False to hide hint
                new_feats["suspicious_words"] = False
        else:
            # numeric count -> reduce or split into visible_text obfuscation
            new_feats["suspicious_words"] = max(0, int(new_feats.get("suspicious_words", 0)) - random.randint(0, 2))

    # has_ip_address: try to mask it by replacing with domain-looking string
    if "has_ip_address" in new_feats and new_feats.get("has_ip_address", False) and random.random() < 0.7:
        new_feats["has_ip_address"] = False
        # and change the url to reflect domain instead of raw ip
        new_url = re.sub(r"(https?://)?\d+\.\d+\.\d+\.\d+", lambda m: "http://example.com", new_url)

    # suspicious_form_action -> flip sometimes
    if "suspicious_form_action" in new_feats and random.random() < 0.5:
        new_feats["suspicious_form_action"] = False

    # add small jitter to url_length if present
    if "url_length" in new_feats:
        new_feats["url_length"] = max(1, int(new_feats.get("url_length", len(new_url))) + random.randint(-5, 5))

    return new_url, new_feats

# ----------------------------
# Dataset synthesis
# ----------------------------

def synthesize_row(
    row: pd.Series,
    text_methods: Optional[List[str]] = None,
    n_variants: int = 1,
    feat_obf_prob: float = 0.7,
    seed: Optional[int] = None
) -> List[pd.Series]:
    """
    Given a dataframe row (with visible_text, url, and heuristic fields),
    produce n_variants obfuscated rows.
    """
    if seed is not None:
        random.seed(seed)
    rows_out = []
    base_row = row.copy()
    for i in range(n_variants):
        r = base_row.copy()
        # Obfuscate text
        r["visible_text"] = obfuscate_text(r.get("visible_text", ""), methods=text_methods)
        # Obfuscate URL and heuristics with probability
        if random.random() < feat_obf_prob:
            new_url, new_feats = obfuscate_url_and_features(r.get("url", ""), r.to_dict())
            r["url"] = new_url
            # apply heuristics back into r
            for k, v in new_feats.items():
                if k in r.index:
                    r[k] = v
        rows_out.append(r)
    return rows_out

def synthesize_dataset(
    df: pd.DataFrame,
    label_col: str = "result",
    target_label: int = 1,
    synthesize_legit: bool = False,
    n_variants: int = 1,
    text_methods: Optional[List[str]] = None,
    feat_obf_prob: float = 0.7,
    seed: Optional[int] = None,
    row_sample: Optional[int] = None,
    keep_original_legit: bool = True,
    flip_label_on_synth: bool = False,
) -> pd.DataFrame:
    """
    Build new dataset by augmenting rows that match `target_label` with n_variants obfuscated copies.
    By default, target_label=1 (phishing). Legitimate rows are preserved unless synthesize_legit=True.
    - row_sample: if provided, sample this many rows (from full df) before filtering/synthesizing for speed.
    - keep_original_legit: if True (default) include untouched legitimate rows in returned dataset.
    - flip_label_on_synth: if True, flip the result label on synthesized rows (use carefully).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # optionally sample first for speed / reproducibility
    if row_sample is not None and row_sample < len(df):
        df_sampled = df.sample(row_sample, random_state=seed).reset_index(drop=True)
    else:
        df_sampled = df.reset_index(drop=True)

    # split target rows (e.g., phishing) and legit rows
    target_df = df_sampled[df_sampled[label_col] == target_label].reset_index(drop=True)
    legit_df = df_sampled[df_sampled[label_col] != target_label].reset_index(drop=True)

    out_rows = []
    # synthesize target rows only (unless synthesize_legit True, then we'll do both)
    def _synth_rows_from_df(dframe):
        res = []
        for idx, row in dframe.iterrows():
            new_rows = synthesize_row(row, text_methods=text_methods, n_variants=n_variants, feat_obf_prob=feat_obf_prob, seed=None)
            for r in new_rows:
                if flip_label_on_synth:
                    r[label_col] = 1 - int(r[label_col])
                res.append(r)
        return res

    # handle target (phishing) rows
    out_rows.extend(_synth_rows_from_df(target_df))

    # handle legitimate rows if requested
    if synthesize_legit:
        out_rows.extend(_synth_rows_from_df(legit_df))
        final_df = pd.DataFrame(out_rows).reset_index(drop=True)
    else:
        # keep synthesized phishing rows + optionally keep original legitimate rows unchanged
        final_df = pd.DataFrame(out_rows).reset_index(drop=True)
        if keep_original_legit and len(legit_df) > 0:
            final_df = pd.concat([final_df, legit_df], ignore_index=True).reset_index(drop=True)

    # ensure consistent dtypes where possible (user may need to cast for their vectorizer)
    return final_df

# ----------------------------
# Model scoring wrappers (use your model loaders)
# ----------------------------

def wrap_model_functions(
    textual_model_loader: Callable[[], Any],
    heuristic_vectorizer_loader: Callable[[], Any],
    heuristic_model_loader: Callable[[], Any],
    fusion_fn: Optional[Callable[[float, float], float]] = None
) -> Tuple[Callable[[str], float], Callable[[Dict], float], Callable[[float, float], float]]:
    """
    Accepts loader callables that return loaded models/objects.
    Returns: get_textual_score(text) -> float, get_heuristic_score(feat_dict) -> float, get_fusion_score(text_score, heur_score) -> float
    """

    # Load textual model
    textual_model = textual_model_loader()
    heuristic_vectorizer = heuristic_vectorizer_loader()
    heuristic_model = heuristic_model_loader()

    def get_textual_score_local(text: str) -> float:
        if textual_model is None:
            return 0.0
        text_clean = str(text).replace("\n", " ")
        # assume fasttext-like API
        try:
            labels, raw_probs = textual_model.predict(text_clean, k=2)
            label_probs = dict(zip(labels, raw_probs))
            return float(label_probs.get("__label__1", 0.0))
        except Exception:
            # fallback: if model has predict_proba or score method implement custom logic
            try:
                prob = float(textual_model.predict_proba([text_clean])[0][1])
                return prob
            except Exception:
                return 0.0

    def get_heuristic_score_local(feat_dict: Dict[str, Any]) -> float:
        if heuristic_vectorizer is None or heuristic_model is None:
            return 0.0
        # vectorizer.transform expects mapping or dataframe row
        try:
            vec = heuristic_vectorizer.transform([feat_dict])
            probs = heuristic_model.predict_proba(vec)
            return float(probs[0, 1])
        except Exception:
            # try direct predict_proba on raw features
            try:
                probs = heuristic_model.predict_proba(pd.DataFrame([feat_dict]).values)
                return float(probs[0, 1])
            except Exception:
                return 0.0

    if fusion_fn is None:
        def fusion_default(ts: float, hs: float) -> float:
            return 0.5 * ts + 0.5 * hs
        fusion_fn = fusion_default

    return get_textual_score_local, get_heuristic_score_local, fusion_fn

# ----------------------------
# Evaluation utilities
# ----------------------------

def _compute_metrics(y_true: np.ndarray, y_scores: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    lb = LabelBinarizer()
    y_true_bin = lb.fit_transform(y_true).ravel()
    y_pred = (y_scores >= threshold).astype(int)
    metrics = {}
    
    # standard metrics
    try:
        metrics["auc"] = float(roc_auc_score(y_true_bin, y_scores))
    except Exception:
        metrics["auc"] = float("nan")
    metrics["accuracy"] = float(accuracy_score(y_true_bin, y_pred))
    metrics["precision"] = float(precision_score(y_true_bin, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true_bin, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_true_bin, y_pred, zero_division=0))
    
    # confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred).ravel()
    
    # cybersecurity-specific metrics
    metrics["fpr"] = float(fp / (fp + tn)) if (fp + tn) > 0 else float("nan")
    metrics["fnr"] = float(fn / (fn + tp)) if (fn + tp) > 0 else float("nan")
    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    metrics["mcc"] = float(matthews_corrcoef(y_true_bin, y_pred))
    
    # PR-AUC
    precision_curve, recall_curve, _ = precision_recall_curve(y_true_bin, y_scores)
    metrics["pr_auc"] = float(auc(recall_curve, precision_curve))
    
    return metrics

def evaluate_ablation(
    df: pd.DataFrame,
    label_col: str,
    get_textual_score_fn: Callable[[str], float],
    get_heuristic_score_fn: Callable[[Dict], float],
    fusion_fn: Callable[[float, float], float],
    text_col: str = "visible_text",
    feats_columns: Optional[List[str]] = None,
    threshold: float = 0.5,
    sample_limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Evaluate three scoring methods on df:
      - text-only (score from textual model)
      - heuristic-only (score from heuristic model)
      - fusion (combined)
    Returns a dataframe with metrics for each method.
    """
    if sample_limit is not None and sample_limit < len(df):
        df_eval = df.sample(sample_limit, random_state=42).reset_index(drop=True)
    else:
        df_eval = df.reset_index(drop=True)

    y_true = df_eval[label_col].values
    text_scores = []
    heur_scores = []
    fusion_scores = []

    # determine feature columns if not provided
    if feats_columns is None:
        # assume heuristics are all columns except label, text, url
        exclude = {label_col, text_col, "url"}
        feats_columns = [c for c in df_eval.columns if c not in exclude]

    for idx, row in df_eval.iterrows():
        txt = row.get(text_col, "")
        feats = {k: row.get(k) for k in feats_columns}
        ts = get_textual_score_fn(txt)
        hs = get_heuristic_score_fn(feats)
        fs = fusion_fn(ts, hs)
        text_scores.append(ts)
        heur_scores.append(hs)
        fusion_scores.append(fs)

    # compute metrics
    res = []
    for name, scores in [("textual", np.array(text_scores)), ("heuristic", np.array(heur_scores)), ("fusion", np.array(fusion_scores))]:
        metrics = _compute_metrics(np.array(y_true), scores, threshold=threshold)
        metrics["method"] = name
        metrics["n_samples"] = len(y_true)
        res.append(metrics)

    res_df = pd.DataFrame(res).set_index("method")
    return res_df

# ----------------------------
# Orchestrator for ablation benchmark
# ----------------------------

def run_ablation_benchmark(
    df: pd.DataFrame,
    label_col: str = "result",
    text_col: str = "visible_text",
    n_variants: int = 2,
    synth_seed: Optional[int] = None,
    sample_limit: Optional[int] = 1000,
    target_label: int = 1,
    synthesize_legit: bool = False,
    keep_original_legit: bool = True,
    flip_label_on_synth: bool = False,
    textual_model_path: str = "../models/textual/fasttext/model.ftz",
    vectorizer_path: str = "../models/heuristic/xgboost/vectorizer.pkl",
    heuristic_model_path: str = "../models/heuristic/xgboost/model.pkl",
    fusion_fn: Optional[Callable[[float, float], float]] = None,
    save_csv: Optional[str] = None
) -> pd.DataFrame:
    """
    Full pipeline:
      - sample (optional)
      - synthesize augmented dataset (n_variants per target row, default target_label=1)
      - load models
      - evaluate ablation on original and synthesized datasets
    Returns a combined metrics DataFrame.
    """
    # 1) sample first for speed / reproducibility
    if sample_limit is not None and sample_limit < len(df):
        df_sampled = df.sample(min(sample_limit, len(df)), random_state=synth_seed).reset_index(drop=True)
    else:
        df_sampled = df.reset_index(drop=True)

    print("[INFO] Synthesizing dataset (target_label=%s)..." % str(target_label)
          + f" synthesize_legit={synthesize_legit}, keep_original_legit={keep_original_legit}")
    synth_df = synthesize_dataset(
        df_sampled,
        label_col=label_col,
        target_label=target_label,
        synthesize_legit=synthesize_legit,
        n_variants=n_variants,
        seed=synth_seed,
        keep_original_legit=keep_original_legit,
        flip_label_on_synth=flip_label_on_synth
    )
    print(f"[INFO] Sampled rows: {len(df_sampled)} | Synthesized rows: {len(synth_df)}")

    # 2) loaders (same as before)
    def textual_loader():
        if fasttext is None:
            print("[WARN] fasttext not available; textual scoring will return 0.0")
            return None
        try:
            return fasttext.load_model(textual_model_path)
        except Exception as e:
            print("[WARN] Failed to load fasttext model:", e)
            return None

    def vectorizer_loader():
        if joblib is None:
            print("[WARN] joblib not available; heuristic scoring will return 0.0")
            return None
        try:
            return joblib.load(vectorizer_path)
        except Exception as e:
            print("[WARN] Failed to load vectorizer:", e)
            return None

    def heuristic_loader():
        if joblib is None:
            print("[WARN] joblib not available; heuristic scoring will return 0.0")
            return None
        try:
            return joblib.load(heuristic_model_path)
        except Exception as e:
            print("[WARN] Failed to load heuristic model:", e)
            return None

    get_textual_score_fn, get_heuristic_score_fn, fusion_fn_local = wrap_model_functions(
        textual_loader, vectorizer_loader, heuristic_loader, fusion_fn=fusion_fn
    )

    # 3) Evaluate on original sampled data
    print("[INFO] Evaluating on original data...")
    metrics_orig = evaluate_ablation(df_sampled.reset_index(drop=True), label_col, get_textual_score_fn, get_heuristic_score_fn, fusion_fn_local, text_col=text_col)

    # 4) Evaluate on synthesized data
    print("[INFO] Evaluating on synthesized (obfuscated) data...")
    metrics_synth = evaluate_ablation(synth_df.reset_index(drop=True), label_col, get_textual_score_fn, get_heuristic_score_fn, fusion_fn_local, text_col=text_col)

    # 5) Combine and return
    metrics_orig["dataset"] = "original"
    metrics_synth["dataset"] = "synthesized"
    combined = pd.concat([metrics_orig, metrics_synth])
    if save_csv:
        combined.to_csv(save_csv)
        print(f"[INFO] Saved metrics to {save_csv}")
    return combined