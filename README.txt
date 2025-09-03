"""
Semantic matcher using sentence-transformers.
- Precompute embeddings for a catalog (list of titles).
- Cache embeddings to disk for fast reuse.
- Provide best-match among catalog (cosine similarity).
"""

import os
import json
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

CACHE_DIR = "coder/data/embeddings_cache"  # adjust if you prefer

class SemanticMatcher:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None):
        """
        Instantiate model. device: 'cpu' or 'cuda' or None (auto).
        """
        kwargs = {}
        if device:
            kwargs["device"] = device
        self.model = SentenceTransformer(model_name, **kwargs)
        self.model_name = model_name
        os.makedirs(CACHE_DIR, exist_ok=True)

    def _cache_path(self, name: str):
        # name: a short identifier for the catalog e.g. "isco_08"
        safe = name.replace(" ", "_").lower()
        return os.path.join(CACHE_DIR, f"{safe}__{self.model_name.replace('/', '_')}.npz")

    def build_and_cache(self, catalog_titles: List[str], catalog_name: str):
        """Encode catalog titles and save embeddings + metadata to disk."""
        if not catalog_titles:
            raise ValueError("catalog_titles must be non-empty")
        cache_p = self._cache_path(catalog_name)
        # encode in batches (sentence-transformers handles batching)
        embeddings = self.model.encode(catalog_titles, convert_to_tensor=True, show_progress_bar=True)
        # convert to numpy for saving
        emb_np = embeddings.cpu().numpy()
        # save titles & embeddings
        np.savez_compressed(cache_p, titles=np.array(catalog_titles, dtype=object), embeddings=emb_np)
        return cache_p

    def load_cached(self, catalog_name: str):
        p = self._cache_path(catalog_name)
        if not os.path.exists(p):
            return None, None
        data = np.load(p, allow_pickle=True)
        titles = data["titles"].tolist()
        embeddings = data["embeddings"]
        return titles, embeddings

    def best_semantic(self, query: str, cached_titles: List[str], cached_embeddings: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Return top_k (index, similarity_score) pairs (score in 0..1) for query vs cached_embeddings.
        Uses cosine similarity via sentence-transformers util.cos_sim (works with numpy/torch).
        """
        if not query or not cached_titles or cached_embeddings is None:
            return []
        q_emb = self.model.encode([query], convert_to_tensor=True)
        # cached_embeddings -> convert to tensor inside util.cos_sim
        scores = util.cos_sim(q_emb, cached_embeddings)[0]  # shape (n,)
        # get top_k indices
        values, indices = scores.topk(k=min(top_k, scores.shape[0]))
        results = []
        for v, idx in zip(values, indices):
            results.append((int(idx.cpu().item()), float(v.cpu().item())))
        return results




"""
Utility CLI to precompute embeddings for ISCO/ISIC catalogs.
Run this once (or whenever catalogs change) to cache embeddings for faster processing.
"""

import argparse
import os
from coder.loader import load_isco_catalog, load_isic_catalog
from coder.ml_matcher import SemanticMatcher

def main():
    parser = argparse.ArgumentParser(description="Precompute catalog embeddings")
    parser.add_argument("--isco", required=False, help="Path to ISCO workbook")
    parser.add_argument("--isic", required=False, help="Path to ISIC workbook")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model")
    args = parser.parse_args()

    sm = SemanticMatcher(model_name=args.model)

    if args.isco:
        print("Loading ISCO catalog...")
        isco_catalog = load_isco_catalog(args.isco)
        titles = [e.title for e in isco_catalog.entries]
        print(f"Encoding {len(titles)} ISCO titles...")
        p = sm.build_and_cache(titles, "isco_catalog")
        print("Cached ISCO embeddings to:", p)

    if args.isic:
        print("Loading ISIC catalog...")
        isic_catalog = load_isic_catalog(args.isic)
        titles = [e.title for e in isic_catalog.entries]
        print(f"Encoding {len(titles)} ISIC titles...")
        p = sm.build_and_cache(titles, "isic_catalog")
        print("Cached ISIC embeddings to:", p)

if __name__ == "__main__":
    main()



import os
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from rapidfuzz import process, fuzz

from coder.loader import load_isco_catalog, load_isic_catalog
from coder.isco import match_isco as fallback_match_isco
from coder.isic import match_isic as fallback_match_isic
from coder.ml_matcher import SemanticMatcher

# default config keys used:
# matching:
#   use_semantic: true
#   fuzz_weight: 0.5
#   embed_weight: 0.5
#   top_k_candidates: 10

def load_config(path=None):
    if path and os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    # fallback packaged config
    pkg_cfg = os.path.join(os.path.dirname(__file__), "config_default.yaml")
    with open(pkg_cfg, "r") as f:
        return yaml.safe_load(f)

def _rapidfuzz_candidates(query: str, catalog_titles: List[str], top_k: int = 10):
    # return list of (title, score(0..1), index_in_catalog_titles)
    if not query or not catalog_titles:
        return []
    res = process.extract(query, catalog_titles, scorer=fuzz.WRatio, limit=top_k)
    out = []
    for matched, score, pos in res:
        out.append((matched, float(score) / 100.0, pos))
    return out

def hybrid_best_match(query: str,
                      catalog_titles: List[str],
                      cached_embeddings: np.ndarray,
                      sem_matcher: SemanticMatcher,
                      fuzz_top_k: int = 20,
                      sem_top_k: int = 5,
                      fuzz_weight: float = 0.5,
                      embed_weight: float = 0.5):
    """
    Use RapidFuzz to get candidate set, then rescore using embeddings.
    Returns (best_title, combined_score, best_index)
    """
    if not query:
        return None, 0.0, None

    # 1) RapidFuzz candidate generation
    rf_cands = _rapidfuzz_candidates(query, catalog_titles, top_k=fuzz_top_k)
    if not rf_cands:
        return None, 0.0, None

    # candidate indices from RF
    cand_idxs = [c[2] for c in rf_cands]

    # 2) Get semantic scores for the same candidates
    # If cached_embeddings provided, use them
    sem_scores = {}
    if cached_embeddings is not None:
        # compute cosine sim of query against all cached embeddings (fast to vectorize),
        # but we only need the cand indices; use semantic matcher best_semantic on limited set
        # We will use sem_matcher.best_semantic which expects full cache; to avoid re-encoding the
        # whole catalog, use its model to encode query and compute dot with cached_embeddings.
        q_emb = sem_matcher.model.encode([query], convert_to_tensor=True)
        # cached_embeddings may be numpy; ensure torch tensor handled by util.cos_sim inside matcher
        from sentence_transformers import util
        # convert cached_embeddings to tensor via util if needed
        # compute sim vector
        sims = util.cos_sim(q_emb, cached_embeddings)[0]
        # extract values at candidate indices
        for idx in cand_idxs:
            sem_scores[idx] = float(sims[idx].cpu().item())
    else:
        # If no cached embeddings, try semantic matcher best_semantic limited to candidates
        sem_results = sem_matcher.best_semantic(query, catalog_titles, None) if hasattr(sem_matcher, 'best_semantic') else []
        # fallback: no semantic info
        sem_scores = {idx: 0.0 for idx in cand_idxs}

    # 3) Combine scores for candidates and pick best
    best_idx = None
    best_combined = -1.0
    for matched_title, rf_score, idx in rf_cands:
        sem_score = sem_scores.get(idx, 0.0)
        # sem_score may be in [-1,1] if not normalized; ensure 0..1
        sem_score_norm = max(0.0, min(1.0, sem_score))
        combined = fuzz_weight * rf_score + embed_weight * sem_score_norm
        if combined > best_combined:
            best_combined = combined
            best_idx = idx

    best_title = catalog_titles[best_idx] if best_idx is not None else None
    return best_title, float(best_combined), best_idx

def run_pipeline(input_csv: str,
                 output_csv: str,
                 isco_xlsx: str,
                 isic_xlsx: str,
                 config_path: str = None):
    cfg = load_config(config_path)
    general = cfg.get("general", {})
    review_th = float(general.get("review_threshold", 0.65))
    max_cand = int(general.get("max_candidates", 200))

    matching_cfg = cfg.get("matching", {}) or {}
    use_semantic = bool(matching_cfg.get("use_semantic", True))
    fuzz_weight = float(matching_cfg.get("fuzz_weight", 0.5))
    embed_weight = float(matching_cfg.get("embed_weight", 0.5))
    fuzz_top_k = int(matching_cfg.get("fuzz_top_k", 20))
    sem_top_k = int(matching_cfg.get("sem_top_k", 5))

    # load synonyms from cfg location
    base_dir = os.path.dirname(config_path) if config_path else os.path.dirname(__file__)
    syn_file = cfg.get("synonyms_file", os.path.join(base_dir, "lexicon_synonyms.csv"))
    synonyms = {}
    if syn_file and os.path.exists(syn_file):
        try:
            syn_df = pd.read_csv(syn_file)
            colnames = [c.lower() for c in syn_df.columns]
            if "from" in colnames and "to" in colnames:
                synonyms = dict(zip(syn_df[syn_df.columns[colnames.index("from")]].astype(str),
                                    syn_df[syn_df.columns[colnames.index("to")]].astype(str)))
            elif "source" in colnames and "target" in colnames:
                synonyms = dict(zip(syn_df["source"].astype(str), syn_df["target"].astype(str)))
            else:
                synonyms = dict(zip(syn_df.iloc[:,0].astype(str), syn_df.iloc[:,1].astype(str)))
        except Exception:
            synonyms = {}

    # load catalogs
    isco_catalog = load_isco_catalog(isco_xlsx)
    isic_catalog = load_isic_catalog(isic_xlsx)

    isco_titles = [t for t in (e.title for e in isco_catalog.entries)]
    isic_titles = [t for t in (e.title for e in isic_catalog.entries)]

    # semantic matcher + cached embeddings
    sem_matcher = None
    isco_cached_emb = None
    isic_cached_emb = None
    if use_semantic:
        sem_matcher = SemanticMatcher()
        # try load cached embed files (consistent names used by precompute_embeddings)
        isco_titles_cached, isco_embed = sem_matcher.load_cached("isco_catalog")
        if isco_titles_cached is not None:
            # ensure same order; assume precompute used same catalog order
            isco_cached_emb = np.asarray(isco_embed)
            # If titles differ, fallback to on-the-fly build (slower)
            if len(isco_titles_cached) != len(isco_titles) or any(a != b for a,b in zip(isco_titles_cached, isco_titles)):
                # rebuild cache with current titles
                sem_matcher.build_and_cache(isco_titles, "isco_catalog")
                _, isco_embed = sem_matcher.load_cached("isco_catalog")
                isco_cached_emb = np.asarray(isco_embed)
        else:
            # no cache -> build for current catalog
            print("Building ISCO embeddings (this may take a minute)...")
            sem_matcher.build_and_cache(isco_titles, "isco_catalog")
            _, isco_embed = sem_matcher.load_cached("isco_catalog")
            isco_cached_emb = np.asarray(isco_embed)

        isic_titles_cached, isic_embed = sem_matcher.load_cached("isic_catalog")
        if isic_titles_cached is not None:
            if len(isic_titles_cached) != len(isic_titles) or any(a != b for a,b in zip(isic_titles_cached, isic_titles)):
                sem_matcher.build_and_cache(isic_titles, "isic_catalog")
                _, isic_embed = sem_matcher.load_cached("isic_catalog")
                isic_cached_emb = np.asarray(isic_embed)
            else:
                isic_cached_emb = np.asarray(isic_embed)
        else:
            print("Building ISIC embeddings (this may take a minute)...")
            sem_matcher.build_and_cache(isic_titles, "isic_catalog")
            _, isic_embed = sem_matcher.load_cached("isic_catalog")
            isic_cached_emb = np.asarray(isic_embed)

    # load input data
    df = pd.read_csv(input_csv, dtype=str).fillna("")

    # prepare output columns
    df["Clean_ISCO_Code"] = ""
    df["ISCO_Title"] = ""
    df["ISCO_Confidence"] = 0.0
    df["Clean_ISIC_Code"] = ""
    df["ISIC_Title"] = ""
    df["ISIC_Confidence"] = 0.0
    if "Enumerator_ISCO" in df.columns:
        df["ISCO_Agreement"] = ""
    if "Enumerator_ISIC" in df.columns:
        df["ISIC_Agreement"] = ""

    review_rows = []

    # run row-by-row hybrid matching
    for i, row in df.iterrows():
        # combine title + description for ISCO matching
        query_isco = " ".join([str(row.get(c, "")) for c in cfg.get("fields", {}).get("title_fields", ["Job_Title"]) + cfg.get("fields", {}).get("description_fields", ["Job_Description"])])
        # apply synonyms (simple replacement)
        for src, dst in synonyms.items():
            if not src:
                continue
            query_isco = query_isco.replace(src, dst)

        # rapidfuzz + semantic hybrid for ISCO
        best_title_isco, combined_isco_score, best_idx = hybrid_best_match(
            query=query_isco,
            catalog_titles=isco_titles,
            cached_embeddings=isco_cached_emb,
            sem_matcher=sem_matcher,
            fuzz_top_k=fuzz_top_k,
            sem_top_k=sem_top_k,
            fuzz_weight=fuzz_weight,
            embed_weight=embed_weight,
        )
        if best_idx is not None:
            # find entry and code
            entry = isco_catalog.entries[best_idx]
            df.at[i, "Clean_ISCO_Code"] = str(entry.code)
            df.at[i, "ISCO_Title"] = entry.title
            df.at[i, "ISCO_Confidence"] = round(float(combined_isco_score), 3)
        else:
            # fallback to simple matcher from earlier (safe)
            entry, score = fallback_match_isco(row, isco_catalog, cfg.get("fields", {}).get("title_fields", ["Job_Title"]), cfg.get("fields", {}).get("description_fields", ["Job_Description"]), synonyms, max_cand)
            if entry:
                df.at[i, "Clean_ISCO_Code"] = str(entry.code)
                df.at[i, "ISCO_Title"] = entry.title
                df.at[i, "ISCO_Confidence"] = round(float(score), 3)

        # ISIC matching
        query_isic = " ".join([str(row.get(c, "")) for c in cfg.get("fields", {}).get("sector_fields", ["Job_Sector"]) + cfg.get("fields", {}).get("description_fields", ["Job_Description"])])
        for src, dst in synonyms.items():
            if not src:
                continue
            query_isic = query_isic.replace(src, dst)

        best_title_isic, combined_isic_score, best_idx_isic = hybrid_best_match(
            query=query_isic,
            catalog_titles=isic_titles,
            cached_embeddings=isic_cached_emb,
            sem_matcher=sem_matcher,
            fuzz_top_k=fuzz_top_k,
            sem_top_k=sem_top_k,
            fuzz_weight=fuzz_weight,
            embed_weight=embed_weight,
        )
        if best_idx_isic is not None:
            entry2 = isic_catalog.entries[best_idx_isic]
            df.at[i, "Clean_ISIC_Code"] = str(entry2.code)
            df.at[i, "ISIC_Title"] = entry2.title
            df.at[i, "ISIC_Confidence"] = round(float(combined_isic_score), 3)
        else:
            entry2, score2 = fallback_match_isic(row, isic_catalog, cfg.get("fields", {}).get("sector_fields", ["Job_Sector"]), cfg.get("fields", {}).get("description_fields", ["Job_Description"]), synonyms, max_cand)
            if entry2:
                df.at[i, "Clean_ISIC_Code"] = str(entry2.code)
                df.at[i, "ISIC_Title"] = entry2.title
                df.at[i, "ISIC_Confidence"] = round(float(score2), 3)

        # agreement flags
        if "Enumerator_ISCO" in df.columns:
            ecode = str(row.get("Enumerator_ISCO", "")).strip()
            if ecode and df.at[i, "Clean_ISCO_Code"]:
                df.at[i, "ISCO_Agreement"] = "AGREE" if ecode == str(df.at[i, "Clean_ISCO_Code"]) else "DISAGREE"
            elif ecode and not df.at[i, "Clean_ISCO_Code"]:
                df.at[i, "ISCO_Agreement"] = "ENUM_ONLY"
            elif not ecode and df.at[i, "Clean_ISCO_Code"]:
                df.at[i, "ISCO_Agreement"] = "AUTO_ONLY"
            else:
                df.at[i, "ISCO_Agreement"] = "MISSING"

        if "Enumerator_ISIC" in df.columns:
            ecode = str(row.get("Enumerator_ISIC", "")).strip()
            if ecode and df.at[i, "Clean_ISIC_Code"]:
                df.at[i, "ISIC_Agreement"] = "AGREE" if ecode == str(df.at[i, "Clean_ISIC_Code"]) else "DISAGREE"
            elif ecode and not df.at[i, "Clean_ISIC_Code"]:
                df.at[i, "ISIC_Agreement"] = "ENUM_ONLY"
            elif not ecode and df.at[i, "Clean_ISIC_Code"]:
                df.at[i, "ISIC_Agreement"] = "AUTO_ONLY"
            else:
                df.at[i, "ISIC_Agreement"] = "MISSING"

        # review queue
        try:
            if (float(df.at[i, "ISCO_Confidence"]) < review_th) or (float(df.at[i, "ISIC_Confidence"]) < review_th):
                review_rows.append(i)
        except Exception:
            review_rows.append(i)

    # save outputs
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False)

    review_path = None
    if review_rows:
        review_df = df.loc[review_rows].copy()
        review_path = output_csv.replace(".csv", ".review_queue.csv")
        review_df.to_csv(review_path, index=False)

    return df, review_path




general:
  review_threshold: 0.65
  max_candidates: 200

fields:
  title_fields: ["Job_Title"]
  description_fields: ["Job_Description"]
  sector_fields: ["Job_Sector"]

synonyms_file: "coder/data/synonyms.csv"

matching:
  use_semantic: true
  fuzz_weight: 0.5
  embed_weight: 0.5
  fuzz_top_k: 20
  sem_top_k: 5


pip install -r requirements.txt



python coder/precompute_embeddings.py --isco "coder/data/ISCO-08 EN Structure and definitions.xlsx" --isic "coder/data/ISIC5_Exp_Notes_11Mar2024.xlsx"




























