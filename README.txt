# Default configuration for NBS code cleaner
general:
  # auto_accept_threshold: not used for forced accept in v0.1; emitted confidence only
  auto_accept_threshold: 0.85
  # below this value a row goes into the review queue
  review_threshold: 0.65
  max_candidates: 200

fields:
  title_fields:
    - Job_Title
    - Occupation
  description_fields:
    - Job_Description
    - Duties
    - Tasks
  sector_fields:
    - Job_Sector
    - Industry
    - Employer_Sector

output:
  include_titles: true
  include_descriptions: false
  add_agreement_flags: true

# synonyms file path (relative to this package when config_path is not provided)
synonyms_file: "lexicon_synonyms.csv"



from,to
okada rider,motorcycle driver
okada,motorcycle taxi
keke napep,tricycle taxi
keke,tricycle taxi
conductor,bus conductor
sales girl,shop sales assistant
sales boy,shop sales assistant
salesman,shop sales assistant
tailor,dressmaker
barbing,barber
secondary school,secondary education
primary school,primary education
nursery,early childhood
teacher,teachers
driver,drivers
taxi cab,taxi



pandas>=1.3
openpyxl
pyyaml
rapidfuzz>=2.0



import re
import unicodedata
from typing import List, Set

_punct_re = re.compile(r"[^0-9a-zA-Z\s]+")
_ws_re = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Normalize text to ASCII lowercase, remove punctuation, collapse whitespace."""
    if text is None:
        return ""
    t = str(text)
    # normalize unicode -> ascii where possible
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    t = t.lower()
    t = _punct_re.sub(" ", t)
    t = _ws_re.sub(" ", t).strip()
    return t


def tokens(text: str) -> List[str]:
    t = normalize(text)
    return t.split() if t else []


def token_set(text: str) -> Set[str]:
    return set(tokens(text))


def token_overlap_score(a: str, b: str) -> float:
    A = token_set(a)
    B = token_set(b)
    if not A or not B:
        return 0.0
    inter = len(A & B)
    denom = max(len(A), len(B))
    return inter / denom


def simple_ratio(a: str, b: str) -> float:
    """
    Lightweight similarity metric that returns 0..1.
    Uses token overlap and length ratio to approximate similarity.
    """
    a_n = normalize(a)
    b_n = normalize(b)
    if not a_n and not b_n:
        return 1.0
    if not a_n or not b_n:
        return 0.0
    overlap = token_overlap_score(a_n, b_n)
    # length ratio on normalized strings
    len_ratio = min(len(a_n), len(b_n)) / max(len(a_n), len(b_n))
    return 0.6 * overlap + 0.4 * len_ratio



from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from .text import normalize, simple_ratio

@dataclass
class CatalogEntry:
    code: str
    title: str
    description: str = ""


class Catalog:
    """
    Catalog of code entries (ISCO or ISIC). Provides fast exact title lookup,
    inverted token index for candidate selection, and a best_match scorer.
    """

    def __init__(self, entries: List[CatalogEntry]):
        self.entries = entries
        # normalized forms
        self.norm_titles = [normalize(e.title) for e in entries]
        self.norm_descs = [normalize(e.description) for e in entries]
        self.codes = [e.code for e in entries]

        # exact normalized title -> index
        self.title_index: Dict[str, int] = {}
        for i, t in enumerate(self.norm_titles):
            if t:
                self.title_index[t] = i

        # inverted index: token -> set(indices)
        self.token_index: Dict[str, set] = {}
        for i, t in enumerate(self.norm_titles):
            for tok in set(t.split()):
                if tok:
                    self.token_index.setdefault(tok, set()).add(i)

    def get_by_title_norm(self, title_norm: str) -> Optional[CatalogEntry]:
        i = self.title_index.get(title_norm)
        return self.entries[i] if i is not None else None

    def candidates(self, query: str, max_candidates: int = 200) -> List[int]:
        q_tokens = set(normalize(query).split())
        idxs = set()
        for tok in q_tokens:
            if tok in self.token_index:
                idxs |= self.token_index[tok]
        if not idxs:
            idxs = set(range(len(self.entries)))
        # return up to max_candidates indexes
        return list(list(idxs)[:max_candidates])

    def best_match(self, query: str, max_candidates: int = 200) -> Tuple[Optional[CatalogEntry], float]:
        if not query or not query.strip():
            return None, 0.0
        qn = normalize(query)

        # 1) exact normalized title
        if qn in self.title_index:
            return self.entries[self.title_index[qn]], 1.0

        # 2) containment heuristic (title within query or query within title)
        best_i = None
        best_s = 0.0
        for i, t in enumerate(self.norm_titles):
            if t and (t in qn or qn in t):
                s = min(len(qn), len(t)) / max(len(qn), len(t))
                if s > best_s:
                    best_s, best_i = s, i

        # 3) token candidate scoring
        cands = self.candidates(query, max_candidates=max_candidates)
        for i in cands:
            s_title = simple_ratio(query, self.norm_titles[i])
            s_desc = simple_ratio(query, self.norm_descs[i]) if self.norm_descs[i] else 0.0
            s = max(s_title, 0.7 * s_title + 0.3 * s_desc)
            if s > best_s:
                best_s, best_i = s, i

        if best_i is None:
            return None, 0.0
        return self.entries[best_i], float(best_s)



import pandas as pd
from typing import List
from .matcher import Catalog, CatalogEntry


def load_isco_catalog(isco_xlsx_path: str) -> Catalog:
    """
    Load ISCO entries from ISCO-08 Excel.
    Expects sheet 'ISCO-08 EN Struct and defin' with columns including:
      - 'ISCO 08 Code'
      - 'Title EN'
      - 'Definition', 'Tasks include', 'Includes also', 'Excludes' (optional)
    """
    df = pd.read_excel(isco_xlsx_path, sheet_name="ISCO-08 EN Struct and defin")
    # Keep rows with code
    if "ISCO 08 Code" not in df.columns or "Title EN" not in df.columns:
        raise ValueError("ISCO sheet missing expected columns 'ISCO 08 Code' or 'Title EN'")
    df = df[df["ISCO 08 Code"].notna()].copy()
    df["ISCO 08 Code"] = df["ISCO 08 Code"].astype(str).str.strip()
    df["Title EN"] = df["Title EN"].astype(str)
    desc_cols = [c for c in ["Definition", "Tasks include", "Includes also", "Excludes"] if c in df.columns]
    if desc_cols:
        df["desc"] = df[desc_cols].fillna("").astype(str).agg(" ".join, axis=1)
    else:
        df["desc"] = ""

    entries: List[CatalogEntry] = []
    for _, r in df.iterrows():
        entries.append(CatalogEntry(code=r["ISCO 08 Code"], title=r["Title EN"], description=r["desc"]))
    return Catalog(entries)


def load_isic_catalog(isic_xlsx_path: str) -> Catalog:
    """
    Load ISIC entries from 'ISIC5 Notes' sheet (expected).
    Expected columns like:
      - 'ISIC Rev 5 NumCode'
      - 'ISIC Rev 5 Title'
      - additional note columns (Includes/Excludes/Introductory) which will be concatenated
    """
    df = pd.read_excel(isic_xlsx_path, sheet_name="ISIC5 Notes")
    if "ISIC Rev 5 NumCode" not in df.columns or "ISIC Rev 5 Title" not in df.columns:
        raise ValueError("ISIC sheet missing expected columns 'ISIC Rev 5 NumCode' or 'ISIC Rev 5 Title'")
    df["ISIC Rev 5 NumCode"] = df["ISIC Rev 5 NumCode"].astype(str).str.strip()
    df["ISIC Rev 5 Title"] = df["ISIC Rev 5 Title"].astype(str)
    note_cols = [c for c in df.columns if any(k in c for k in ["Introductory", "Includes", "Excludes", "Notes"])]
    if note_cols:
        df["desc"] = df[note_cols].fillna("").astype(str).agg(" ".join, axis=1)
    else:
        df["desc"] = ""

    entries: List[CatalogEntry] = []
    for _, r in df.iterrows():
        entries.append(CatalogEntry(code=r["ISIC Rev 5 NumCode"], title=r["ISIC Rev 5 Title"], description=r["desc"]))
    return Catalog(entries)



import pandas as pd
from typing import Tuple, Optional
from .matcher import Catalog, CatalogEntry
from .text import normalize


def build_query(row: pd.Series, title_fields, description_fields, synonyms: dict) -> str:
    parts = []
    for f in title_fields + description_fields:
        if f in row and pd.notna(row[f]):
            parts.append(str(row[f]))
    q = " ".join(parts)
    qn = normalize(q)
    # apply synonyms
    for src, dst in synonyms.items():
        qn = qn.replace(normalize(src), normalize(dst))
    return qn


def match_isco(row: pd.Series, catalog: Catalog, title_fields, description_fields, synonyms: dict, max_candidates: int) -> Tuple[Optional[CatalogEntry], float]:
    # Try exact normalized title using title fields with synonyms applied
    for f in title_fields:
        if f in row and pd.notna(row[f]):
            raw = str(row[f])
            tn = normalize(raw)
            for src, dst in synonyms.items():
                tn = tn.replace(normalize(src), normalize(dst))
            exact = catalog.get_by_title_norm(tn) if hasattr(catalog, "get_by_title_norm") else None
            if exact:
                return exact, 1.0

    # Else, fuzzy on the combined fields
    query = build_query(row, title_fields, description_fields, synonyms)
    entry, score = catalog.best_match(query, max_candidates=max_candidates)
    return entry, float(score)



import pandas as pd
from typing import Tuple, Optional
from .matcher import Catalog, CatalogEntry
from .text import normalize


def build_query(row: pd.Series, sector_fields, description_fields, synonyms: dict) -> str:
    parts = []
    for f in sector_fields + description_fields:
        if f in row and pd.notna(row[f]):
            parts.append(str(row[f]))
    q = " ".join(parts)
    qn = normalize(q)
    for src, dst in synonyms.items():
        qn = qn.replace(normalize(src), normalize(dst))
    return qn


def match_isic(row: pd.Series, catalog: Catalog, sector_fields, description_fields, synonyms: dict, max_candidates: int) -> Tuple[Optional[CatalogEntry], float]:
    # Try exact normalized title using sector field values
    for f in sector_fields:
        if f in row and pd.notna(row[f]):
            raw = str(row[f])
            tn = normalize(raw)
            for src, dst in synonyms.items():
                tn = tn.replace(normalize(src), normalize(dst))
            exact = catalog.get_by_title_norm(tn) if hasattr(catalog, "get_by_title_norm") else None
            if exact:
                return exact, 1.0

    # Else, fuzzy on sector + description
    query = build_query(row, sector_fields, description_fields, synonyms)
    entry, score = catalog.best_match(query, max_candidates=max_candidates)
    return entry, float(score)



import os
import pandas as pd
import yaml
from typing import Dict, Tuple
from .loader import load_isco_catalog, load_isic_catalog
from .isco import match_isco
from .isic import match_isic


def load_synonyms(csv_path: str) -> Dict[str, str]:
    try:
        df = pd.read_csv(csv_path)
        df = df.dropna()
        mapping = dict(zip(df["from"].astype(str), df["to"].astype(str)))
        return mapping
    except Exception:
        return {}


def run_pipeline(input_csv: str,
                 output_csv: str,
                 isco_xlsx: str,
                 isic_xlsx: str,
                 config_path: str = None) -> Tuple[pd.DataFrame, str]:
    # Load config
    if config_path:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        base_dir = os.path.dirname(os.path.abspath(config_path))
    else:
        pkg_dir = os.path.dirname(__file__)
        with open(os.path.join(pkg_dir, "config_default.yaml"), "r") as f:
            cfg = yaml.safe_load(f)
        base_dir = pkg_dir

    general = cfg.get("general", {})
    review_th = float(general.get("review_threshold", 0.65))
    max_cand = int(general.get("max_candidates", 200))

    fields = cfg.get("fields", {})
    title_fields = fields.get("title_fields", ["Job_Title"])
    desc_fields = fields.get("description_fields", ["Job_Description"])
    sector_fields = fields.get("sector_fields", ["Job_Sector"])

    output_cfg = cfg.get("output", {})
    include_titles = bool(output_cfg.get("include_titles", True))
    add_agreement = bool(output_cfg.get("add_agreement_flags", True))

    synonyms_file = cfg.get("synonyms_file", None)
    if synonyms_file:
        syn_path = synonyms_file if os.path.isabs(synonyms_file) else os.path.join(base_dir, synonyms_file)
        synonyms = load_synonyms(syn_path)
    else:
        synonyms = {}

    # Load catalogs
    isco_catalog = load_isco_catalog(isco_xlsx)
    isic_catalog = load_isic_catalog(isic_xlsx)

    # Load enumerator data
    df = pd.read_csv(input_csv, dtype=object).fillna(value=pd.NA)

    # Prepare output columns
    df["Clean_ISCO_Code"] = ""
    if include_titles:
        df["ISCO_Title"] = ""
    df["ISCO_Confidence"] = 0.0

    df["Clean_ISIC_Code"] = ""
    if include_titles:
        df["ISIC_Title"] = ""
    df["ISIC_Confidence"] = 0.0

    if add_agreement:
        if "Enumerator_ISCO" in df.columns:
            df["ISCO_Agreement"] = ""
        if "Enumerator_ISIC" in df.columns:
            df["ISIC_Agreement"] = ""

    review_rows = []

    total = len(df)
    for i, row in df.iterrows():
        # ISCO
        entry_isco, score_isco = match_isco(row, isco_catalog, title_fields, desc_fields, synonyms, max_cand)
        if entry_isco:
            df.at[i, "Clean_ISCO_Code"] = entry_isco.code
            if include_titles:
                df.at[i, "ISCO_Title"] = entry_isco.title
            df.at[i, "ISCO_Confidence"] = round(float(score_isco), 3)
        else:
            df.at[i, "Clean_ISCO_Code"] = ""
            df.at[i, "ISCO_Confidence"] = 0.0

        # ISIC
        entry_isic, score_isic = match_isic(row, isic_catalog, sector_fields, desc_fields, synonyms, max_cand)
        if entry_isic:
            df.at[i, "Clean_ISIC_Code"] = entry_isic.code
            if include_titles:
                df.at[i, "ISIC_Title"] = entry_isic.title
            df.at[i, "ISIC_Confidence"] = round(float(score_isic), 3)
        else:
            df.at[i, "Clean_ISIC_Code"] = ""
            df.at[i, "ISIC_Confidence"] = 0.0

        # Agreement flags (optional)
        if add_agreement and "Enumerator_ISCO" in df.columns:
            enum_code = str(row.get("Enumerator_ISCO")) if pd.notna(row.get("Enumerator_ISCO")) else ""
            if enum_code and entry_isco:
                df.at[i, "ISCO_Agreement"] = "AGREE" if str(enum_code).strip() == str(entry_isco.code).strip() else "DISAGREE"
            elif enum_code and not entry_isco:
                df.at[i, "ISCO_Agreement"] = "ENUM_ONLY"
            elif not enum_code and entry_isco:
                df.at[i, "ISCO_Agreement"] = "AUTO_ONLY"
            else:
                df.at[i, "ISCO_Agreement"] = "MISSING"

        if add_agreement and "Enumerator_ISIC" in df.columns:
            enum_code = str(row.get("Enumerator_ISIC")) if pd.notna(row.get("Enumerator_ISIC")) else ""
            if enum_code and entry_isic:
                df.at[i, "ISIC_Agreement"] = "AGREE" if str(enum_code).strip() == str(entry_isic.code).strip() else "DISAGREE"
            elif enum_code and not entry_isic:
                df.at[i, "ISIC_Agreement"] = "ENUM_ONLY"
            elif not enum_code and entry_isic:
                df.at[i, "ISIC_Agreement"] = "AUTO_ONLY"
            else:
                df.at[i, "ISIC_Agreement"] = "MISSING"

        # Add to review queue if either confidence below threshold
        if (float(df.at[i, "ISCO_Confidence"]) < review_th) or (float(df.at[i, "ISIC_Confidence"]) < review_th):
            review_rows.append(i)

        # Optional tiny progress print every 5000 rows (safe for 50k)
        if (i + 1) % 5000 == 0:
            print(f"Processed {i+1}/{total} rows...")

    # Save outputs
    df.to_csv(output_csv, index=False)

    review_path = None
    if review_rows:
        review_df = df.loc[review_rows].copy()
        review_path = output_csv.replace(".csv", ".review_queue.csv")
        review_df.to_csv(review_path, index=False)

    return df, review_path



import argparse
from .pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="NBS ISCO & ISIC Cleaning Tool")
    parser.add_argument("--input", required=True, help="Path to input CSV from enumerators")
    parser.add_argument("--output", required=True, help="Path to output cleaned CSV")
    parser.add_argument("--isco", required=True, help="Path to ISCO Excel file")
    parser.add_argument("--isic", required=True, help="Path to ISIC Excel file")
    parser.add_argument("--config", default=None, help="Optional YAML config path")
    args = parser.parse_args()

    df, review_path = run_pipeline(
        input_csv=args.input,
        output_csv=args.output,
        isco_xlsx=args.isco,
        isic_xlsx=args.isic,
        config_path=args.config
    )

    print(f"Saved cleaned file to: {args.output}")
    if review_path:
        print(f"Saved review queue to: {review_path}")


if __name__ == "__main__":
    main()




# Simple demo runner - edit paths as needed
from nbs_code_cleaner.pipeline import run_pipeline

input_csv = "sample_input.csv"  # create a small sample csv locally
output_csv = "cleaned_output.csv"
isco_xlsx = "ISCO-08 EN Structure and definitions.xlsx"
isic_xlsx = "ISIC Rev. 5 Explanatory Notes.xlsx"

df, review = run_pipeline(
    input_csv=input_csv,
    output_csv=output_csv,
    isco_xlsx=isco_xlsx,
    isic_xlsx=isic_xlsx,
    config_path=None
)

print("Wrote:", output_csv)
if review:
    print("Review:", review)




# NBS ISCO & ISIC Cleaning Tool (Prototype)

Simple CLI tool to auto-clean ISCO and ISIC codes using official Excel references.

## Install

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
