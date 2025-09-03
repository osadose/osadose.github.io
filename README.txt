import click
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer

from .loader import load_isco, load_isic

@click.command()
@click.option("--isco", required=True, help="Path to ISCO Excel file")
@click.option("--isic", required=True, help="Path to ISIC Excel file")
@click.option("--out", required=True, help="Output pickle file for embeddings")
def main(isco, isic, out):
    # Load ISCO & ISIC
    isco_df = load_isco(isco)
    isic_df = load_isic(isic)

    isco_titles = isco_df["title"].tolist()
    isic_titles = isic_df["title"].tolist()

    # Load model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Encode
    isco_emb = model.encode(isco_titles, convert_to_tensor=True)
    isic_emb = model.encode(isic_titles, convert_to_tensor=True)

    # Save to file
    with open(out, "wb") as f:
        pickle.dump(
            {
                "isco_titles": isco_titles,
                "isco_embeddings": isco_emb,
                "isic_titles": isic_titles,
                "isic_embeddings": isic_emb,
            },
            f,
        )

    print(f"[INFO] Embeddings saved to {out}")

if __name__ == "__main__":
    main()
