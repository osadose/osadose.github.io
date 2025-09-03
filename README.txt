import pandas as pd

def load_isco(path):
    """
    Load ISCO catalog from Excel.
    Expects columns with 'Code' and 'Title'.
    """
    df = pd.read_excel(path)
    # normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Find best guesses for code and title columns
    code_col = next((c for c in df.columns if "code" in c), None)
    title_col = next((c for c in df.columns if "title" in c), None)

    if not code_col or not title_col:
        raise ValueError("ISCO file must contain 'code' and 'title' columns")

    return df[[code_col, title_col]].rename(
        columns={code_col: "code", title_col: "title"}
    )


def load_isic(path):
    """
    Load ISIC catalog from Excel.
    Expects columns with 'Code' and 'Title'.
    """
    df = pd.read_excel(path)
    # normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    code_col = next((c for c in df.columns if "code" in c), None)
    title_col = next((c for c in df.columns if "title" in c), None)

    if not code_col or not title_col:
        raise ValueError("ISIC file must contain 'code' and 'title' columns")

    return df[[code_col, title_col]].rename(
        columns={code_col: "code", title_col: "title"}
    )
