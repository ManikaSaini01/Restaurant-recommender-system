import pandas as pd
import re

MAX_REVIEW_CHARS = 250  # strict memory limit


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def clean_name(name: str) -> str:
    name = normalize_text(name)

    blacklist = {"rated", "new", "na", "null", "none", "-", ""}
    if name in blacklist or len(name) < 3:
        return ""

    return name


def clean_review_fast(text) -> str:
    """
    MEMORY-SAFE review cleaning:
    1. Convert to string
    2. TRIM FIRST
    3. Then clean
    """
    text = str(text)[:MAX_REVIEW_CHARS]
    return normalize_text(text)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # Rename columns
    df = df.rename(columns={
        "rate": "rating",
        "reviews_list": "reviews"
    })

    df = df[["name", "cuisines", "reviews", "rating"]]

    # Clean ratings
    df["rating"] = (
        df["rating"]
        .astype(str)
        .str.replace("/5", "", regex=False)
    )
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])

    # Clean names
    df["canonical_name"] = df["name"].apply(clean_name)
    df = df[df["canonical_name"] != ""]

    # Clean cuisines (short text)
    df["clean_cuisines"] = (
        df["cuisines"]
        .astype(str)
        .str.lower()
        .str.replace(r'[^a-zA-Z\s]', ' ', regex=True)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )

    # 🚨 CLEAN REVIEWS SAFELY
    df["clean_reviews"] = df["reviews"].map(clean_review_fast)

    # Deduplicate by canonical name (keep best rating)
    df = (
        df.sort_values("rating", ascending=False)
        .groupby("canonical_name", as_index=False)
        .first()
    )

    # Display-friendly name
    df["display_name"] = df["canonical_name"].str.title()

    df = df.reset_index(drop=True)

    return df
