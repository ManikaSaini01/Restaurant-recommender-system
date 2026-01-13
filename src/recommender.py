import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class CuisineRecommender:
    """
    Cuisine-based Restaurant Recommender
    """

    def __init__(self, df: pd.DataFrame):
        """
        Expected columns:
        - display_name
        - clean_cuisines
        - clean_reviews
        - rating
        """
        self.df = df.reset_index(drop=True)

        # Build text corpus
        self.text_data = (
            self.df["clean_cuisines"] + " " + self.df["clean_reviews"]
        )

        self.vectorizer = TfidfVectorizer(
            max_features=4000,
            stop_words="english",
            ngram_range=(1, 2)
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(self.text_data)

    def recommend_by_cuisine(
        self,
        cuisine: str,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Recommend restaurants primarily based on cuisine.
        """

        cuisine = cuisine.lower().strip()
        if not cuisine:
            return pd.DataFrame()

        # Filter restaurants that contain the cuisine
        cuisine_mask = self.df["clean_cuisines"].str.contains(cuisine)
        filtered_df = self.df[cuisine_mask]

        if filtered_df.empty:
            return pd.DataFrame()

        # Vectorize cuisine query
        cuisine_vector = self.vectorizer.transform([cuisine])

        # Compute similarity ONLY for filtered rows
        filtered_indices = filtered_df.index
        similarities = cosine_similarity(
            cuisine_vector,
            self.tfidf_matrix[filtered_indices]
        ).flatten()

        # Attach similarity scores
        results = filtered_df.copy()
        results["similarity"] = similarities

        # Rank: similarity first, rating second
        results = results.sort_values(
            by=["similarity", "rating"],
            ascending=False
        )

        return results.head(top_n)[
            ["display_name", "rating"]
        ]
