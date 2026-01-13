import streamlit as st
from datasets import load_dataset

from src.preprocessing import preprocess_data
from src.recommender import CuisineRecommender

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Cuisine-Based Restaurant Recommender",
    layout="wide"
)

st.title("🍛 Cuisine-Based Restaurant Recommendation System")
st.write(
    "Select a cuisine and discover top-rated, similar restaurants"
)

# --------------------------------------------------
# Load Dataset (NO CACHE)
# --------------------------------------------------
with st.spinner("Loading dataset..."):
    dataset = load_dataset(
        "ManikaSaini/zomato-restaurant-recommendation",
        split="train"
    )
    df = dataset.to_pandas()

# --------------------------------------------------
# Early Sampling
# --------------------------------------------------
MAX_ROWS = 8000
df = df.sample(
    min(len(df), MAX_ROWS),
    random_state=42
).reset_index(drop=True)

# --------------------------------------------------
# Preprocessing
# --------------------------------------------------
with st.spinner("Cleaning data..."):
    df = preprocess_data(df)

# --------------------------------------------------
# Build Recommender
# --------------------------------------------------
@st.cache_resource
def build_recommender(data):
    return CuisineRecommender(data)

with st.spinner("Building recommender..."):
    recommender = build_recommender(df)

st.success("System ready!")

# --------------------------------------------------
# Sidebar Inputs
# --------------------------------------------------
st.sidebar.header("🔍 Choose Cuisine")

# Extract unique cuisines
all_cuisines = sorted(
    {
        cuisine.strip()
        for sublist in df["clean_cuisines"].str.split()
        for cuisine in sublist
        if len(cuisine) > 3
    }
)

selected_cuisine = st.sidebar.selectbox(
    "Select a cuisine",
    all_cuisines
)

top_n = st.sidebar.slider(
    "Number of recommendations",
    5, 15, 10
)

# --------------------------------------------------
# Recommendations
# --------------------------------------------------
if st.sidebar.button("Get Recommendations"):
    with st.spinner("Finding best restaurants..."):
        results = recommender.recommend_by_cuisine(
            cuisine=selected_cuisine,
            top_n=top_n
        )

    st.subheader(f"🍽️ Top {top_n} {selected_cuisine.title()} Restaurants")

    if not results.empty:
        for i, row in results.iterrows():
            st.write(
                f"⭐ **{row['display_name']}** — Rating: {row['rating']}"
            )
    else:
        st.warning("No restaurants found for this cuisine.")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("Cuisine-first recommender | TF-IDF + Similarity + Ratings")
