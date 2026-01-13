🍛 Cuisine-Based Restaurant Recommendation System

A content-based restaurant recommender system that allows users to discover the best restaurants by selecting a cuisine first, instead of searching by restaurant name.

This project is designed to be memory-efficient, scalable, and aligned with real-world user behavior, making it ideal for beginner-to-intermediate Data Science interviews.

🚀 Live Features

Cuisine-first recommendation flow

Content-based filtering using TF-IDF

Ranking based on text similarity + restaurant ratings

Handles messy real-world restaurant data

Lightweight & laptop-friendly

Interactive UI built with Streamlit

🎯 The action

Users select a cuisine (e.g., Italian, North Indian, Chinese), and the system recommends the most relevant and highly-rated restaurants for that cuisine.

🧠 Recommendation Approach

This is a Content-Based Recommender System.

Steps:

Clean and normalize cuisines, reviews, and restaurant names

Combine cuisines + reviews into a single text corpus

Convert text into vectors using TF-IDF

Compute cosine similarity between cuisine query and restaurants

Rank results using:

Similarity score

Restaurant rating (tie-breaker)

📊 Dataset

Source: Hugging Face
Dataset: Zomato Restaurant Recommendation Data

🔗 Dataset Link:
https://huggingface.co/datasets/ManikaSaini/zomato-restaurant-recommendation

Dataset Contains:

Restaurant names

Cuisines

User reviews

Ratings

Metadata

🛠️ Tech Stack

Python 3.10

Pandas & NumPy – data processing

Scikit-Learn – TF-IDF & cosine similarity

Hugging Face Datasets – dataset loading

Streamlit – web application UI

🧹 Data Cleaning Highlights

Removed duplicate and noisy restaurant names

Normalized cuisines into consistent text format

Handled missing and invalid ratings

Limited dataset size to prevent memory overflow

Optimized text processing to avoid MemoryError



👩‍💻 Author
Manika Saini

