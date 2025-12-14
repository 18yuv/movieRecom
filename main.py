# !pip install gradio requests pandas scikit-learn numpy

import gradio as gr
import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# API KEY
TMDB_API_KEY = "91ea23925f32a42abba3e0512c47b3d1"


# GENRE MAPPING
GENRE_MAP = {
    "action": 28, "adventure": 12, "animation": 16,
    "comedy": 35, "crime": 80, "documentary": 99,
    "drama": 18, "family": 10751, "fantasy": 14,
    "history": 36, "horror": 27, "music": 10402,
    "mystery": 9648, "romance": 10749,
    "sci-fi": 878, "thriller": 53,
    "war": 10752, "western": 37
}

# FETCH MOVIES
def fetch_movies():
    print("Fetching movie dataset... (one-time process)")
    all_movies = []

    for page in range(1, 101):  # ~2000 movies
        url = f"https://api.themoviedb.org/3/movie/popular"
        params = {
            "api_key": TMDB_API_KEY,
            "language": "en-US",
            "page": page
        }

        data = requests.get(url, params=params).json()
        results = data.get("results", [])

        for m in results:
            all_movies.append({
                "title": m.get("title", ""),
                "overview": m.get("overview", ""),
                "rating": m.get("vote_average", 0),
                "genre_ids": m.get("genre_ids", [])
            })

    df = pd.DataFrame(all_movies)
    df.to_csv("movies.csv", index=False)
    print("Dataset ready.")
    return df

# LOAD DATA
def load_data():
    if os.path.exists("movies.csv"):
        return pd.read_csv("movies.csv")
    else:
        return fetch_movies()

df = load_data()
df["overview"] = df["overview"].fillna("")

# TRAIN MODEL
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["overview"])
similarity_matrix = cosine_similarity(tfidf_matrix)


# RECOMMENDER FUNCTION
def recommend_ui(genre, movie_name, min_rating, num_suggestions):
    movie_name = movie_name.lower()
    genre_id = GENRE_MAP.get(genre.lower(), None)

    # Filter by genre
    if genre_id:
        df_filtered = df[df["genre_ids"].apply(lambda g: str(genre_id) in str(g))]
    else:
        df_filtered = df.copy()

    # Filter by rating
    df_filtered = df_filtered[df_filtered["rating"] >= min_rating]

    # Find closest movie match
    matches = df[df["title"].str.lower().str.contains(movie_name)]
    if matches.empty:
        return f"No movie found similar to '{movie_name}'. Try another movie!"

    idx = matches.index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    result = "🎬 Movie Recommendations:\n\n"

    count = 0
    for i, s in sorted_scores:
        row = df.iloc[i]

        if genre_id and genre_id not in eval(str(row["genre_ids"])):
            continue
        if row["rating"] < min_rating:
            continue
        if row["title"].lower() == movie_name:
            continue

        result += f"{row['title']}**  ⭐({row['rating']})\n"
        result += f"{row['overview'][:150]}...\n\n"

        count += 1
        if count >= num_suggestions:
            break

    if count == 0:
        return "No movies matched your filters."

    return result


# GRADIO UI
genre_list = list(GENRE_MAP.keys())

ui = gr.Interface(
    fn=recommend_ui,
    inputs=[
        gr.Dropdown(genre_list, label="Favorite Genre"),
        gr.Textbox(label="Favorite Movie (e.g., Captain America)"),
        gr.Slider(0, 10, value=5, label="Minimum Rating"),
        gr.Slider(1, 10, value=5, step=1, label="Number of Suggestions")
    ],
    outputs=gr.Markdown(),
    title="🎥 Movie Recommender System (ML Model)",
    description="A simple ML-based movie recommender using TF-IDF vectorization + cosine similarity."
)

ui.launch()