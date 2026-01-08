import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Load & shuffle data
# -----------------------------
df = pd.read_csv("dataset.csv")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# -----------------------------
# Feature selection
# -----------------------------
NUMERICAL_FEATURES = [
    'valence', 'danceability', 'energy', 'tempo',
    'acousticness', 'liveness', 'speechiness', 'instrumentalness'
]

# -----------------------------
# Preprocessing (done ONCE)
# -----------------------------
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[NUMERICAL_FEATURES])

# -----------------------------
# KMeans clustering
# -----------------------------
OPTIMAL_K = 5
kmeans = KMeans(
    n_clusters=OPTIMAL_K,
    random_state=42,
    n_init=10
)

df['cluster'] = kmeans.fit_predict(scaled_features)

# -----------------------------
# Clean song names ONCE
# -----------------------------
df['clean_name'] = (
    df['track_name']
    .str.lower()
    .str.replace(" ", "", regex=False)
)

# -----------------------------
# Recommendation function
# -----------------------------
def recommendation(song_name: str, no_of_reco: int = 5):
    clean_song = song_name.lower().replace(" ", "")

    if clean_song not in df['clean_name'].values:
        return {"success": False, "recommendations": []}

    # Get song info
    song_row = df[df['clean_name'] == clean_song].iloc[0]
    song_cluster = song_row['cluster']

    # Filter same cluster
    cluster_songs = df[df['cluster'] == song_cluster]

    # Remove duplicates
    cluster_songs = cluster_songs.drop_duplicates(
        subset=['track_name', 'artists']
    ).reset_index(drop=True)

    # Get index of query song
    song_idx = cluster_songs[cluster_songs['clean_name'] == clean_song].index[0]

    # Compute similarity ONLY inside the cluster
    features = cluster_songs[NUMERICAL_FEATURES]
    similarity = cosine_similarity(features)

    # Top similar songs
    similar_indices = (
        np.argsort(similarity[song_idx])[::-1][1:no_of_reco + 1]
    )

    # Final list
    recommendations = (
        cluster_songs.iloc[[song_idx]][['track_name', 'artists']]
        .to_dict(orient="records")
        +
        cluster_songs.iloc[similar_indices][['track_name', 'artists']]
        .to_dict(orient="records")
    )

    return {
        "success": True,
        "recommendations": recommendations
    }

# -----------------------------
# Mood definitions (core logic)
# -----------------------------
MOOD_PROFILES = {
    "happy": {
        "valence": 0.9,
        "danceability": 0.8,
        "energy": 0.7,
        "tempo": 0.7,
        "acousticness": 0.2,
        "liveness": 0.3,
        "speechiness": 0.3,
        "instrumentalness": 0.1
    },
    "sad": {
        "valence": 0.1,
        "danceability": 0.3,
        "energy": 0.2,
        "tempo": 0.3,
        "acousticness": 0.7,
        "liveness": 0.2,
        "speechiness": 0.2,
        "instrumentalness": 0.4
    },
    "energetic": {
        "valence": 0.7,
        "danceability": 0.8,
        "energy": 0.9,
        "tempo": 0.9,
        "acousticness": 0.1,
        "liveness": 0.6,
        "speechiness": 0.4,
        "instrumentalness": 0.1
    },
    "calm": {
        "valence": 0.5,
        "danceability": 0.3,
        "energy": 0.2,
        "tempo": 0.3,
        "acousticness": 0.8,
        "liveness": 0.1,
        "speechiness": 0.1,
        "instrumentalness": 0.6
    },
    "romantic": {
        "valence": 0.6,
        "danceability": 0.5,
        "energy": 0.4,
        "tempo": 0.4,
        "acousticness": 0.6,
        "liveness": 0.2,
        "speechiness": 0.3,
        "instrumentalness": 0.3
    }
}

# -----------------------------
# Mood-based recommendation
# -----------------------------
def recommend_by_mood(mood: str, no_of_reco: int = 5):
    mood = mood.lower()

    if mood not in MOOD_PROFILES:
        return {"success": False, "recommendations": []}

    # Convert mood profile to dataframe
    mood_vector = pd.DataFrame([MOOD_PROFILES[mood]])

    # Scale mood vector
    mood_scaled = scaler.transform(mood_vector[NUMERICAL_FEATURES])

    # Find closest cluster
    cluster_centers = kmeans.cluster_centers_
    cluster_sim = cosine_similarity(mood_scaled, cluster_centers)
    best_cluster = np.argmax(cluster_sim)

    # Songs from best cluster
    cluster_songs = df[df['cluster'] == best_cluster].copy()

    # Compute similarity to mood
    song_features = scaler.transform(cluster_songs[NUMERICAL_FEATURES])
    similarities = cosine_similarity(mood_scaled, song_features)[0]

    cluster_songs['similarity'] = similarities

    # Top N recommendations
    recommendations = (
        cluster_songs
        .sort_values(by="similarity", ascending=False)
        .head(no_of_reco)[['track_name', 'artists']]
        .to_dict(orient="records")
    )

    return {
        "success": True,
        "recommendations": recommendations
    }
