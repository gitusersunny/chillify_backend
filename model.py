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
scaled_features = scaler.fit_transform(df[NUMERICAL_FEATURES]).astype(np.float32)
df_scaled = scaled_features

# -----------------------------
# KMeans clustering
# -----------------------------
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=5,        # reduce from 10
    max_iter=100
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

    match_idx = df.index[df['clean_name'] == clean_song]
    if len(match_idx) == 0:
        return {"success": False, "recommendations": []}

    song_idx = match_idx[0]
    song_cluster = df.at[song_idx, 'cluster']

    # Indices of songs in same cluster
    cluster_indices = df.index[df['cluster'] == song_cluster].values

    # Query vector
    query_vec = df_scaled[song_idx]

    # Cluster vectors
    cluster_features = df_scaled[cluster_indices]

    # Similarity (1 × N)
    sims = cosine_similarity(
        query_vec.reshape(1, -1),
        cluster_features
    )[0]

    # Sort by similarity (descending)
    sorted_idx = np.argsort(sims)[::-1]

    recommendations = []
    seen = set()

    for idx in sorted_idx:
        real_idx = cluster_indices[idx]

        # Skip the same song
        if real_idx == song_idx:
            continue

        key = (df.at[real_idx, 'track_name'], df.at[real_idx, 'artists'])

        # Skip duplicates
        if key in seen:
            continue

        seen.add(key)
        recommendations.append({
            "track_name": key[0],
            "artists": key[1]
        })

        if len(recommendations) == no_of_reco:
            break

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

    mood_vector = np.array(
        [MOOD_PROFILES[mood][f] for f in NUMERICAL_FEATURES],
        dtype=np.float32
    ).reshape(1, -1)

    mood_scaled = scaler.transform(mood_vector).astype(np.float32)

    # Find closest cluster
    sims = cosine_similarity(mood_scaled, kmeans.cluster_centers_)[0]
    best_cluster = np.argmax(sims)

    cluster_indices = df.index[df['cluster'] == best_cluster].values
    cluster_features = df_scaled[cluster_indices]

    similarities = cosine_similarity(mood_scaled, cluster_features)[0]

    top_indices = np.argsort(similarities)[::-1][:no_of_reco]
    final_indices = cluster_indices[top_indices]

    recommendations = df.loc[
        final_indices, ['track_name', 'artists']
    ].to_dict(orient="records")

    return {"success": True, "recommendations": recommendations}