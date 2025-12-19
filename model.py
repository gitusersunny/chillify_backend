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

# Export reference
export = recommendation
