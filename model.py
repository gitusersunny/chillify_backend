import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("dataset.csv")
df = df.sample(n=5000,random_state=42).reset_index(drop=True)
df.head()

numerical_features = [
    'valence' , 'danceability' , 'energy' , 'tempo' , 'acousticness' , 'liveness' , 'speechiness' , 'instrumentalness'
]
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[numerical_features]),columns=numerical_features)
train_data , test_data = train_test_split(df_scaled,test_size=0.2,random_state=42)

inertia = []
k_values = range(1,11)

for k in k_values:
    Kmeans = KMeans(n_clusters=k,random_state=42)
    Kmeans.fit(train_data)
    inertia.append(Kmeans.inertia_)

optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['cluster'] = kmeans.fit_predict(df_scaled)

def recommendation(song_name, no_of_reco = 5):
    clean_song_name = song_name.lower().replace(" ", "")
    df['clean_name'] = (
        df['track_name']
        .str.lower()
        .str.replace(" ", "", regex=False)
    )

    if clean_song_name not in df['clean_name'].values:
        return { "success": False, "recommendations": []}

    song_cluster = df[df['clean_name'] == clean_song_name]['cluster'].values[0]
    same_cluster_song = df[df['cluster'] == song_cluster]

    same_cluster_song = same_cluster_song.drop_duplicates(subset=['track_name', 'artists']).reset_index(drop=True)

    song_index_list = same_cluster_song[same_cluster_song['clean_name'] == clean_song_name].index
    if len(song_index_list) == 0:
        return { "success": False, "recommendations": []}

    song_index = song_index_list[0]

    # Extract features
    cluster_features = same_cluster_song[numerical_features]
    similarity = cosine_similarity(cluster_features, cluster_features)

    # Similar songs (excluding the song itself)
    similar_song_indices = np.argsort(similarity[song_index])[-(no_of_reco+1):-1][::-1]

    # -----------------------------
    # ✅ Add the original song at index 0
    # -----------------------------
    original_song = same_cluster_song.iloc[song_index][['track_name', 'artists']].to_dict()

    similar_songs = same_cluster_song.iloc[similar_song_indices][['track_name', 'artists']].to_dict(orient="records")

    final_list = [original_song] + similar_songs  # original song at 0th position

    return {
        "success": True,
        "recommendations": final_list
    }


export = recommendation