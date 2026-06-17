# app/main.py
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
from typing import List, Optional
import time
from contextlib import asynccontextmanager

# --- Models ---
class RecommendationRequest(BaseModel):
    song_name: str
    no_of_reco: Optional[int] = 5

class RecommendationResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    recommendations: Optional[List[dict]] = None
    response_time_ms: Optional[float] = None

# --- Load Model with Lifespan Context Manager ---
class ModelManager:
    def __init__(self):
        self.model = None
        self.metadata = {}
    
    def load(self):
        """Load model once at startup."""
        print("🔄 Loading recommendation model...")
        start_time = time.time()
        
        with open("app/ml_model/recommender_latest.pkl", 'rb') as f:
            model_data = pickle.load(f)
            self.model = model_data
            self.metadata = {
                'version': model_data.get('model_version', '1.0.0'),
                'training_date': model_data.get('training_date', 'unknown'),
                'clusters': model_data.get('optimal_k', 0)
            }
        
        # Warm up the model (optional)
        test_predict = self.predict("let me love you", 3)
        load_time = (time.time() - start_time) * 1000
        
        print(f"✅ Model loaded in {load_time:.2f}ms")
        return self
    
    def predict(self, song_name, no_of_reco=5):
        """Fast prediction using pre-computed data."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        start_time = time.time()
        
        clean_song_name = song_name.lower().replace(" ", "")
        
        # O(1) lookup
        if clean_song_name not in self.model['song_lookup']:
            return {
                "success": False,
                "message": "No song found with this name.",
                "response_time_ms": 0
            }
        
        # Get cluster
        song_info = self.model['song_lookup'][clean_song_name]
        cluster_id = song_info['cluster']
        
        # Get pre-computed data
        cluster_data = self.model['cluster_data'][cluster_id]
        songs_df = cluster_data['songs']
        similarity_matrix = cluster_data['similarity_matrix']
        
        # Get song index
        if clean_song_name not in cluster_data['track_indices']:
            return {
                "success": False,
                "message": "Song found but not in cluster indices.",
                "response_time_ms": 0
            }
        
        song_idx = cluster_data['track_indices'][clean_song_name]
        
        # Get recommendations (using pre-computed similarity)
        similar_indices = np.argsort(similarity_matrix[song_idx])[-(no_of_reco+1):-1][::-1]
        
        recommendations = songs_df.iloc[similar_indices][
            ['track_name', 'artists']
        ].to_dict(orient='records')
        
        response_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "recommendations": recommendations,
            "response_time_ms": round(response_time, 2)
        }

# --- Initialize ---
model_manager = ModelManager()

# Load model at startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    model_manager.load()
    yield
    # Shutdown (optional cleanup)
    print("🔄 Shutting down...")

# --- FastAPI App ---
app = FastAPI(
    title="Song Recommendation API",
    description="Fast song recommendations using pre-computed similarities",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Endpoints ---
@app.get("/")
async def root():
    return {
        "message": "Song Recommendation API",
        "version": model_manager.metadata.get('version', '1.0.0'),
        "clusters": model_manager.metadata.get('clusters', 0),
        "training_date": model_manager.metadata.get('training_date', 'unknown')
    }

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Get song recommendations."""
    try:
        result = model_manager.predict(request.song_name, request.no_of_reco)
        return RecommendationResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend/batch")
async def get_batch_recommendations(songs: List[str], no_of_reco: int = 5):
    """Batch recommendations for multiple songs."""
    results = []
    for song in songs:
        result = model_manager.predict(song, no_of_reco)
        results.append({
            "song": song,
            "recommendations": result.get('recommendations', []),
            "success": result.get('success', False)
        })
    return {"batch_results": results}


@app.get("/stats")
async def get_stats():
    """Get model statistics."""
    if model_manager.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    total_songs = len(model_manager.model['song_lookup'])
    return {
        "total_songs": total_songs,
        "clusters": model_manager.metadata.get('clusters', 0),
        "model_version": model_manager.metadata.get('version', '1.0.0')
    }

@app.get("/model/version")
async def model_version():
    return {
        "version": model_manager.metadata.get('version', 'unknown'),
        "training_date": model_manager.metadata.get('training_date'),
        "clusters": model_manager.metadata.get('optimal_k', 0)
    }

@app.post("/model/reload")
async def reload_model(background_tasks: BackgroundTasks):
    background_tasks.add_task(model_manager.reload)
    return {"status": "reload initiated"}

@app.get("/model/metrics")
async def model_metrics():
    return model_manager.metadata.get('performance_metrics', {})

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_manager.model is not None,
        "model_version": model_manager.metadata.get('version', 'unknown')
    }