from model import recommendation
from model import recommend_by_mood
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "Hello World"}

class SongRequest(BaseModel):
    song_name: str
    mood: Optional[str] = ""

@app.post("/recommend")
def recommend_song(data: SongRequest):
    if data.mood.strip() == "":
        return {
            "isMood":False,
            "song": data.song_name,
            "recommendations": recommendation(data.song_name)
        }
    else:
        return {
            "isMood":True,
            "song": data.song_name,
            "recommendations": recommend_by_mood(data.mood)
        }


