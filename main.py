from model import recommendation
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hello World"}

class SongRequest(BaseModel):
    song_name: str

@app.post("/recommend")
def recommendsong(data:SongRequest):
    return {"song" :data.song_name , "recommendations":recommendation(data.song_name)}

