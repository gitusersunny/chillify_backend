from model import recommendation
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

@app.post("/recommend")
def recommendsong(data:SongRequest):
    return {"song" :data.song_name , "recommendations":recommendation(data.song_name)}

