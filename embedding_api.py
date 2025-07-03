from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from insightface.app import FaceAnalysis
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import cv2
from typing import List
import json
from pydantic import BaseModel
import base64

app = FastAPI()

class Embedder:
    def __init__(self):
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.2)
        
    def get_embeddings(self, image):
        faces = self.app.get(image)
        if len(faces) == 0:
            return None
        embedding = faces[0]['embedding']
        return normalize(embedding.reshape(1, -1))[0]

embedder = Embedder()

class Photo(BaseModel):
    image: str
    
@app.post("/report")
async def get_embedding(photo: Photo):
    
    file_bytes = base64.b64decode(photo.image)

    try:
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        embedding = embedder.get_embeddings(image)
        
        if embedding is None:
            raise HTTPException(status_code=400, detail="No faces detected in image")
            
        return JSONResponse({
            "image_embedding": str(embedding.tolist())
        })
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
class Item(BaseModel):
    image: str
    embedding_strings: List = []
    
    
@app.post("/search")
async def process_image(item: Item):
    file_bytes = base64.b64decode(item.image)
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    image_embedding = embedder.get_embeddings(img)
    if image_embedding is None:
        raise HTTPException(status_code=400, detail="No faces detected in the image")
    
    list_embeddings = []
    for emb in item.embedding_strings:
        list_embeddings.append(emb)
        
    similarities = []
    for idx, emb_str in enumerate(list_embeddings):
        
        try:            
            target_embedding = json.loads(emb_str)
        
            target_embedding = np.array(target_embedding).reshape(1, -1)
            image_embedding_reshaped = image_embedding.reshape(1, -1)
            
            similarity = cosine_similarity(target_embedding, image_embedding_reshaped)[0][0]
            
            if similarity >= 0.4:
                similarities.append({
                    "index": idx,
                    "similarity": float(similarity),
                })
        except Exception as e:
            continue  
    
    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    top_5 = similarities[:5]
    
    return {
        "image_embedding": str(image_embedding.tolist()),
        "similar_embeddings": top_5
    }