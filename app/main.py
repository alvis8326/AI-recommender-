from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# Import your existing logic function
from app import model_logic  # make sure your function is importable

app = FastAPI()

class BookingRequest(BaseModel):
    user_id: int
    facility_id: int
    date: str

class RecommendationResponse(BaseModel):
    court_number: str | int
    time: str
    type: str

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_slot(request: BookingRequest):
    result = recommend_best_slot(request.user_id, request.facility_id, request.date)
    if isinstance(result, tuple):
        return {
            "court_number": result[0],
            "time": result[1],
            "type": "ai_suggestion"
        }
    else:
        raise HTTPException(status_code=404, detail="No available courts found.")
