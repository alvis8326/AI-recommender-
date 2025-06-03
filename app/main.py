# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel # For request body validation
import logging # For logging

# Import your core logic function
from model_logic import recommend_best_slot # This is key!

# Configure basic logging for the FastAPI app part
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

app = FastAPI(
    title="Facility Booking Recommender API",
    description="Provides facility booking recommendations based on user preference and AI.",
    version="1.0.0"
)

class BookingRequest(BaseModel):
    user_id: int
    facility_id: int
    date: str # Example: "2025-04-15". Consider using datetime for more robust validation with Pydantic.

class RecommendationResponse(BaseModel):
    court_number: str | int # Can be string like "Court 1" or int
    time: str
    type: str # e.g., "user_preference", "ai_suggestion"

class ErrorResponse(BaseModel):
    error: str
    message: str

@app.post("/recommend", response_model=RecommendationResponse, responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def recommend_slot_api(request_data: BookingRequest):
    """
    Recommends a booking slot for a user at a specific facility on a given date.
    """
    try:
        logging.info(f"Received recommendation request: user_id={request_data.user_id}, facility_id={request_data.facility_id}, date={request_data.date}")
        
        recommendation = recommend_best_slot(
            user_id=request_data.user_id,
            facility_id=request_data.facility_id,
            date=request_data.date
        )

        if recommendation and "error" not in recommendation:
            logging.info(f"Recommendation found: {recommendation}")
            return RecommendationResponse(**recommendation)
        elif recommendation and "error" in recommendation:
            logging.warning(f"No recommendation found or error: {recommendation.get('message', 'Unknown error')}")
            raise HTTPException(status_code=404, detail=recommendation)
        else: # Should not happen if recommend_best_slot always returns a dict
            logging.error("recommend_best_slot returned an unexpected value.")
            raise HTTPException(status_code=500, detail={"error": "Internal Server Error", "message": "Unexpected response from recommendation logic."})

    except HTTPException as http_exc: # Re-raise HTTPExceptions
        raise http_exc
    except ValueError as ve: # Example: if date format is wrong and model_logic raises it
        logging.error(f"ValueError during recommendation: {ve}", exc_info=True)
        raise HTTPException(status_code=400, detail={"error": "Bad Request", "message": str(ve)})
    except Exception as e:
        logging.error(f"An unexpected error occurred in /recommend endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": "Internal Server Error", "message": "An unexpected error occurred processing your request."})

@app.get("/health", tags=["Utilities"])
async def health_check():
    """
    Performs a basic health check of the service.
    (You can extend this to check database connection from model_logic if needed)
    """
    # Simple health check. To check DB, you'd need to import 'engine' or a test function from model_logic
    # from model_logic import engine # If you want to test DB connection here
    # try:
    #     with engine.connect() as connection:
    #         connection.execute(pd.io.sql.SQLAlchemyTextClause("SELECT 1"))
    #     db_status = "connected"
    # except Exception as e:
    #     db_status = f"error: {e}"
    #     return {"status": "unhealthy", "database_connection": db_status}

    return {"status": "healthy", "message": "API is up and running!"}

# If you want to run this locally with uvicorn for testing:
# uvicorn main:app --reload