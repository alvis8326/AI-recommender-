# model_logic.py

import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sqlalchemy import create_engine
import logging

# --- Database Configuration ---
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    logging.error("CRITICAL: DATABASE_URL environment variable not set.")
    # You might want to raise an exception here or handle it gracefully
    # For local testing, you could use a default, but it's better to set the env var
    # Example: DATABASE_URL = "mysql+pymysql://user:pass@host/db"
    raise ValueError("DATABASE_URL not set")

try:
    engine = create_engine(DATABASE_URL)
    logging.info("Database engine created successfully.")
except Exception as e:
    logging.error(f"Failed to create database engine: {e}")
    raise

# --- All your data loading functions (load_booking_data, load_court_data, etc.) ---
# Make sure they use the 'engine' defined above.
# Example:
def load_booking_data():
    # ... your query ...
    df = pd.read_sql(query, engine)
    # ...
    return df

# (Include load_court_data, load_sport_facility_data, load_user_data here)
# ... (rest of your helper functions: get_facility_name, get_user_name, get_user_preference, check_available_courts) ...
# Ensure these functions are present and correct as in the previous detailed app.py example.

# --- Train AI Model (KMeans) ---
# This will run once when model_logic.py is first imported by main.py
logging.info("Attempting to train KMeans model in model_logic.py...")
kmeans_model = None # Initialize
try:
    df_bookings_for_model = load_booking_data() # Ensure this function is defined
    if not df_bookings_for_model.empty:
        df_bookings_for_model["start_minutes"] = df_bookings_for_model["booking_start_time"].dt.hour * 60 + df_bookings_for_model["booking_start_time"].dt.minute
        X = df_bookings_for_model[["start_minutes"]].values
        if X.shape[0] > 0: # Check if X is not empty
            # Determine n_clusters, ensuring it's not more than n_samples
            n_samples = X.shape[0]
            k = min(3, n_samples) # Use at most 3 clusters, or fewer if fewer samples
            if k > 0:
                kmeans_model = KMeans(n_clusters=k, random_state=42, n_init='auto')
                kmeans_model.fit(X)
                logging.info(f"KMeans model trained successfully with {k} clusters.")
            else:
                logging.warning("Not enough samples to train KMeans model (k=0). Model will not be available.")
        else:
            logging.warning("No data points (X is empty) to train KMeans model. Model will not be available.")
    else:
        logging.warning("No booking data available to train KMeans model. Model will not be available.")
except Exception as e:
    logging.error(f"Error training KMeans model: {e}", exc_info=True)
    # kmeans_model remains None

# --- Your core recommendation logic function ---
def recommend_best_slot(user_id: int, facility_id: int, date: str):
    # ... (your full recommend_best_slot logic here, using the globally trained kmeans_model)
    # Make sure it uses the `kmeans_model` defined above in this file.
    # Refer to the logic from the previous combined app.py example.
    # Remember to handle the case where kmeans_model is None.

    facility_name = get_facility_name(facility_id) # Ensure get_facility_name is defined
    preferred_time = get_user_preference(user_id, facility_id) # Ensure get_user_preference is defined

    if preferred_time:
        available_courts_at_preferred_time = check_available_courts(facility_id, date, preferred_time) # Ensure this is defined
        if available_courts_at_preferred_time:
            logging.info(f"✅ Preferred time {preferred_time} is available for {facility_name}!")
            return {"court_number": available_courts_at_preferred_time[0][0], "time": available_courts_at_preferred_time[0][1], "type": "user_preference"}

    logging.info(f"⚠️ No preferred time for {facility_name} or not available. Checking AI suggestions...")

    if kmeans_model is None: # Check if model was trained
        logging.error("KMeans model is not available for AI-based suggestions.")
        return {"error": "AI model not available.", "message": "No available slots found due to model error."}

    # ... (rest of the AI-based suggestion logic using kmeans_model, time_slots, check_available_courts etc.)
    # This logic would be similar to what was in the recommend_best_slot in the combined app.py.
    # Ensure 'time_slots' is defined globally or passed appropriately.
    global time_slots # If time_slots is defined globally in this file
    # ... (The logic for finding available_slots, using kmeans_model.predict, and finding best_court)

    # Placeholder for the rest of the AI logic:
    # This part needs to be filled with your detailed AI slot recommendation logic
    # from the `recommend_best_slot` function in the previous `app.py` example.
    # It involves:
    # 1. Getting all available_slots (e.g., by iterating `time_slots` and calling `check_available_courts`)
    # 2. Using `kmeans_model.predict` on these available slots (converted to minutes)
    # 3. Selecting the `best_time` based on KMeans (e.g., belonging to a less popular cluster or other logic)
    # 4. Getting the `best_court` at that `best_time`.

    # Example sketch (replace with your full logic):
    all_potentially_available_slots = []
    for time_slot_str in time_slots: # Assuming time_slots is defined (e.g. pd.date_range("08:00", "22:00", freq="60min").strftime("%H:%M").tolist())
        courts_at_this_slot = check_available_courts(facility_id, date, time_slot_str)
        if courts_at_this_slot:
            all_potentially_available_slots.append(time_slot_str)
    
    if not all_potentially_available_slots:
        logging.info(f"❌ No available slots found for {facility_name} on {date}")
        return {"error": "No available slots found", "message": f"No courts are available for {facility_name} on {date}."}

    # ... (the rest of your KMeans logic to pick best_time_slot_from_ai from all_potentially_available_slots) ...
    # For now, let's assume a simplified fallback if the detailed AI logic isn't fully here:
    best_time_slot_from_ai = all_potentially_available_slots[0] # Simplified: pick first available
    
    best_courts_at_ai_time = check_available_courts(facility_id, date, best_time_slot_from_ai)

    if best_courts_at_ai_time:
        logging.info(f"🤖 AI-selected slot for {facility_name}: Court {best_courts_at_ai_time[0][0]} at {best_courts_at_ai_time[0][1]}")
        return {"court_number": best_courts_at_ai_time[0][0], "time": best_courts_at_ai_time[0][1], "type": "ai_suggestion"}
    
    return {"error": "No AI suggestion available", "message": "Could not find a suitable slot using AI."}

# Make sure all helper functions (load_booking_data, load_court_data, load_sport_facility_data,
# load_user_data, get_facility_name, get_user_name, get_user_preference, check_available_courts)
# are defined in this model_logic.py file. And `time_slots` should also be defined here.
# Example time_slots definition if not already present
time_slots = pd.date_range("08:00", "22:00", freq="60min").strftime("%H:%M").tolist()

# Add logging configuration if not already done globally
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')