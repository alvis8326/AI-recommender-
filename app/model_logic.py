import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sqlalchemy import create_engine

# Database setup - use your real connection string here
DATABASE_URL = "mysql+pymysql://u555583069_campus:Jun8326789@srv1409.hstgr.io:3306/u555583069_campus"
engine = create_engine(DATABASE_URL)

# Load data functions
def load_booking_data():
    query = """
    SELECT user_id, facility_id, facilitycourt_id, booking_start_time, booking_end_time
    FROM facility_bookings
    """
    df = pd.read_sql(query, engine)
    df["booking_start_time"] = pd.to_datetime(df["booking_start_time"])
    df["booking_end_time"] = pd.to_datetime(df["booking_end_time"])
    return df

def load_court_data():
    query = """
    SELECT id, facility_id, court_number, status
    FROM facility_courts
    """
    return pd.read_sql(query, engine)

def load_sport_facility_data():
    query = """
    SELECT id, name
    FROM sports_facilities
    """
    return pd.read_sql(query, engine)

def load_user_data():
    query = """
    SELECT id, name
    FROM users
    """
    return pd.read_sql(query, engine)

# Helper functions (get facility name, user name, etc)
def get_facility_name(facility_id):
    sport_df = load_sport_facility_data()
    facility_name = sport_df.loc[sport_df["id"] == facility_id, "name"]
    return facility_name.iloc[0] if not facility_name.empty else "Unknown Facility"

def get_user_name(user_id):
    user_df = load_user_data()
    user_name = user_df.loc[user_df["id"] == user_id, "name"]
    return user_name.iloc[0] if not user_name.empty else "Unknown user"

# Your time slots definition
time_slots = pd.date_range("08:00", "22:00", freq="60min").strftime("%H:%M").tolist()

# User preference function
def get_user_preference(user_id, facility_id):
    df = load_booking_data()
    user_bookings = df[(df["user_id"] == user_id) & (df["facility_id"] == facility_id)].copy()

    if user_bookings.empty:
        return None

    user_bookings["start_minutes"] = user_bookings["booking_start_time"].dt.hour * 60 + user_bookings["booking_start_time"].dt.minute
    preferred_times = user_bookings["start_minutes"].mode()
    preferred_time = max(preferred_times) if len(preferred_times) > 1 else preferred_times.iloc[0]

    preferred_hour = preferred_time // 60
    preferred_minute = preferred_time % 60
    preferred_time_str = f"{preferred_hour:02d}:{preferred_minute:02d}"

    return preferred_time_str

# Check court availability
def check_available_courts(facility_id, date, preferred_time):
    courts_df = load_court_data()
    available_courts = courts_df[(courts_df["facility_id"] == facility_id) & (courts_df["status"] == "available")].copy()

    if available_courts.empty:
        return []

    return [(row["court_number"], preferred_time) for _, row in available_courts.iterrows()]

# Train KMeans model once and reuse it
def train_kmeans():
    df = load_booking_data()
    df["start_minutes"] = df["booking_start_time"].dt.hour * 60 + df["booking_start_time"].dt.minute
    X = df[["start_minutes"]].values
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X)
    return kmeans

kmeans_model = train_kmeans()

def recommend_best_slot(user_id, facility_id, date):
    preferred_time = get_user_preference(user_id, facility_id)

    if preferred_time:
        available_courts = check_available_courts(facility_id, date, preferred_time)
        if available_courts:
            return available_courts[0]

    # AI suggestion fallback
    available_slots = []
    for time in time_slots:
        courts = check_available_courts(facility_id, date, time)
        if courts:
            available_slots.append(time)

    if not available_slots:
        return None

    available_minutes = [int(time.split(":")[0]) * 60 + int(time.split(":")[1]) for time in available_slots]
    predicted_slot_idx = np.argmin(kmeans_model.predict(np.array(available_minutes).reshape(-1, 1)))
    best_time = available_slots[predicted_slot_idx]

    best_court = check_available_courts(facility_id, date, best_time)
    if best_court:
        return best_court[0]

    return None
