from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load model
with open("bangalore_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load location encoder
with open("locations.pkl", "rb") as f:
    le = pickle.load(f)

LOCATIONS = [
    "1st Block Jayanagar", "1st Phase JP Nagar", "2nd Phase Judicial Layout",
    "5th Block Hbr Layout", "7th Phase JP Nagar", "8th Phase JP Nagar",
    "Abbigere", "Akshaya Nagar", "Ambalipura", "Ambedkar Nagar",
    "Anekal", "Anjanapura", "Electronic City", "Electronic City Phase II",
    "Hebbal", "Hoodi", "Horamavu Agara", "HSR Layout", "Indira Nagar",
    "JP Nagar", "Kadugodi", "Kanakpura Road", "Kengeri",
    "Koramangala", "KR Puram", "Kudlu Gate", "Kumaraswamy Layout",
    "Marathahalli", "Mysore Road", "Nagarbhavi", "Neeladri Nagar",
    "Old Airport Road", "Panathur", "Raja Rajeshwari Nagar",
    "Rajaji Nagar", "Ramamurthy Nagar", "Sarjapur", "Sarjapur Road",
    "Thanisandra", "Thumkunta", "Uttarahalli", "Varthur", "Vidyaranyapura",
    "Vijayanagar", "Whitefield", "Yelahanka", "Yeshwanthpur", "Other"
]

AREA_TYPES = ["Super built-up  Area", "Built-up  Area", "Plot  Area", "Carpet  Area"]

class HouseInput(BaseModel):
    location: str
    area_type: str
    total_sqft: float
    bath: float
    balcony: float
    bhk: int

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "locations": LOCATIONS,
        "area_types": AREA_TYPES
    })

@app.post("/predict")
async def predict(data: HouseInput):
    try:
        # Encode location safely
        known_classes = list(le.classes_)
        if data.location in known_classes:
            location_encoded = int(le.transform([data.location])[0])
        else:
            location_encoded = 0

        # Encode area type
        area_map = {
            "Super built-up  Area": 3,
            "Built-up  Area": 1,
            "Plot  Area": 2,
            "Carpet  Area": 0
        }
        area_encoded = area_map.get(data.area_type, 3)

        # Feature engineering
        price_per_sqft = 5000
        total_rooms = data.bhk + data.bath

        features = np.array([[
            area_encoded,
            location_encoded,
            data.total_sqft,
            data.bath,
            data.balcony,
            data.bhk,
            price_per_sqft,
            total_rooms
        ]])

        prediction = model.predict(features)[0]
        price_lakhs = round(float(prediction), 2)
        price_crores = round(price_lakhs / 100, 3)

        return {
            "success": True,
            "price_lakhs": f"₹{price_lakhs:,.2f} Lakhs",
            "price_crores": f"₹{price_crores:.3f} Crores",
            "price_numeric": price_lakhs
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/locations")
async def get_locations():
    return {"locations": LOCATIONS}