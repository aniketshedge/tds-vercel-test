import fastapi
import json

app = fastapi.FastAPI()

# Load data from the JSON file
def load_data():
    with open('q-vercel-python.json', 'r') as file:
        return json.load(file)

data = load_data()

# Create a dictionary for quick lookup
marks_dict = {entry["name"]: entry["marks"] for entry in data}

@app.get("/api")
def read_marks(names: list = fastapi.Query(...)):  # Use fastapi.Query explicitly
    # Retrieve marks in the order of names provided
    marks = [marks_dict.get(name, None) for name in names]
    return {"marks": marks}
