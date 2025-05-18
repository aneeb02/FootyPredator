# api/index.py
import os, pickle, requests
from flask import Flask, render_template, request
from datetime import datetime, timedelta

app = Flask(
    __name__,
    template_folder="../templates",  # point back to the root templates folder
)

# load your model & encoder
model = pickle.load(open("player_position_model.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

API_KEY = os.getenv("FOOTBALL_API_KEY")  # set this in Vercel dashboard
API_BASE = "http://api.football-data.org/v4"
HEADERS = {"X-Auth-Token": API_KEY}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["GET","POST"])
def predict():
    if request.method == "POST":
        stats = {k: float(request.form[k]) for k in request.form}
        features = [
            stats["age"],
            stats["height_cm"],
            stats["weight_kgs"],
            stats["overall_rating"],
            stats["potential"],
            stats["sprint_speed"],
            stats["short_passing"],
            stats["long_passing"],
            stats["dribbling"],
            stats["strength"]
        ]
        pred = model.predict([features])[0]
        pos = label_encoder.inverse_transform([pred])[0]
        conf = max(model.predict_proba([features])[0]) * 100
        return render_template("predict.html", stats=stats, position=pos, confidence=conf)
    return render_template("predict.html")

@app.route("/wiki")
def wiki():
    name = request.args.get("player","")
    if not name:
        return render_template("wiki.html")
    r = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{name}")
    data = r.json() if r.ok else {}
    return render_template(
        "wiki.html",
        player_name=name,
        summary=data.get("extract","No summary available"),
        image_url=data.get("thumbnail",{}).get("source")
    )

@app.route("/live")
def live():
    # live matches
    r = requests.get(f"{API_BASE}/matches?status=LIVE", headers=HEADERS)
    matches = r.json().get("matches",[]) if r.ok else []
    if not matches:
        # fallback to last week
        week_ago = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
        today    = datetime.utcnow().strftime("%Y-%m-%d")
        r = requests.get(f"{API_BASE}/matches?dateFrom={week_ago}&dateTo={today}", headers=HEADERS)
        matches = r.json().get("matches",[]) if r.ok else []
        return render_template("live.html", matches=matches, is_live=False)
    return render_template("live.html", matches=matches, is_live=True)
