import base64
import json
import logging
import os
import re

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

load_dotenv()

app = Flask(__name__)

# Suppress Flask request logs — no image data or readings are ever written to logs
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-1.5-flash:generateContent"
)

EXTRACT_PROMPT = """Look at this image of blood pressure readings (handwritten or printed).

Extract every blood pressure reading you can find. Each reading is two numbers separated by a slash — systolic on top, diastolic on bottom (e.g. 120/80).

Return ONLY a JSON object in this exact format, no other text:
{
  "readings": [
    {"systolic": 120, "diastolic": 80},
    {"systolic": 135, "diastolic": 85}
  ]
}

If no blood pressure readings are found, return:
{"readings": []}

Only include readings where you are confident both numbers are blood pressure values."""


def extract_readings_from_image(image_bytes: bytes, mime_type: str) -> dict:
    image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    payload = {
        "contents": [{
            "parts": [
                {"text": EXTRACT_PROMPT},
                {"inline_data": {"mime_type": mime_type, "data": image_b64}},
            ]
        }]
    }

    resp = requests.post(
        GEMINI_URL,
        params={"key": GEMINI_API_KEY},
        json=payload,
        timeout=30,
    )

    if resp.status_code == 400:
        raise ValueError("bad_image")
    if resp.status_code == 403:
        raise PermissionError("bad_key")
    if resp.status_code == 429:
        raise RuntimeError("rate_limit")
    resp.raise_for_status()

    text = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text.strip())
    return json.loads(text)


def calculate_averages(readings: list) -> dict:
    if not readings:
        return None

    systolic_values = [r["systolic"] for r in readings]
    diastolic_values = [r["diastolic"] for r in readings]

    avg_systolic = round(sum(systolic_values) / len(systolic_values))
    avg_diastolic = round(sum(diastolic_values) / len(diastolic_values))

    return {
        "avg_systolic": avg_systolic,
        "avg_diastolic": avg_diastolic,
        "count": len(readings),
        "category": classify_bp(avg_systolic, avg_diastolic),
    }


def classify_bp(systolic: int, diastolic: int) -> dict:
    if systolic < 120 and diastolic < 80:
        return {"label": "Normal", "color": "green"}
    elif systolic < 130 and diastolic < 80:
        return {"label": "Elevated", "color": "yellow"}
    elif systolic < 140 or diastolic < 90:
        return {"label": "High — Stage 1", "color": "orange"}
    elif systolic >= 180 or diastolic >= 120:
        return {"label": "Hypertensive Crisis", "color": "red"}
    else:
        return {"label": "High — Stage 2", "color": "red"}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyse", methods=["POST"])
def analyse():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No image selected"}), 400

    mime_type = file.content_type or "image/jpeg"
    allowed = {"image/jpeg", "image/png", "image/gif", "image/webp"}
    if mime_type not in allowed:
        mime_type = "image/jpeg"

    image_bytes = file.read()
    if len(image_bytes) > 20 * 1024 * 1024:
        return jsonify({"error": "Image too large (max 20 MB)"}), 400

    try:
        result = extract_readings_from_image(image_bytes, mime_type)
    except PermissionError:
        return jsonify({"error": "Invalid API key. Please check GEMINI_API_KEY in your .env file."}), 500
    except RuntimeError:
        return jsonify({"error": "Free usage limit reached. Try again in a minute."}), 429
    except ValueError:
        return jsonify({"error": "Could not read the image. Please try a different photo."}), 400
    except json.JSONDecodeError:
        return jsonify({"error": "Could not parse readings. Please try a clearer photo."}), 500
    except Exception:
        return jsonify({"error": "Something went wrong. Please try again."}), 500

    readings = result.get("readings", [])
    averages = calculate_averages(readings)

    return jsonify({"readings": readings, "averages": averages})


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
