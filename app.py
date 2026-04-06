import base64
import json
import os
import re

import anthropic
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from anthropic import AuthenticationError, APIConnectionError, APIStatusError

load_dotenv()

app = Flask(__name__)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

EXTRACT_PROMPT = """You are a medical data extraction assistant. Look at this image of handwritten or printed blood pressure readings.

Extract every blood pressure reading you can find. Each reading has a systolic (top number) and diastolic (bottom number), written as systolic/diastolic (e.g. 120/80).

Return ONLY a JSON object in this exact format, with no other text:
{
  "readings": [
    {"systolic": 120, "diastolic": 80},
    {"systolic": 135, "diastolic": 85}
  ],
  "notes": "any relevant notes about the readings or image quality"
}

If you cannot find any blood pressure readings, return:
{"readings": [], "notes": "No blood pressure readings found"}

Important: Only include readings where you are confident both numbers are blood pressure values."""


def extract_readings_from_image(image_data: str, media_type: str) -> dict:
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {"type": "text", "text": EXTRACT_PROMPT},
                ],
            }
        ],
    )

    text = next((b.text for b in response.content if b.type == "text"), "")

    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
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

    # Validate media type
    content_type = file.content_type or "image/jpeg"
    allowed = {"image/jpeg", "image/png", "image/gif", "image/webp"}
    if content_type not in allowed:
        # Default to jpeg for unknown types
        content_type = "image/jpeg"

    image_bytes = file.read()
    if len(image_bytes) > 20 * 1024 * 1024:  # 20 MB limit
        return jsonify({"error": "Image too large (max 20 MB)"}), 400

    image_data = base64.standard_b64encode(image_bytes).decode("utf-8")

    try:
        result = extract_readings_from_image(image_data, content_type)
    except AuthenticationError:
        return jsonify({"error": "Invalid API key. Please set ANTHROPIC_API_KEY in your .env file."}), 500
    except APIConnectionError:
        return jsonify({"error": "Could not reach the Anthropic API. Check your internet connection."}), 500
    except APIStatusError as e:
        return jsonify({"error": f"Anthropic API error: {e.message}"}), 500
    except json.JSONDecodeError:
        return jsonify({"error": "Could not parse readings from the image. Please try a clearer photo."}), 500

    readings = result.get("readings", [])
    averages = calculate_averages(readings)

    return jsonify(
        {
            "readings": readings,
            "averages": averages,
            "notes": result.get("notes", ""),
        }
    )


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
