from flask import Flask, request, render_template, redirect, url_for
import boto3
import os
from werkzeug.utils import secure_filename
import cv2
from collections import defaultdict
import time

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ---- AWS clients ----
REGION = "us-east-1"
BUCKET_NAME = "my-hackathon-uploads"

s3 = boto3.client("s3", region_name=REGION)
comprehend = boto3.client("comprehend", region_name=REGION)
rekognition = boto3.client("rekognition", region_name=REGION)

# ---- AI vs Human detection (Text) ----
def detect_ai_text(text):
    sentiment_resp = comprehend.detect_sentiment(Text=text, LanguageCode="en")
    entities_resp = comprehend.detect_entities(Text=text, LanguageCode="en")
    syntax_resp = comprehend.detect_syntax(Text=text, LanguageCode="en")

    entity_count = len(entities_resp["Entities"])
    tokens = syntax_resp["SyntaxTokens"]

    score = 0
    if entity_count < 2:
        score += 40
    if sentiment_resp["Sentiment"] == "NEUTRAL":
        score += 30
    if 80 < len(tokens) < 200:
        score += 20

    confidence = min(score, 100)
    detection = "AI-generated 🤖" if confidence >= 50 else "Human 👤"

    return {"detection": f"{detection} ({confidence}% confidence)", "sentiment": sentiment_resp["Sentiment"]}


# ---- AI vs Human detection (Image) ----
def detect_ai_image(image_bytes):
    labels = rekognition.detect_labels(
        Image={"Bytes": image_bytes},
        MaxLabels=20,
        MinConfidence=50
    )

    moderation = rekognition.detect_moderation_labels(
        Image={"Bytes": image_bytes},
        MinConfidence=50
    )

    score = 0
    label_names = [l["Name"].lower() for l in labels["Labels"]]

    ai_indicators = {"art", "illustration", "drawing", "digital", "texture", "cartoon",
                     "rendering", "painting", "graphics", "animation", "fictional character"}
    if any(x in label_names for x in ai_indicators):
        score += 50

    if moderation["ModerationLabels"]:
        score += 25

    if ("cat" in label_names and "rat" in label_names) or ("cat" in label_names and "mouse" in label_names):
        score += 20

    if ("animal" in label_names and "computer hardware" in label_names) or ("electronics" in label_names and "cat" in label_names):
        score += 20

    if len(label_names) > 12:
        score += 15

    if any(x in label_names for x in ["face", "person", "man", "woman", "beard", "skin", "wrinkle", "freckle"]):
        score -= 20

    confidence = max(10, min(score, 100))
    detection = "AI-generated 🖼️" if confidence >= 50 else "Non-AI 🖼️"

    return {
        "detection": f"{detection} ({confidence}% confidence)",
        "labels": label_names,
        "moderation": [m["Name"] for m in moderation["ModerationLabels"]]
    }


# ---- AI vs Human detection (Video) ----
def detect_ai_video(file_path, file_name, timeout_seconds=300, poll_interval=5):
    s3.upload_file(file_path, BUCKET_NAME, file_name)

    start_resp = rekognition.start_label_detection(
        Video={"S3Object": {"Bucket": BUCKET_NAME, "Name": file_name}},
        MinConfidence=50
    )
    job_id = start_resp["JobId"]

    start_time = time.time()
    while True:
        status_resp = rekognition.get_label_detection(JobId=job_id, SortBy="TIMESTAMP")
        job_status = status_resp.get("JobStatus")
        if job_status in ("SUCCEEDED", "FAILED"):
            break
        if time.time() - start_time > timeout_seconds:
            return {"detection": "Analysis timed out", "confidence": 0, "job_id": job_id}
        time.sleep(poll_interval)

    if job_status == "FAILED":
        return {"detection": "Analysis failed", "confidence": 0, "job_id": job_id}

    next_token = status_resp.get("NextToken")
    all_label_entries = status_resp.get("Labels", [])[:]
    while next_token:
        page = rekognition.get_label_detection(JobId=job_id, SortBy="TIMESTAMP", NextToken=next_token)
        all_label_entries.extend(page.get("Labels", []))
        next_token = page.get("NextToken")

    ts_labels = defaultdict(set)
    for entry in all_label_entries:
        ts = entry.get("Timestamp", 0)
        label_name = entry["Label"]["Name"].lower()
        ts_labels[ts].add(label_name)

    ordered = sorted(ts_labels.items(), key=lambda x: x[0])
    label_sets = [s for t, s in ordered]
    unique_labels = set().union(*label_sets) if label_sets else set()

    ai_indicators = {"art", "illustration", "drawing", "digital", "cartoon",
                     "rendering", "animation", "graphics", "fictional character", "painting"}
    ignored_labels = {"file", "page", "monitor", "webpage", "screen"}
    filtered_ai_hits = [l for s in label_sets for l in s if l in ai_indicators and l not in ignored_labels]
    ai_hits_total = len(filtered_ai_hits)

    def jaccard(a, b):
        if not a and not b:
            return 1.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union else 1.0

    if len(label_sets) >= 2:
        sims = [jaccard(label_sets[i], label_sets[i + 1]) for i in range(len(label_sets) - 1)]
        avg_jaccard = sum(sims) / len(sims)
    else:
        avg_jaccard = 1.0

    score = 0
    if ai_hits_total >= 5:
        score += 50
    elif ai_hits_total >= 3:
        score += 25

    variance_score = int((1.0 - avg_jaccard) * 50)
    score += variance_score

    if len(unique_labels) > 30:
        score += 20
    elif len(unique_labels) > 15:
        score += 10

    real_world_tags = {"person", "human", "car", "street", "building", "animal", "face"}
    if any(tag in unique_labels for tag in real_world_tags):
        score -= 20

    confidence = max(0, min(score, 100))
    detection = "AI-generated 🎥" if confidence >= 50 else "Non-AI 🎥"

    return {
        "detection": f"{detection} ({confidence}% confidence)",
        "confidence": confidence,
        "ai_hits_total": ai_hits_total,
        "unique_labels_count": len(unique_labels),
        "avg_jaccard": round(avg_jaccard, 3),
        "job_id": job_id
    }


# ---- Routes ----
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", active_tab="text")


@app.route("/text", methods=["POST"])
def analyze_text():
    text = request.form.get("user_text", "").strip()
    word_count = len(text.split())
    if word_count < 50:
        error_msg = f"Please enter at least 50 words. You currently typed {word_count}."
        return render_template("index.html", active_tab="text", text_error=error_msg, user_text=text)
    result = detect_ai_text(text)
    return render_template("index.html", text_result=result, active_tab="text")


@app.route("/image", methods=["POST"])
def analyze_image():
    file = request.files.get("image_file")
    if file:
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
        with open(path, "rb") as f:
            image_bytes = f.read()
        result = detect_ai_image(image_bytes)
        return render_template("index.html", image_result=result, active_tab="image")
    return redirect(url_for("home"))


@app.route("/video", methods=["POST"])
def analyze_video():
    file = request.files.get("video_file")
    if file:
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)

        # Run AI detection
        result_dict = detect_ai_video(path, filename)

        # Ensure 'detection' key exists
        if "detection" not in result_dict:
            result_dict["detection"] = f"Analysis result unavailable. Confidence: {result_dict.get('confidence', 0)}"

        # Pass the full dictionary to template
        return render_template("index.html", video_result=result_dict, active_tab="video")
    
    return redirect(url_for("home"))




if __name__ == "__main__":
    app.run(debug=True)
