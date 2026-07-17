import json
import logging
import os
import time
from datetime import datetime
from uuid import uuid4

from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from config import Config

from detection.report import generate_detection_report

from utils.helper import (
    allowed_file,
    extract_features_safe,
    predict_safe,
    readable_feature_summary,
    validate_image_file,
)


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    for path in (
        app.config["UPLOAD_FOLDER"],
        app.config["RESULT_FOLDER"],
        os.path.dirname(app.config["LOG_FILE"]),
    ):
        os.makedirs(path, exist_ok=True)

    logging.basicConfig(
        filename=app.config["LOG_FILE"],
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/upload", methods=["GET", "POST"])
    def upload():
        if request.method == "GET":
            return render_template("upload.html")

        if "image" not in request.files:
            return render_template("upload.html", error="Please choose an image before analyzing."), 400

        image = request.files["image"]
        if not image or not image.filename:
            return render_template("upload.html", error="Please choose an image before analyzing."), 400

        if not allowed_file(image.filename, app.config["ALLOWED_EXTENSIONS"]):
            return render_template("upload.html", error="Unsupported image type. Try PNG, JPG, BMP, WEBP, or TIFF."), 400

        result, status = analyze_uploaded_file(image)
        if status != 200:
            return render_template("upload.html", error=result["error"]), status

        return redirect(url_for("result", result_id=result["result_id"]))

    @app.route("/api/analyze", methods=["POST"])
    def api_analyze():
        if "image" not in request.files:
            return jsonify({"error": "Missing image file."}), 400

        image = request.files["image"]
        result, status = analyze_uploaded_file(image)
        return jsonify(result), status

    @app.route("/result/<result_id>")
    def result(result_id):
        data = load_result(result_id)
        if not data:
            return render_template("upload.html", error="That result could not be found. Please analyze the image again."), 404
        return render_template("result.html", result=data)

    @app.route("/report/<result_id>")
    def report(result_id):
        data = load_result(result_id)
        if not data:
            return jsonify({"error": "Result not found."}), 404

        report_path = generate_detection_report(data, app.config["RESULT_FOLDER"])
        return send_file(report_path, as_attachment=True)

    @app.route("/about")
    def about():
        return render_template("about.html")

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        message = "Image is too large. Please upload a file smaller than 8 MB."
        if request.path.startswith("/api/"):
            return jsonify({"error": message}), 413
        return render_template("upload.html", error=message), 413

    @app.errorhandler(500)
    def internal_error(_error):
        logging.exception("Internal server error")
        message = "Something went wrong while processing the image. Please try again."
        if request.path.startswith("/api/"):
            return jsonify({"error": message}), 500
        return render_template("upload.html", error=message), 500

    def analyze_uploaded_file(image):
        start_time = time.perf_counter()
        original_name = secure_filename(image.filename)

        if not original_name:
            return {"error": "Invalid file name."}, 400
        if not allowed_file(original_name, app.config["ALLOWED_EXTENSIONS"]):
            return {"error": "Unsupported image type. Try PNG, JPG, BMP, WEBP, or TIFF."}, 400

        result_id = uuid4().hex
        ext = original_name.rsplit(".", 1)[1].lower()
        stored_name = f"{result_id}.{ext}"
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], stored_name)

        try:
            image.save(image_path)
            validate_image_file(image_path)
            logging.info("Upload accepted: %s -> %s", original_name, image_path)

            features = extract_features_safe(image_path)
            prediction = predict_safe(features)
        except ValueError as exc:
            logging.warning("Upload validation failed for %s: %s", original_name, exc)
            cleanup_file(image_path)
            return {"error": str(exc)}, 400
        except RuntimeError as exc:
            logging.exception("Pipeline failure for %s", original_name)
            cleanup_file(image_path)
            return {"error": str(exc)}, 502
        except Exception:
            logging.exception("Unexpected analysis failure for %s", original_name)
            cleanup_file(image_path)
            return {"error": "Unable to analyze this image. Please try another file."}, 500

        elapsed = time.perf_counter() - start_time
        confidence = round(float(prediction.get("confidence", 0.0)), 2)
        prediction_label = str(prediction.get("prediction", "Unknown")).title()
        status_label = "Suspicious" if prediction_label.lower() == "stego" else "No Steganography Detected"

        data = {
            "result_id": result_id,
            "image_name": original_name,
            "image_url": url_for("static", filename=f"uploads/{stored_name}"),
            "stored_image_path": image_path,
            "prediction": prediction_label,
            "confidence": confidence,
            "features": features,
            "feature_summary": readable_feature_summary(features),
            "processing_time": f"{elapsed:.2f} sec",
            "processing_seconds": round(elapsed, 4),
            "status": status_label,
            "model": app.config["REPORT_MODEL_NAME"],
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        result_path = os.path.join(app.config["RESULT_FOLDER"], f"{result_id}.json")
        with open(result_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)

        logging.info(
            "Prediction complete: %s prediction=%s confidence=%.2f elapsed=%.2fs",
            original_name,
            prediction_label,
            confidence,
            elapsed,
        )
        return data, 200

    def load_result(result_id):
        safe_id = secure_filename(result_id)
        result_path = os.path.join(app.config["RESULT_FOLDER"], f"{safe_id}.json")
        if not os.path.exists(result_path):
            return None
        with open(result_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def cleanup_file(path):
        if path and os.path.exists(path):
            os.remove(path)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
