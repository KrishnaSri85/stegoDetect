import os


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "quantum-stego-dev-key")
    MAX_CONTENT_LENGTH = 8 * 1024 * 1024
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff"}
    UPLOAD_FOLDER = os.path.join("static", "uploads")
    RESULT_FOLDER = os.path.join("static", "results")
    REPORT_MODEL_NAME = "Qiskit Quantum Classifier"
    LOG_FILE = os.path.join("logs", "app.log")
