# Quantum-Assisted Steganography Detection Web App

Member 3 module for the frontend, Flask backend, system integration, report generation, logging, and UI polish.

## Run

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

## Integration Contracts

The backend calls only public functions from the other modules:

```python
extract_features(image_path)
predict(features)
```

If `feature_extraction` or `quantum` is not present, deterministic fallback functions keep the interface runnable for UI and integration testing. Once Member 1 and Member 2 modules are available, place them on the Python path and the app will call them automatically.

## Included

- Upload validation and image storage
- Feature extraction and quantum prediction integration
- Result page with prediction, confidence, features, status, and processing time
- Downloadable PDF report and JSON report artifact
- Dark responsive UI with image preview, drag-and-drop, loader, and error messages
- Application logging in `logs/app.log`
