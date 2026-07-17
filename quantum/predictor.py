def predict(features):
    entropy = float(features.get("entropy", 0.0))
    lsb_ratio = float(features.get("lsb_ratio", 0.5))
    variance = float(features.get("variance", 0.0))

    score = 0.0
    score += abs(lsb_ratio - 0.5) * 3.2
    score += max(0.0, entropy - 7.4) * 0.18
    score += min(variance / 12000, 0.25)
    score = min(score, 1.0)

    if score >= 0.46:
        return {"prediction": "Stego", "confidence": round(58 + score * 38, 2)}

    return {"prediction": "Clean", "confidence": round(92 - score * 42, 2)}