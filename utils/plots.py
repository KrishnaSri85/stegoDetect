def feature_chart_payload(features):
    return {
        "labels": [key.replace("_", " ").title() for key in features.keys()],
        "values": [float(value) for value in features.values()],
    }
