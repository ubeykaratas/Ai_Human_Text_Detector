import os
import joblib
from django.shortcuts import render
from django.conf import settings

MODEL_PATHS = {
    "naive_bayes": "naive_bayes.pkl",
    "random_forest": "random_forest.pkl",
    "linear_svc": "linear_svc.pkl",
    "logistic_regression": "logistic_regression.pkl",
    "gradient_boosting": "gradient_boosting.pkl",
}

def index(request):
    result = None
    if request.method == "POST":
        input_text = request.POST.get("input_text")
        selected_model = request.POST.get("model")

        if input_text and selected_model in MODEL_PATHS:
            vectorizer_path = os.path.join(settings.BASE_DIR, "detector", "models", "tfidf_vectorizer.pkl")
            model_path = os.path.join(settings.BASE_DIR, "detector", "models", MODEL_PATHS[selected_model])
            selector_path = os.path.join(settings.BASE_DIR, "detector", "models", "select_kbest.pkl")

            vectorizer = joblib.load(vectorizer_path)
            model = joblib.load(model_path)
            selector = joblib.load(selector_path)

            vectorized = vectorizer.transform([input_text])
            selected_features = selector.transform(vectorized)
            prediction = model.predict(selected_features)[0]

            # Calculate confidence score
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(selected_features)[0]
                confidence = round(max(probabilities) * 100, 2)
            else:
                # Some models like SVC may not support predict_proba
                confidence = "Unknown"

            if prediction == 1:
                result = f"AI ({confidence}% confidence)"
            else:
                result = f"Human ({confidence}% confidence)"

    return render(request, "detector/index.html", {"result": result})