from flask import Flask, render_template, request

from summarizer import summarize
from evaluation import evaluate


def load_spacy_model():
    try:
        import spacy
    except ImportError:
        return None, "spaCy is not installed. Install it with `pip install spacy`."

    try:
        return spacy.load("en_core_web_sm"), None
    except OSError:
        return None, (
            "spaCy is installed, but the model 'en_core_web_sm' is not available. "
            "Install it with `python -m spacy download en_core_web_sm`."
        )


def build_summary_result(form_data: dict) -> dict:
    text = form_data.get("text", "").strip()
    if not text:
        return {
            "error": "Please enter some text to summarize.",
            "result": None,
        }

    sentences = int(form_data.get("sentences", 3))
    similarity_threshold = float(form_data.get("similarity_threshold", 0.7))
    use_stemming = bool(form_data.get("use_stemming"))
    use_spacy = bool(form_data.get("use_spacy"))
    use_length_norm = bool(form_data.get("use_length_norm",False))

    spacy_model = None
    if use_spacy:
        spacy_model, spacy_error = load_spacy_model()
        if spacy_model is None:
            return {
                "error": spacy_error,
                "result": None,
            }

    result = summarize(
        text=text,
        n_sentences=sentences,
        use_stemming=use_stemming,
        use_spacy=spacy_model,
        length_norm=use_length_norm,
        similarity_threshold=similarity_threshold,
        use_ner_boost=True,
    )

    metrics = evaluate(result)

    return {
        "error": None,
        "result": {
            "summary": result["summary"],
            "all_sentences": result["all_sentences"],
            "all_scores": result["all_scores"],
            "selected_indices": result["selected_indices"],
            "metrics": metrics,
            "settings": {
                "sentences": sentences,
                "similarity_threshold": similarity_threshold,
                "use_stemming": use_stemming,
                "use_spacy": use_spacy,
                "use_length_norm": use_length_norm,
            },
        },
    }


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    summary_data = None
    error = None
    if request.method == "POST":
        response = build_summary_result(request.form)
        error = response["error"]
        summary_data = response["result"]

    return render_template(
        "index.html",
        summary_data=summary_data,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)
