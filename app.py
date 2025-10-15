import os
import pickle
import numpy as np
import streamlit as st

# Optional dependency for TF-IDF/joblib artifacts
try:
    import joblib
except Exception:
    joblib = None

# gensim imports (may be optional if you only use TF-IDF model)
try:
    from gensim.utils import simple_preprocess
    from gensim.models import Word2Vec
    gensim_available = True
except Exception:
    simple_preprocess = None
    Word2Vec = None
    gensim_available = False

st.set_page_config(page_title="Sentiment Predictor", layout="centered")

# FILES IN PROJECT ROOT (no models/ folder)
TFIDF_LR_FILENAME = "tfidf_lr.joblib"
W2V_FILENAME = "w2v.pkl"        # changed to pickle file
RF_W2V_FILENAME = "rf_w2v.pkl" # changed to pickle file

st.title("Product Review Sentiment Predictor")
st.write("Enter a product review below and pick a model to predict sentiment (Positive / Negative).")

# Determine available models (and indicate missing runtime deps)
available_models = []
if os.path.exists(TFIDF_LR_FILENAME):
    if joblib is not None:
        available_models.append("TF-IDF + LogisticRegression")
    else:
        available_models.append("TF-IDF + LogisticRegression (joblib missing)")
if os.path.exists(W2V_FILENAME) and os.path.exists(RF_W2V_FILENAME) and gensim_available:
    available_models.append("Word2Vec(avg) + RandomForest")
elif os.path.exists(W2V_FILENAME) and os.path.exists(RF_W2V_FILENAME) and not gensim_available:
    available_models.append("Word2Vec(avg) + RandomForest (gensim missing)")

if not available_models:
    st.error(
        "No saved model files found in the project root. Please ensure the files "
        f"'{TFIDF_LR_FILENAME}', '{W2V_FILENAME}', and '{RF_W2V_FILENAME}' are present next to app.py."
    )
    st.stop()

model_choice = st.sidebar.selectbox("Choose model", available_models)

# Lazy load when needed
tfidf_pipeline = None
w2v_model = None
rf_w2v = None

if model_choice.startswith("TF-IDF") and os.path.exists(TFIDF_LR_FILENAME):
    if joblib is None:
        st.error("Missing dependency: `joblib` is not installed in this environment. Install joblib to use TF-IDF model.")
        st.write("To fix this locally, run:")
        st.code("python -m pip install joblib", language="bash")
        st.stop()
    try:
        data = joblib.load(TFIDF_LR_FILENAME)
        tfidf_pipeline = data  # expected dict with 'vectorizer' and 'clf'
    except Exception as e:
        st.error(f"Failed to load TF-IDF pipeline: {e}")
        st.stop()
elif model_choice.startswith("Word2Vec"):
    if not gensim_available:
        st.error("gensim is not available in this environment. Install gensim to use Word2Vec model.")
        st.stop()
    # Load pickled Word2Vec and RF artifacts
    try:
        with open(W2V_FILENAME, "rb") as f:
            w2v_model = pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load Word2Vec pickle ({W2V_FILENAME}): {e}")
        st.stop()
    try:
        with open(RF_W2V_FILENAME, "rb") as f:
            rf_w2v = pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load RandomForest pickle ({RF_W2V_FILENAME}): {e}")
        st.stop()

st.markdown("---")
user_text = st.text_area("Paste or type a product review here:", height=150)

if st.button("Predict"):
    text = (user_text or "").strip()
    if not text:
        st.warning("Please enter a review to predict.")
    else:
        if model_choice.startswith("TF-IDF") and tfidf_pipeline is not None:
            vectorizer = tfidf_pipeline.get("vectorizer")
            clf = tfidf_pipeline.get("clf")
            if vectorizer is None or clf is None:
                st.error("TF-IDF pipeline is missing expected components ('vectorizer'/'clf').")
            else:
                X = vectorizer.transform([text])
                proba = clf.predict_proba(X)[0, 1] if hasattr(clf, "predict_proba") else None
                pred = clf.predict(X)[0]
                label = "Positive" if int(pred) == 1 else "Negative"
                st.success(f"Predicted sentiment: {label}")
                if proba is not None:
                    st.write(f"Confidence (pos probability): {proba:.3f}")
        elif model_choice.startswith("Word2Vec") and w2v_model is not None and rf_w2v is not None:
            tokens = simple_preprocess(text, deacc=True)
            # rf_w2v may be a dict if saved that way: {"clf": clf, "vector_size": size}
            vector_size = None
            if isinstance(rf_w2v, dict):
                vector_size = rf_w2v.get("vector_size", None)
            if vector_size is None:
                vector_size = getattr(w2v_model, "vector_size", None)
                if vector_size is None:
                    vector_size = getattr(getattr(w2v_model, "wv", None), "vector_size", None)

            # gensim >=4: wv.key_to_index ; fallback for older versions
            try:
                key_index = getattr(w2v_model.wv, "key_to_index", None)
                vecs = [w2v_model.wv[t] for t in tokens if key_index is None or t in key_index]
            except Exception:
                vecs = [w2v_model.wv[t] for t in tokens if t in w2v_model.wv]

            if len(vecs) == 0:
                st.warning("No in-vocabulary tokens found for this input. Cannot compute averaged embedding.")
            else:
                avg_vec = np.mean(vecs, axis=0).reshape(1, -1)
                clf = rf_w2v.get("clf") if isinstance(rf_w2v, dict) else rf_w2v
                if clf is None:
                    st.error("RandomForest classifier not found in saved artifact.")
                else:
                    proba = clf.predict_proba(avg_vec)[0, 1] if hasattr(clf, "predict_proba") else None
                    pred = clf.predict(avg_vec)[0]
                    label = "Positive" if int(pred) == 1 else "Negative"
                    st.success(f"Predicted sentiment: {label}")
                    if proba is not None:
                        st.write(f"Confidence (pos probability): {proba:.3f}")
        else:
            st.error("Model artifacts not available or failed to load.")

st.markdown("---")
st.write(
    "Tip: If this app cannot load models, ensure the model files are in the project root and that "
    "`joblib` and `gensim` are listed in requirements.txt. Note: this app expects Word2Vec and RF artifacts "
    "saved as pickles named './w2v.pkl' and './rf_w2v.pkl' (TF-IDF pipeline may still be a joblib file)."
)