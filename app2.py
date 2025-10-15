import os
import numpy as np
import pandas as pd
import streamlit as st

# Safe imports
try:
    import joblib
except Exception:
    joblib = None

try:
    from gensim.utils import simple_preprocess
    from gensim.models import Word2Vec
    gensim_available = True
except Exception:
    simple_preprocess = None
    Word2Vec = None
    gensim_available = False

st.set_page_config(page_title="Sentiment Predictor", layout="centered")

if joblib is None:
    st.title("Product Review Sentiment Predictor")
    st.error("Missing dependency: joblib is not installed.")
    st.code("python -m pip install joblib", language="bash")
    st.stop()

# Filenames (project root)
TFIDF_FILE = "tfidf_lr.joblib"
W2V_FILE = "w2v.model"
RF_FILE = "rf_w2v.joblib"

st.title("Product Review Sentiment Predictor")
st.write("Enter a product review and choose a model.")

available_models = []
if os.path.exists(TFIDF_FILE):
    available_models.append("TF-IDF + LogisticRegression")
if os.path.exists(W2V_FILE) and os.path.exists(RF_FILE):
    if gensim_available:
        available_models.append("Word2Vec(avg) + RandomForest")
    else:
        available_models.append("Word2Vec(avg) + RandomForest (gensim missing)")

if not available_models:
    st.error(
        "No saved models found in project root. Place tfidf_lr.joblib, w2v.model, rf_w2v.joblib alongside app.py."
    )
    st.stop()

model_choice = st.sidebar.selectbox("Choose model", available_models)
user_text = st.text_area("Paste or type a product review here:", height=150)

# Helper: explain TF-IDF + LogisticRegression predictions
def explain_tfidf_prediction(vectorizer, clf, text, top_n=10):
    """
    Returns two pandas DataFrames (top_pos, top_neg) with columns ['word', 'contribution'].
    contribution = tfidf_value * coef_for_positive_class (log-odds weight).
    """
    # transform input
    X = vectorizer.transform([text])
    # convert to dense 1D array
    if hasattr(X, "toarray"):
        X_arr = X.toarray()[0]
    else:
        X_arr = np.array(X)[0]

    # get coefficients
    if not hasattr(clf, "coef_"):
        return None, None

    # For binary classification sklearn's LogisticRegression.coef_ is shape (1, n_features)
    coefs = clf.coef_[0]
    contributions = X_arr * coefs

    # feature names: support both get_feature_names_out and legacy get_feature_names
    if hasattr(vectorizer, "get_feature_names_out"):
        feature_names = vectorizer.get_feature_names_out()
    elif hasattr(vectorizer, "get_feature_names"):
        feature_names = vectorizer.get_feature_names()
    else:
        # fallback: create indices
        feature_names = np.array([f"f{i}" for i in range(len(coefs))])

    # Only consider features that appear in the example (tfidf > 0)
    nonzero_idx = np.where(X_arr != 0)[0]
    contribs = [(feature_names[i], float(contributions[i])) for i in nonzero_idx]

    if len(contribs) == 0:
        return pd.DataFrame(columns=["word", "contribution"]), pd.DataFrame(columns=["word", "contribution"])

    # sort descending for positive contributors and ascending for negative contributors
    contribs_sorted_desc = sorted(contribs, key=lambda x: x[1], reverse=True)
    contribs_sorted_asc = sorted(contribs, key=lambda x: x[1])

    top_pos = contribs_sorted_desc[:top_n]
    top_neg = contribs_sorted_asc[:top_n]

    df_pos = pd.DataFrame(top_pos, columns=["word", "contribution"])
    df_neg = pd.DataFrame(top_neg, columns=["word", "contribution"])
    return df_pos, df_neg

# Main prediction logic: lazy-load models on demand
if st.button("Predict"):
    text = (user_text or "").strip()
    if not text:
        st.warning("Please enter a review to predict.")
    else:
        with st.spinner("Loading model and predicting..."):
            try:
                if model_choice.startswith("TF-IDF"):
                    # load pipeline artifact (expect dict with 'vectorizer' and 'clf' or a pipeline)
                    data = joblib.load(TFIDF_FILE)
                    # Support two common artifact shapes:
                    # 1) dict: {'vectorizer': vectorizer, 'clf': clf}
                    # 2) sklearn Pipeline (with named steps)
                    vectorizer = None
                    clf = None
                    if isinstance(data, dict):
                        vectorizer = data.get("vectorizer")
                        clf = data.get("clf")
                    else:
                        # try pipeline-like object
                        try:
                            # pipeline has .named_steps or we try to find a vectorizer and clf attributes
                            if hasattr(data, "named_steps"):
                                steps = data.named_steps
                                # common names: 'tfidf' or 'vectorizer' and 'clf' or 'classifier'
                                for name, step in steps.items():
                                    from sklearn.feature_extraction.text import TfidfVectorizer
                                    if isinstance(step, TfidfVectorizer) or hasattr(step, "transform"):
                                        vectorizer = step
                                    # classifier detection
                                    from sklearn.base import ClassifierMixin
                                    if isinstance(step, ClassifierMixin) or hasattr(step, "predict"):
                                        clf = step
                        except Exception:
                            pass

                    if vectorizer is None or clf is None:
                        st.error("Could not find 'vectorizer' and 'clf' in TF-IDF artifact. Ensure you saved them as a dict with keys 'vectorizer' and 'clf' or as a sklearn Pipeline.")
                    else:
                        # transform and predict
                        X = vectorizer.transform([text])
                        pred = clf.predict(X)[0]
                        proba = clf.predict_proba(X)[0, 1] if hasattr(clf, "predict_proba") else None
                        label = "Positive" if int(pred) == 1 else "Negative"
                        st.success(f"Predicted: {label}")
                        if proba is not None:
                            st.write(f"Confidence (pos probability): {proba:.3f}")

                        # Explanation: show top contributing words (if logistic regression)
                        if hasattr(clf, "coef_"):
                            df_pos, df_neg = explain_tfidf_prediction(vectorizer, clf, text, top_n=10)
                            st.markdown("### Explanation — Top contributing words")
                            st.write("Positive contributors (push prediction towards Positive):")
                            if not df_pos.empty:
                                st.table(df_pos)
                                # bar chart for visualization
                                st.bar_chart(df_pos.set_index("word")["contribution"])
                            else:
                                st.write("No in-vocabulary tokens found that contributed positively.")

                            st.write("Negative contributors (push prediction towards Negative):")
                            if not df_neg.empty:
                                st.table(df_neg)
                                st.bar_chart(df_neg.set_index("word")["contribution"])
                            else:
                                st.write("No in-vocabulary tokens found that contributed negatively.")
                        else:
                            st.info("Model does not expose coefficients; explanation for TF-IDF model is unavailable.")
                else:
                    # Word2Vec + RF branch
                    if not gensim_available:
                        st.error("gensim is not installed in this environment.")
                        st.stop()
                    w2v_model = Word2Vec.load(W2V_FILE)
                    rf_data = joblib.load(RF_FILE)
                    clf = rf_data.get("clf") if isinstance(rf_data, dict) else rf_data
                    tokens = simple_preprocess(text, deacc=True)
                    vecs = [w2v_model.wv[t] for t in tokens if t in w2v_model.wv.key_to_index]
                    if len(vecs) == 0:
                        st.warning("No in-vocabulary tokens found for this input. Cannot compute averaged embedding.")
                    else:
                        avg_vec = np.mean(vecs, axis=0).reshape(1, -1)
                        proba = clf.predict_proba(avg_vec)[0, 1] if hasattr(clf, "predict_proba") else None
                        pred = clf.predict(avg_vec)[0]
                        label = "Positive" if int(pred) == 1 else "Negative"
                        st.success(f"Predicted sentiment: {label}")
                        if proba is not None:
                            st.write(f"Confidence (pos probability): {proba:.3f}")

                        # Note: for RandomForest you can show feature importances only if you trained on fixed features.
                        # For Word2Vec averaged vectors there isn't a direct human-readable feature mapping.
                        st.info("Explanation for Word2Vec+RF: consider using permutation importance or SHAP for per-prediction explanations (not implemented here).")
            except Exception as e:
                st.error("Prediction failed: " + str(e))