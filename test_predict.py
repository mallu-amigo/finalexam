# test_predict.py
import joblib
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import numpy as np
print("Starting model-load test...")
w2v = Word2Vec.load("w2v.model")
print("Loaded w2v.model, vector_size:", getattr(w2v, "vector_size", getattr(getattr(w2v, 'wv', None), 'vector_size', None)))
data = joblib.load("rf_w2v.joblib")
rf = data.get("clf") if isinstance(data, dict) else data
print("Loaded rf_w2v.joblib, classifier type:", type(rf))
text = "This product exceeded my expectations and works perfectly."
tokens = simple_preprocess(text, deacc=True)
print("Tokens:", tokens)
vecs = [w2v.wv[t] for t in tokens if t in w2v.wv.key_to_index]
print("In-vocab token count:", len(vecs))
if vecs:
    avg = np.mean(vecs, axis=0).reshape(1, -1)
    print("Pred:", rf.predict(avg), "Proba:", (rf.predict_proba(avg)[0,1] if hasattr(rf,'predict_proba') else None))
print("Done.")