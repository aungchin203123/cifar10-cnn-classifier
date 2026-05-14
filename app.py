# -*- coding: utf-8 -*-
import os
import io
import glob
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

# Auto-download model if not exists
MODEL_PATH = "cifar10_model_88.keras"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model file (30MB)... Please wait..."):
        import urllib.request
        url = "https://github.com/aungchin203123/cifar10-cnn-classifier/releases/download/v1/cifar10_model_88.keras"
        try:
            urllib.request.urlretrieve(url, MODEL_PATH)
            st.success("Model downloaded successfully!")
        except:
            st.warning("Could not download model. Using fallback mode.")
            MODEL_PATH = None

import keras

st.set_page_config(
    page_title="CIFAR-10 Classifier",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

CLASS_NAMES_DISPLAY = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck",
]
CLASS_EMOJIS = ["✈️", "🚗", "🐦", "🐱", "🦌", "🐶", "🐸", "🐴", "🚢", "🚛"]
CLASS_KEYS = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
CLASS_COLORS = ["#4361EE","#F72585","#7209B7","#3A0CA3","#4CC9F0","#4895EF","#560BAD","#B5179E","#06D6A0","#FFB703"]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
html,body,[class*="css"]{font-family:'Syne',sans-serif;}
.stApp{background:#0a0a12;color:#e8e8f0;}
section[data-testid="stSidebar"]{background:#10101e!important;border-right:1px solid #2a2a3e;}
.card{background:#14141f;border:1px solid #2a2a3e;border-radius:12px;padding:20px 24px;margin-bottom:16px;}
.pred-badge{display:inline-block;background:linear-gradient(135deg,#4361EE,#7209B7);color:white;font-size:1.4rem;font-weight:800;padding:10px 22px;border-radius:50px;margin:8px 0;}
.conf-row{display:flex;align-items:center;margin-bottom:6px;gap:10px;}
.conf-label{width:110px;font-size:0.82rem;color:#a0a0c0;}
.conf-bar-wrap{flex:1;background:#1e1e30;border-radius:4px;height:14px;overflow:hidden;}
.conf-bar{height:100%;border-radius:4px;}
.conf-pct{width:46px;text-align:right;font-family:'Space Mono',monospace;font-size:0.78rem;color:#c0c0e0;}
.metric-tile{background:#14141f;border:1px solid #2a2a3e;border-radius:10px;padding:16px;text-align:center;}
.metric-val{font-size:2rem;font-weight:800;color:#4361EE;}
.metric-lbl{font-size:0.78rem;color:#808090;margin-top:2px;}
.stButton>button{background:linear-gradient(135deg,#4361EE,#7209B7);color:white;border:none;border-radius:8px;font-weight:700;padding:0.5rem 1.5rem;}
h1,h2,h3{font-family:'Syne',sans-serif!important;font-weight:800!important;}
.stTabs [data-baseweb="tab"]{font-weight:700;color:#606080;}
.stTabs [aria-selected="true"]{color:#4361EE!important;border-bottom-color:#4361EE!important;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_my_model(path):
    if not path or not os.path.exists(path):
        return None, None
    try:
        import keras as _keras
        _orig = _keras.layers.Dense.__init__
        def _patched(self, *a, **kw):
            kw.pop("quantization_config", None)
            _orig(self, *a, **kw)
        _keras.layers.Dense.__init__ = _patched
        m = _keras.models.load_model(path, compile=False)
        _keras.layers.Dense.__init__ = _orig
        return m, path
    except Exception:
        pass
    try:
        import tensorflow as tf
        return tf.keras.models.load_model(path, compile=False), path
    except Exception as e:
        return None, str(e)

def preprocess(img):
    img = img.convert("RGB").resize((32, 32), Image.LANCZOS)
    return np.expand_dims(np.array(img, np.float32) / 255.0, 0)

def do_predict(model, arr):
    p = model.predict(arr, verbose=0)[0]
    return p, int(np.argmax(p))

def conf_html(probs):
    html = ""
    for i in np.argsort(probs)[::-1]:
        pct = probs[i] * 100
        bold = "font-weight:700;color:#e8e8f0;" if i == np.argmax(probs) else ""
        label = CLASS_EMOJIS[i] + " " + CLASS_NAMES_DISPLAY[i]
        html += (
            f'<div class="conf-row">'
            f'<div class="conf-label" style="{bold}">{label}</div>'
            f'<div class="conf-bar-wrap"><div class="conf-bar" style="width:{pct:.1f}%;background:{CLASS_COLORS[i]};"></div></div>'
            f'<div class="conf-pct">{pct:.1f}%</div>'
            f'</div>'
        )
    return html

def bar_chart(probs):
    fig, ax = plt.subplots(figsize=(7, 3))
    fig.patch.set_facecolor("#14141f")
    ax.set_facecolor("#14141f")
    ax.bar(CLASS_KEYS, probs * 100, color=CLASS_COLORS, width=0.6, zorder=3)
    ax.set_ylabel("Probability (%)", color="#a0a0c0", fontsize=9)
    ax.set_ylim(0, 105)
    ax.tick_params(colors="#a0a0c0", labelsize=8)
    for s in ax.spines.values():
        s.set_edgecolor("#2a2a3e")
    ax.yaxis.grid(True, color="#2a2a3e", zorder=0)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    return fig

# Sidebar
with st.sidebar:
    st.markdown("## CIFAR-10\n### CNN Classifier")
    st.markdown("---")
    st.markdown("**Classes**")
    for emoji, name, color in zip(CLASS_EMOJIS, CLASS_NAMES_DISPLAY, CLASS_COLORS):
        st.markdown(f'<span style="color:{color};font-size:0.9rem;">{emoji} {name}</span>', unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Built with Streamlit + Keras")

with st.spinner("Loading model..."):
    model, model_path_used = load_my_model(MODEL_PATH)

# Header
st.markdown("# CIFAR-10 Image Classifier")
st.markdown("VGG-style CNN · 10 classes · 32x32 input")

c1, c2, c3, c4 = st.columns(4)
if model is not None:
    c1.markdown('<div class="metric-tile"><div class="metric-val">OK</div><div class="metric-lbl">Model loaded</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-tile"><div class="metric-val">{len(model.layers)}</div><div class="metric-lbl">Layers</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-tile"><div class="metric-val">{model.count_params()/1e6:.2f}M</div><div class="metric-lbl">Parameters</div></div>', unsafe_allow_html=True)
    c4.markdown('<div class="metric-tile"><div class="metric-val">88%</div><div class="metric-lbl">Val Accuracy</div></div>', unsafe_allow_html=True)
else:
    st.error("Model not available. Please check the model file.")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["Single Predict", "Batch Predict", "Explore Dataset", "Architecture"])

# Tab 1 - Single Predict
with tab1:
    col_l, col_r = st.columns(2, gap="large")
    with col_l:
        st.markdown("### Upload an Image")
        up = st.file_uploader("PNG / JPG / WEBP - any size", type=["png","jpg","jpeg","webp"], key="single")
        if up:
            pil = Image.open(up)
            st.image(pil, caption="Original", use_container_width=True)
            st.image(pil.resize((32,32), Image.NEAREST).resize((128,128), Image.NEAREST), caption="As model sees it (32x32)", width=128)
    with col_r:
        st.markdown("### Prediction")
        if up and model is not None:
            with st.spinner("Classifying..."):
                probs, idx = do_predict(model, preprocess(pil))
            label = CLASS_EMOJIS[idx] + " " + CLASS_NAMES_DISPLAY[idx]
            st.markdown(f'<div class="pred-badge">{label}</div>', unsafe_allow_html=True)
            st.markdown(f"**Confidence:** `{probs[idx]*100:.2f}%`")
            st.markdown("#### All class probabilities")
            st.markdown(f'<div class="card">{conf_html(probs)}</div>', unsafe_allow_html=True)
            fig = bar_chart(probs)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        elif model is None:
            st.info("Model not loaded. Please check the model file.")
        else:
            st.info("Upload an image on the left.")

# Tab 2 - Batch Predict
with tab2:
    st.markdown("### Batch Prediction")
    files = st.file_uploader("Upload images", type=["png","jpg","jpeg","webp"], accept_multiple_files=True, key="batch")
    if files and model is not None:
        if st.button("Run Batch Prediction"):
            results = []
            prog = st.progress(0)
            cols = st.columns(min(len(files), 5))
            for i, f in enumerate(files):
                pil = Image.open(f)
                probs, idx = do_predict(model, preprocess(pil))
                label = CLASS_EMOJIS[idx] + " " + CLASS_NAMES_DISPLAY[idx]
                top2 = CLASS_NAMES_DISPLAY[int(np.argsort(probs)[-2])]
                top3 = CLASS_NAMES_DISPLAY[int(np.argsort(probs)[-3])]
                results.append({
                    "File": f.name,
                    "Prediction": label,
                    "Confidence": f"{probs[idx]*100:.1f}%",
                    "Top-2": top2,
                    "Top-3": top3,
                })
                cols[i % 5].image(pil, caption=label, use_container_width=True)
                prog.progress((i + 1) / len(files))
            prog.empty()
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)
            st.download_button("Download CSV", df.to_csv(index=False).encode(), "predictions.csv", "text/csv")
    elif model is None:
        st.info("Load a model first.")

# Tab 3 - Explore Dataset
with tab3:
    st.markdown("### Explore CIFAR-10 Test Set")
    if model is not None:
        if st.button("Sample 50 random test images"):
            with st.spinner("Loading CIFAR-10 & predicting..."):
                (_, _), (x_test, y_test) = keras.datasets.cifar10.load_data()
                x_test = x_test.astype(np.float32) / 255.0
                si = np.random.choice(len(x_test), 50, replace=False)
                imgs, labels = x_test[si], y_test[si].flatten()
                preds = np.argmax(model.predict(imgs, verbose=0), axis=1)
            correct = int((preds == labels).sum())
            st.markdown(
                f'<div class="card"><b>Sample accuracy:</b> '
                f'<span style="color:#06D6A0;font-size:1.3rem;">{correct/50*100:.0f}%</span> '
                f'({correct}/50 correct)</div>',
                unsafe_allow_html=True,
            )
            fig, axes = plt.subplots(5, 10, figsize=(14, 7))
            fig.patch.set_facecolor("#0a0a12")
            for ax, img, true, pred in zip(axes.flat, imgs, labels, preds):
                ax.imshow(img)
                color = "#06D6A0" if true == pred else "#F72585"
                ax.set_title(CLASS_KEYS[pred], fontsize=6, color=color, pad=2)
                for sp in ax.spines.values():
                    sp.set_edgecolor(color)
                    sp.set_linewidth(2)
                ax.set_xticks([])
                ax.set_yticks([])
            plt.suptitle("Green=correct  Pink=wrong", color="#e8e8f0", fontsize=9, y=1.01)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
    else:
        st.info("Load a model first.")

# Tab 4 - Architecture
with tab4:
    st.markdown("### Model Architecture")
    if model is not None:
        buf = io.StringIO()
        model.summary(print_fn=lambda x: buf.write(x + "\n"))
        st.code(buf.getvalue(), language="text")
        rows = [
            {
                "Layer": l.name,
                "Type": l.__class__.__name__,
                "Output Shape": str(getattr(l, "output_shape", "-")),
                "Params": f"{l.count_params():,}",
            }
            for l in model.layers
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=400)
    else:
        st.info("Load a model first.")