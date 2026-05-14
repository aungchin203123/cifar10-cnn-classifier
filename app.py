"""
CIFAR-10 Image Classifier — Streamlit App
Mirrors: https://cifar-10-cnn-with-keras.streamlit.app/
"""

import io
import os
import pathlib
import numpy as np
import streamlit as st
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CIFAR-10 Classifier",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — clean dark-tech aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Space Mono', monospace;
    }

    /* Background */
    .stApp {
        background: linear-gradient(135deg, #0f1117 0%, #1a1d2e 100%);
        color: #e8eaf0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #13151f !important;
        border-right: 1px solid #2a2d3e;
    }

    /* Cards */
    .card {
        background: #1e2130;
        border: 1px solid #2a2d3e;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    /* Prediction bar */
    .pred-bar-wrap {
        margin: 4px 0;
    }
    .pred-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.85rem;
        margin-bottom: 3px;
        color: #c8cad8;
    }
    .pred-bar-bg {
        background: #2a2d3e;
        border-radius: 6px;
        height: 10px;
        overflow: hidden;
    }
    .pred-bar-fill {
        height: 100%;
        border-radius: 6px;
        background: linear-gradient(90deg, #4f8ef7, #a78bfa);
        transition: width 0.4s ease;
    }
    .pred-bar-fill-top {
        background: linear-gradient(90deg, #34d399, #4f8ef7);
    }

    /* Badge */
    .badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .badge-blue  { background:#1d3461; color:#4f8ef7; border:1px solid #4f8ef7; }
    .badge-green { background:#0d3d2e; color:#34d399; border:1px solid #34d399; }

    /* Metric boxes */
    .metric-box {
        background: #1e2130;
        border: 1px solid #2a2d3e;
        border-radius: 10px;
        padding: 0.8rem 1.2rem;
        text-align: center;
    }
    .metric-val { font-size: 1.8rem; font-weight: 700; color: #4f8ef7; }
    .metric-lbl { font-size: 0.75rem; color: #888; margin-top: 2px; letter-spacing: 0.4px; }

    /* Upload zone */
    [data-testid="stFileUploader"] {
        background: #1e2130 !important;
        border: 2px dashed #2a2d3e !important;
        border-radius: 12px !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #4f8ef7, #a78bfa) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 0.85rem !important;
        padding: 0.5rem 1.4rem !important;
    }

    .divider { border-top: 1px solid #2a2d3e; margin: 1.2rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
CLASSES = [
    "✈️ Airplane", "🚗 Automobile", "🐦 Bird", "🐱 Cat", "🦌 Deer",
    "🐶 Dog", "🐸 Frog", "🐴 Horse", "🚢 Ship", "🚛 Truck",
]
CLASS_NAMES = [c.split(" ", 1)[1] for c in CLASSES]   # plain names for model
MODEL_PATH = "improved_cifar10_cnn.keras"

# ─────────────────────────────────────────────────────────────────────────────
# Model loader (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """Load the trained Keras model, or return None if not found."""
    try:
        import tensorflow as tf
        if os.path.exists(MODEL_PATH):
            return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.warning(f"Model load error: {e}")
    return None


@st.cache_resource(show_spinner=False)
def get_demo_predictions():
    """Return precomputed demo predictions (when no model is loaded)."""
    demo = {
        "frog":  [0.01, 0.01, 0.04, 0.02, 0.04, 0.03, 0.82, 0.01, 0.01, 0.01],
        "horse": [0.01, 0.01, 0.02, 0.01, 0.05, 0.04, 0.02, 0.82, 0.01, 0.01],
        "ship":  [0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.88, 0.02],
    }
    return demo


def preprocess(img: Image.Image) -> np.ndarray:
    """Resize to 32×32 and prepare a batch."""
    img = img.convert("RGB").resize((32, 32), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)   # model normalises internally
    return np.expand_dims(arr, 0)           # (1, 32, 32, 3)


def predict(model, img: Image.Image):
    """Return (top_idx, top_conf, probs_array)."""
    x = preprocess(img)
    probs = model.predict(x, verbose=0)[0]
    top_idx = int(np.argmax(probs))
    return top_idx, float(probs[top_idx]), probs


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 CIFAR-10\nClassifier")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("**About the model**")
    st.markdown(
        """
        <div class="card" style="font-size:0.85rem; line-height:1.7;">
        Architecture: Deep CNN<br>
        Blocks: 3× (Conv→Conv→Pool→Drop)<br>
        BatchNorm after every conv<br>
        Data augmentation during training<br>
        Callbacks: EarlyStopping + ReduceLR
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**10 Classes**")
    cols = st.columns(2)
    for i, c in enumerate(CLASSES):
        cols[i % 2].markdown(f"<small>{c}</small>", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("**Model Metrics**")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            '<div class="metric-box"><div class="metric-val">86%</div>'
            '<div class="metric-lbl">Test Accuracy</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="metric-box"><div class="metric-val">10</div>'
            '<div class="metric-lbl">Classes</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.caption("Built from cifar10_improve_aml13.ipynb")

# ─────────────────────────────────────────────────────────────────────────────
# Main area
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='margin-bottom:4px;'>CIFAR-10 Image Classifier</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    '<span class="badge badge-blue">CNN</span>&nbsp;'
    '<span class="badge badge-green">~86% Accuracy</span>&nbsp;'
    '<span class="badge badge-blue">Keras / TensorFlow</span>',
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)

model = load_model()
model_loaded = model is not None

if not model_loaded:
    st.info(
        "⚠️ **Trained model not found** (`improved_cifar10_cnn.keras`).  \n"
        "Run `python train_model.py` to train and save the model, then restart the app.  \n"
        "Until then, a **demo mode** with a sample CIFAR-10 test image is shown.",
        icon="ℹ️",
    )

# ── Upload or demo ────────────────────────────────────────────────────────────
tab_upload, tab_demo, tab_architecture, tab_about = st.tabs(
    ["📤 Upload Image", "🎯 Demo", "🏗️ Architecture", "📖 About"]
)

# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Upload
# ─────────────────────────────────────────────────────────────────────────────
with tab_upload:
    st.markdown("### Upload an image to classify")
    st.caption(
        "Upload any image. It will be resized to 32×32 for the model — "
        "works best with clear subject photos matching the CIFAR-10 classes."
    )

    uploaded = st.file_uploader(
        "Choose an image…", type=["jpg", "jpeg", "png", "webp", "bmp"]
    )

    if uploaded:
        img = Image.open(uploaded)
        col_img, col_results = st.columns([1, 2], gap="large")

        with col_img:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.image(img, caption="Uploaded image", use_container_width=True)
            thumb = img.convert("RGB").resize((32, 32), Image.LANCZOS)
            st.image(thumb, caption="32×32 model input", width=96)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_results:
            if model_loaded:
                with st.spinner("Running inference…"):
                    top_idx, top_conf, probs = predict(model, img)

                st.markdown(
                    f"<div class='card'>"
                    f"<div style='font-size:0.8rem;color:#888;margin-bottom:4px;'>TOP PREDICTION</div>"
                    f"<div style='font-size:2rem;font-weight:700;'>{CLASSES[top_idx]}</div>"
                    f"<div style='font-size:1rem;color:#34d399;margin-top:4px;'>"
                    f"Confidence: <strong>{top_conf*100:.1f}%</strong></div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                st.markdown("**All class probabilities**")
                order = np.argsort(probs)[::-1]
                for rank, idx in enumerate(order):
                    fill_class = "pred-bar-fill-top" if rank == 0 else "pred-bar-fill"
                    pct = probs[idx] * 100
                    st.markdown(
                        f"<div class='pred-bar-wrap'>"
                        f"<div class='pred-label'><span>{CLASSES[idx]}</span>"
                        f"<span>{pct:.1f}%</span></div>"
                        f"<div class='pred-bar-bg'>"
                        f"<div class='{fill_class}' style='width:{pct:.1f}%'></div>"
                        f"</div></div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.warning("Train the model first to enable live predictions.")

# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Demo
# ─────────────────────────────────────────────────────────────────────────────
with tab_demo:
    st.markdown("### Sample CIFAR-10 Images (from test set)")

    # Load CIFAR-10 test images for demo
    @st.cache_data(show_spinner=False)
    def load_cifar_samples(n=20):
        import tensorflow as tf
        (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        y_test = y_test.flatten()
        indices = []
        for c in range(10):
            idxs = np.where(y_test == c)[0]
            indices.append(np.random.default_rng(42).choice(idxs, 2, replace=False))
        indices = np.concatenate(indices)
        return x_test[indices], y_test[indices]

    with st.spinner("Loading sample images…"):
        try:
            x_demo, y_demo = load_cifar_samples()
            demo_loaded = True
        except Exception:
            demo_loaded = False

    if demo_loaded:
        cols = st.columns(10)
        selected_idx = st.session_state.get("demo_selected", 0)

        # Grid of thumbnails
        for i, col in enumerate(cols):
            with col:
                pil_img = Image.fromarray(x_demo[i])
                if st.button(CLASS_NAMES[y_demo[i]], key=f"demo_btn_{i}"):
                    st.session_state["demo_selected"] = i
                st.image(pil_img, use_container_width=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        sel = st.session_state.get("demo_selected", 0)
        sel_img = Image.fromarray(x_demo[sel])
        true_label = CLASS_NAMES[y_demo[sel]]

        col_a, col_b = st.columns([1, 2], gap="large")
        with col_a:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.image(sel_img.resize((128, 128), Image.NEAREST), caption=f"True: {true_label}")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_b:
            if model_loaded:
                top_idx, top_conf, probs = predict(model, sel_img)
                correct = CLASS_NAMES[top_idx] == true_label
                badge_html = (
                    '<span class="badge badge-green">✓ Correct</span>'
                    if correct
                    else '<span class="badge" style="background:#3d1a1a;color:#f87171;border:1px solid #f87171;">✗ Incorrect</span>'
                )
                st.markdown(
                    f"<div class='card'>"
                    f"<div style='font-size:0.8rem;color:#888;'>PREDICTION</div>"
                    f"<div style='font-size:1.8rem;font-weight:700;'>{CLASSES[top_idx]}</div>"
                    f"<div style='margin-top:6px;'>{badge_html}</div>"
                    f"<div style='margin-top:4px;font-size:0.9rem;color:#34d399;'>Confidence: {top_conf*100:.1f}%</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                order = np.argsort(probs)[::-1]
                for rank, idx in enumerate(order[:5]):
                    fill_class = "pred-bar-fill-top" if rank == 0 else "pred-bar-fill"
                    pct = probs[idx] * 100
                    st.markdown(
                        f"<div class='pred-bar-wrap'>"
                        f"<div class='pred-label'><span>{CLASSES[idx]}</span>"
                        f"<span>{pct:.1f}%</span></div>"
                        f"<div class='pred-bar-bg'>"
                        f"<div class='{fill_class}' style='width:{pct:.1f}%'></div>"
                        f"</div></div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.info("Run `python train_model.py` to enable live predictions on demo images.")

# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Architecture
# ─────────────────────────────────────────────────────────────────────────────
with tab_architecture:
    st.markdown("### Model Architecture")

    layers_info = [
        ("Input", "32 × 32 × 3", "RGB image"),
        ("Rescaling", "÷255", "Normalise to [0, 1]"),
        ("Conv2D 64 + BN + ReLU", "32 × 32 × 64", "Block 1"),
        ("Conv2D 64 + BN + ReLU", "32 × 32 × 64", "Block 1"),
        ("MaxPool2D + Dropout(0.3)", "16 × 16 × 64", "Block 1"),
        ("Conv2D 128 + BN + ReLU", "16 × 16 × 128", "Block 2"),
        ("Conv2D 128 + BN + ReLU", "16 × 16 × 128", "Block 2"),
        ("MaxPool2D + Dropout(0.4)", "8 × 8 × 128", "Block 2"),
        ("Conv2D 256 + BN + ReLU", "8 × 8 × 256", "Block 3"),
        ("Conv2D 256 + BN + ReLU", "8 × 8 × 256", "Block 3"),
        ("MaxPool2D + Dropout(0.5)", "4 × 4 × 256", "Block 3"),
        ("GlobalAveragePooling2D", "256", "Head"),
        ("Dense 512 + BN + ReLU + Dropout(0.5)", "512", "Head"),
        ("Dense 10 + Softmax", "10", "Output"),
    ]

    for layer, shape, block in layers_info:
        st.markdown(
            f"<div class='card' style='padding:0.6rem 1.2rem;margin-bottom:6px;'>"
            f"<span style='font-family:Space Mono,monospace;font-size:0.85rem;color:#4f8ef7;'>{layer}</span>"
            f"&nbsp;&nbsp;<span style='font-size:0.78rem;color:#888;'>→ {shape}</span>"
            f"&nbsp;&nbsp;<span class='badge badge-blue' style='font-size:0.65rem;'>{block}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### Training Techniques")

    techniques = {
        "🔄 Data Augmentation": "Random flip, crop (+4px pad), brightness, contrast, saturation — reduces overfitting",
        "📏 Batch Normalisation": "After every Conv layer — stabilises training, speeds convergence",
        "💧 Dropout": "0.3 → 0.4 → 0.5 (increasing depth) — strong regularisation",
        "⏹️ EarlyStopping": "Monitors val_accuracy with patience=15, restores best weights",
        "📉 ReduceLROnPlateau": "Halves LR when val_loss stalls (patience=5, min_lr=1e-6)",
        "🌐 GlobalAveragePooling": "Reduces feature maps to vector without flattening — fewer params",
    }
    c1, c2 = st.columns(2)
    for i, (name, desc) in enumerate(techniques.items()):
        col = c1 if i % 2 == 0 else c2
        col.markdown(
            f"<div class='card'><strong>{name}</strong><br>"
            f"<small style='color:#aaa;'>{desc}</small></div>",
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 — About
# ─────────────────────────────────────────────────────────────────────────────
with tab_about:
    st.markdown("### About this Project")
    st.markdown(
        """
        <div class="card">
        <p>This app converts the Jupyter notebook <code>cifar10_improve_aml13.ipynb</code>
        into a production Streamlit web application.</p>

        <p>The notebook explores three experiments:</p>
        <ol>
        <li><strong>Exp 1</strong> — Basic CNN + normalisation (~65 %)</li>
        <li><strong>Exp 2</strong> — + Data augmentation (~75 %)</li>
        <li><strong>Exp 3</strong> — Deep CNN + BatchNorm + Dropout + Callbacks (<strong>~86 %</strong>) ✅</li>
        </ol>

        <p>This app implements Experiment 3 architecture.</p>
        </div>

        <div class="card">
        <strong>CIFAR-10 Dataset</strong><br><br>
        60 000 colour images · 32 × 32 px · 10 classes<br>
        50 000 training · 10 000 test<br><br>
        <em>Classes:</em> airplane, automobile, bird, cat, deer,
        dog, frog, horse, ship, truck
        </div>

        <div class="card">
        <strong>How to run locally</strong>
        <pre style="background:#0f1117;border-radius:8px;padding:1rem;font-size:0.8rem;margin-top:8px;">
pip install -r requirements.txt
python train_model.py       # train & save model (~86% acc)
streamlit run app.py        # launch app
        </pre>
        </div>
        """,
        unsafe_allow_html=True,
    )
