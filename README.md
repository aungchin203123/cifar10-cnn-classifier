# CIFAR-10 CNN Streamlit App

A full-featured Streamlit web interface for your CIFAR-10 TensorFlow/Keras classifier.

---

## 📁 Project Structure

```
cifar10_streamlit_app/
├── app.py                   ← Main Streamlit application
├── cifar10_model_88.keras   ← ⚠️ Copy your trained model here
├── requirements.txt
├── .streamlit/
│   └── config.toml          ← Dark theme config
└── README.md
```

---

## 🚀 Setup in PyCharm

### Step 1 — Copy Your Model
Place your trained model file in the same directory as `app.py`:
```
cifar10_model_88.keras   (or .h5 — update the sidebar path)
```

### Step 2 — Create a Virtual Environment (in PyCharm)
1. Open **File → Settings → Project → Python Interpreter**
2. Click the ⚙️ gear icon → **Add Interpreter → Add Local Interpreter**
3. Choose **Virtualenv** → Create new → OK

### Step 3 — Install Dependencies
Open the **PyCharm Terminal** (bottom bar) and run:
```bash
pip install -r requirements.txt
```

### Step 4 — Run the App
**Option A — PyCharm Terminal:**
```bash
streamlit run app.py
```

**Option B — PyCharm Run Configuration:**
1. Click **Run → Edit Configurations**
2. Click **+** → **Python**
3. Set:
   - **Script path**: Browse to `streamlit` in your venv's `Scripts/` folder  
     e.g. `C:\Users\<you>\venv\Scripts\streamlit.exe`
   - **Parameters**: `run app.py`
   - **Working directory**: folder containing `app.py`
4. Click **OK** then press ▶️

The app opens automatically at `http://localhost:8501`

---

## 🖥️ App Features

| Tab | Description |
|-----|-------------|
| 🔍 **Single Predict** | Upload one image → see prediction + all class probabilities |
| 📦 **Batch Predict** | Upload many images → table + downloadable CSV |
| 🧪 **Explore Dataset** | Sample 50 CIFAR-10 test images + live accuracy |
| 🏗️ **Architecture** | Full model summary + layer table + design rationale |

---

## 🔧 Customisation

- **Different model path**: Change the default in the sidebar text input at runtime, or edit `MODEL_PATH` at the top of `app.py`
- **Different val accuracy badge**: Edit the `88%` string in the metrics row
- **Port**: Change `port` in `.streamlit/config.toml`

---

## 📋 Requirements

- Python 3.9+
- TensorFlow 2.15+
- A trained `.keras` or `.h5` model for CIFAR-10
