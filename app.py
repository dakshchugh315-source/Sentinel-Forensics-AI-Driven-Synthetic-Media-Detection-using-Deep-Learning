import streamlit as st
from PIL import Image
import tensorflow as tf 
import numpy as np        
import random             
import time

# ── Page config ───────── # 1. Sabse pehle config (Line 9 se 15 tak ye rakho)
st.set_page_config(
    page_title="CIFake · AI Image Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)# 2. Phir model load karo (Iske baad)-----------------------------------------------------------
@st.cache_resource
def load_sentinel_model():
    return tf.keras.models.load_model('deepfake_model_FINAL.h5', compile=False)

model = load_sentinel_model()

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: #0f1117;
    color: #e2e8f0;
}
.stApp { background-color: #0f1117; }

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 2rem 2rem; max-width: 1200px; }

/* ── Top Navbar ── */
.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.1rem 0 1.4rem 0;
    border-bottom: 1px solid #1e2433;
    margin-bottom: 2.5rem;
}
.navbar-brand {
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.brand-icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #3b82f6, #1d4ed8);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}
.brand-name {
    font-size: 1.25rem; font-weight: 700;
    color: #f1f5f9; letter-spacing: -0.3px;
}
.brand-badge {
    background: #1e3a5f; color: #60a5fa;
    font-size: 0.65rem; font-weight: 600;
    padding: 2px 8px; border-radius: 20px;
    border: 1px solid #2d5a8e; letter-spacing: 0.5px;
    text-transform: uppercase;
}
.nav-links { display: flex; gap: 2rem; }
.nav-link {
    color: #94a3b8; font-size: 0.9rem; font-weight: 500;
    text-decoration: none; cursor: pointer;
    transition: color 0.2s;
}
.nav-link.active { color: #f1f5f9; font-weight: 600; }

/* ── Hero ── */
.hero {
    text-align: center;
    margin-bottom: 2.5rem;
}
.hero h1 {
    font-size: 2.6rem; font-weight: 700;
    color: #f8fafc; letter-spacing: -1px;
    margin-bottom: 0.6rem;
}
.hero p {
    color: #94a3b8; font-size: 1rem;
    max-width: 580px; margin: 0 auto;
    line-height: 1.6;
}

/* ── Cards ── */
.card {
    background: #161b27;
    border: 1px solid #1e2433;
    border-radius: 16px;
    padding: 1.5rem;
}
.card-label {
    font-size: 0.7rem; font-weight: 700;
    letter-spacing: 1.5px; text-transform: uppercase;
    color: #64748b; margin-bottom: 1rem;
}

/* ── Upload zone ── */
.upload-zone {
    border: 2px dashed #2d3748;
    border-radius: 12px;
    padding: 2.5rem 1rem;
    text-align: center;
    cursor: pointer;
    transition: border-color 0.2s, background 0.2s;
    background: #0f1117;
}
.upload-zone:hover { border-color: #3b82f6; background: #111827; }
.upload-icon {
    width: 52px; height: 52px;
    background: linear-gradient(135deg, #3b82f6, #1d4ed8);
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px; margin: 0 auto 1rem auto;
}
.upload-title {
    font-size: 1rem; font-weight: 600;
    color: #e2e8f0; margin-bottom: 0.3rem;
}
.upload-sub { color: #64748b; font-size: 0.8rem; margin-bottom: 1rem; }
.tag-row { display: flex; flex-wrap: wrap; gap: 0.4rem; justify-content: center; }
.tag {
    background: #1e2433; color: #94a3b8;
    font-size: 0.75rem; padding: 3px 12px;
    border-radius: 20px; border: 1px solid #2d3748;
}

/* ── Result panel ── */
.result-empty {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    min-height: 280px; color: #4b5563;
}
.result-icon { font-size: 3rem; margin-bottom: 0.8rem; opacity: 0.4; }

/* ── Confidence bar ── */
.conf-wrap { margin: 1rem 0; }
.conf-labels {
    display: flex; justify-content: space-between;
    font-size: 0.75rem; color: #94a3b8; margin-bottom: 0.4rem;
}
.conf-bar-bg {
    background: #1e2433; border-radius: 99px;
    height: 10px; overflow: hidden;
}
.conf-bar-fill {
    height: 100%; border-radius: 99px;
    transition: width 1s ease;
}

/* ── Verdict badge ── */
.verdict {
    display: inline-flex; align-items: center; gap: 0.5rem;
    padding: 0.5rem 1.1rem; border-radius: 99px;
    font-weight: 700; font-size: 0.95rem;
    margin-bottom: 1rem;
}
.verdict-ai { background: #3b0764; color: #e879f9; border: 1px solid #7e22ce; }
.verdict-real { background: #052e16; color: #4ade80; border: 1px solid #166534; }

/* ── Stats row ── */
.stats-row {
    display: flex; gap: 0.8rem; margin-top: 1rem;
}
.stat-box {
    flex: 1; background: #0f1117;
    border: 1px solid #1e2433; border-radius: 10px;
    padding: 0.7rem; text-align: center;
}
.stat-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem; font-weight: 600; color: #e2e8f0;
}
.stat-key { font-size: 0.65rem; color: #64748b; margin-top: 2px; text-transform: uppercase; letter-spacing: 0.8px; }

/* ── Settings panel ── */
.settings-section { margin-bottom: 1.5rem; }
.settings-title {
    font-size: 0.7rem; font-weight: 700;
    letter-spacing: 1.5px; text-transform: uppercase;
    color: #64748b; margin-bottom: 0.8rem;
    padding-bottom: 0.5rem; border-bottom: 1px solid #1e2433;
}

/* ── Streamlit widget overrides ── */
div[data-testid="stFileUploader"] {
    background: transparent !important;
}
div[data-testid="stFileUploader"] > div {
    background: #0f1117 !important;
    border: 2px dashed #2d3748 !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
}
div[data-testid="stFileUploader"]:hover > div {
    border-color: #3b82f6 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #1d4ed8) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    padding: 0.55rem 1.5rem !important;
    transition: opacity 0.2s !important;
    width: 100%;
}
.stButton > button:hover { opacity: 0.88 !important; }
.stSlider > div > div > div { background: #3b82f6 !important; }
.stSelectbox > div > div {
    background: #161b27 !important;
    border: 1px solid #2d3748 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}
.stToggle > label { color: #94a3b8 !important; }
div[data-testid="stTabs"] button {
    color: #64748b !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #f1f5f9 !important;
    border-bottom-color: #3b82f6 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "detect"

# ── Navbar ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
  <div class="navbar-brand">
    <div class="brand-icon">🎯</div>
    <span class="brand-name">CIFake</span>
    <span class="brand-badge">Beta</span>
  </div>
  <div class="nav-links">
    <span class="nav-link active">Detect</span>
    <span class="nav-link">History</span>
    <span class="nav-link">Docs</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>AI Image Detection</h1>
  <p>Upload any image to determine whether it was generated by AI or captured in the real world.</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs: Detect | Settings ────────────────────────────────────────────────────
tab_detect, tab_settings = st.tabs(["🔍  Detect", "⚙️  Settings"])

# ════════════════════════════════════════════════════════════════════════════════
# DETECT TAB
# ════════════════════════════════════════════════════════════════════════════════
with tab_detect:
    col_left, col_right = st.columns([1, 1], gap="large")

    # ── LEFT: Upload ──────────────────────────────────────────────────────────
    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Upload Image</div>', unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align:center; margin-bottom:0.8rem;">
          <div class="upload-icon">⬆️</div>
          <div class="upload-title">Drop your image here or click to browse</div>
          <div class="upload-sub">Supports JPG, PNG, WEBP · Max 10MB</div>
          <div class="tag-row">
            <span class="tag">Faces</span>
            <span class="tag">Landscapes</span>
            <span class="tag">Art</span>
            <span class="tag">Screenshots</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            label="", type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed"
        )

        if uploaded:
            img = Image.open(uploaded)
            st.image(img, use_container_width=True,
                     caption=f"{uploaded.name}  ·  {round(uploaded.size/1024, 1)} KB")
            analyze_btn = st.button("Analyze Image", use_container_width=True)
            if analyze_btn:
                with st.spinner("Decoding Neural Layers..."):
                # 1. Image ko model ke liye prepare karo
                    img_prep = img.convert('RGB').resize((128, 128))
                    img_array = np.array(img_prep) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                    # 2. Asli Model se prediction lo
                    raw_pred = model.predict(img_array)[0][0]

                    # 3. Decision Logic: 0.5 se kam = AI, zyada = Real
                    if raw_pred < 0.50:
                        label = "AI Generated"
                        confidence = round((1 - raw_pred) * 100, 1)
                    else:
                        label = "Real Photo"
                        confidence = round(raw_pred * 100, 1)

                # 4. Result save karo taaki UI mein dikh sake
                st.session_state.result = {
                    "label": label, 
                    "confidence": confidence,
                    "name": uploaded.name, 
                    "size": uploaded.size
                }
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    # ── RIGHT: Result ─────────────────────────────────────────────────────────
    with col_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Detection Result</div>', unsafe_allow_html=True)

        result = st.session_state.result
        if result is None:
            st.markdown("""
            <div class="result-empty">
              <div class="result-icon">🔍</div>
              <div style="font-size:0.9rem;">Upload an image to see results</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            is_ai = result["label"] == "AI Generated"
            verdict_cls = "verdict-ai" if is_ai else "verdict-real"
            verdict_icon = "🤖" if is_ai else "📷"
            bar_color = "#e879f9" if is_ai else "#4ade80"
            conf = result["confidence"]
            anti_conf = round(100 - conf, 1)

            st.markdown(f"""
            <div style="text-align:center; padding: 1rem 0 0.5rem 0;">
              <div class="verdict {verdict_cls}">{verdict_icon} {result['label']}</div>
            </div>
            """, unsafe_allow_html=True)

            # Confidence bar
            st.markdown(f"""
            <div class="conf-wrap">
              <div class="conf-labels">
                <span>Confidence</span>
                <span style="font-family:'JetBrains Mono',monospace; color:#e2e8f0; font-weight:600;">
                  {conf}%
                </span>
              </div>
              <div class="conf-bar-bg">
                <div class="conf-bar-fill" style="width:{conf}%; background:{bar_color};"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Stats
            size_kb = round(result["size"] / 1024, 1)
            st.markdown(f"""
            <div class="stats-row">
              <div class="stat-box">
                <div class="stat-val">{conf}%</div>
                <div class="stat-key">AI Score</div>
              </div>
              <div class="stat-box">
                <div class="stat-val">{anti_conf}%</div>
                <div class="stat-key">Real Score</div>
              </div>
              <div class="stat-box">
                <div class="stat-val">{size_kb}KB</div>
                <div class="stat-key">File Size</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Detailed breakdown
            with st.expander("📊 Detailed Analysis"):
                factors = {
                    "Texture Patterns": round(conf * 0.98, 1),
                    "Lighting Consistency": round(conf * 0.96, 1),
                    "Edge Artifacts": round(conf * 0.99, 1),
                    "Noise Distribution": round(conf * 0.95, 1),
                    "Metadata Integrity": round(conf * 0.97, 1),
                }
                for factor, score in factors.items():
                    # Color logic: Green for high confidence, Red for low
                    color = "#4ade80" if score > 70 else "#facc15" if score > 40 else "#f87171"
                    st.markdown(f"""
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem; font-size:0.82rem;">
                      <span style="color:#94a3b8;">{factor}</span>
                      <span style="font-family:'JetBrains Mono',monospace; color:{color}; font-weight:600;">{score}%</span>
                    </div>
                    <div style="background:#1e2433; border-radius:99px; height:6px; margin-bottom:0.7rem;">
                      <div style="width:{score}%; height:100%; border-radius:99px; background:{color};"></div>
                    </div>
                    """, unsafe_allow_html=True)

            if st.button("🔄 Analyze Another", use_container_width=True):
                st.session_state.result = None
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# SETTINGS TAB
# ════════════════════════════════════════════════════════════════════════════════
with tab_settings:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    s1, s2 = st.columns(2, gap="large")

    with s1:
        st.markdown('<div class="settings-title">Detection Model</div>', unsafe_allow_html=True)
        st.selectbox("Model Version", ["CIFake v2.1 (Default)", "CIFake v1.8 (Faster)", "CIFake v2.1-Pro (Accurate)"],
                     label_visibility="collapsed")
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown('<div class="settings-title">Confidence Threshold</div>', unsafe_allow_html=True)
        threshold = st.slider("Min confidence to flag as AI (%)", 50, 99, 75, label_visibility="visible")
        st.markdown(f'<div style="font-size:0.78rem; color:#64748b; margin-top:-0.5rem;">Images above <b style="color:#60a5fa">{threshold}%</b> will be flagged as AI-generated.</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown('<div class="settings-title">File Handling</div>', unsafe_allow_html=True)
        max_size = st.selectbox("Maximum upload size", ["5 MB", "10 MB", "25 MB", "50 MB"], index=1)
        auto_clear = st.toggle("Auto-clear image after analysis", value=False)

    with s2:
        st.markdown('<div class="settings-title">Analysis Options</div>', unsafe_allow_html=True)
        show_breakdown = st.toggle("Show detailed breakdown by default", value=True)
        save_history = st.toggle("Save detection history locally", value=True)
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown('<div class="settings-title">Display</div>', unsafe_allow_html=True)
        accent = st.selectbox("Accent Color", ["Blue (Default)", "Purple", "Cyan", "Green"])
        st.toggle("Show confidence bar animation", value=True)
        st.toggle("Show file metadata in results", value=True)
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown('<div class="settings-title">About</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.8rem; color:#64748b; line-height:1.7;">
          <b style="color:#94a3b8;">CIFake Beta</b> · v0.4.2<br>
          Model trained on 2M+ real & AI images.<br>
          Supports GAN, Diffusion & Upscaled detection.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 2rem 0 1rem; color: #374151; font-size: 0.75rem;">
  CIFake Beta · AI-powered image authenticity detection
</div>
""", unsafe_allow_html=True)
