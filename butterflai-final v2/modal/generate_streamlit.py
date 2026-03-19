"""
butterflai/modal/generate_streamlit.py

Generates a self-contained Streamlit app for any trained ButterflAI model.
Called after training completes. Uploads streamlit_app.py to Drive.
"""

STREAMLIT_TEMPLATE = '''
import streamlit as st
import torch
import timm
import json
import numpy as np
from PIL import Image
import io, os, urllib.request
from torchvision import transforms

# ─── Config ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ButterflAI Demo · {job_id}",
    page_icon="🦋",
    layout="centered",
    initial_sidebar_state="collapsed",
)

CLASSES      = {classes}
MODEL_NAME   = "{model_name}"
IMAGE_SIZE   = {image_size}
JOB_ID       = "{job_id}"
BEST_ACC     = {best_acc:.4f}
DATASET_NAME = "{dataset_name}"

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;1,400&family=DM+Sans:wght@400;500;600&display=swap');
body, .stApp {{ background: #0b0b0e; color: #ede9e2; }}
.stApp {{ font-family: 'DM Sans', sans-serif; }}
h1, h2, h3 {{ font-family: 'Playfair Display', serif !important; }}
.metric-card {{
    background: #131318;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}}
.pred-bar-wrap {{
    background: #1b1b22;
    border-radius: 99px;
    height: 8px;
    overflow: hidden;
    margin: 4px 0;
}}
.pred-bar-fill {{
    height: 100%;
    background: linear-gradient(90deg, #c8a84b, #2dd4bf);
    border-radius: 99px;
    transition: width 0.4s ease;
}}
.result-box {{
    background: linear-gradient(135deg, rgba(200,168,75,0.08), rgba(45,212,191,0.05));
    border: 1px solid rgba(200,168,75,0.25);
    border-radius: 14px;
    padding: 24px;
    text-align: center;
    margin: 20px 0;
}}
</style>
""", unsafe_allow_html=True)

# ─── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = timm.create_model(
        MODEL_NAME.replace("-", "_"),
        pretrained=False,
        num_classes=len(CLASSES)
    )
    # Try loading from local file first, then Drive
    ckpt_path = "best_model.pth"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state, strict=False)
    model.eval()
    return model

# ─── Transform ────────────────────────────────────────────────────────────────
val_tfm = transforms.Compose([
    transforms.Resize(int(IMAGE_SIZE * 1.14)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@torch.no_grad()
def predict(img: Image.Image):
    tensor = val_tfm(img.convert("RGB")).unsqueeze(0)
    model  = load_model()
    logits = model(tensor)[0]
    probs  = torch.softmax(logits, dim=0).numpy()
    top5   = np.argsort(probs)[::-1][:5]
    return [(CLASSES[i], float(probs[i])) for i in top5]

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
  <svg width="32" height="20" viewBox="0 0 40 26" fill="none">
    <path d="M20 13 C14 5 1 1 1 9 C1 17 10 21 20 13Z" fill="#c8a84b" opacity="0.95"/>
    <path d="M20 13 C26 5 39 1 39 9 C39 17 30 21 20 13Z" fill="#2dd4bf" opacity="0.95"/>
    <path d="M20 13 C14 19 2 22 3 17 C4 12 12 11 20 13Z" fill="#c8a84b" opacity="0.6"/>
    <path d="M20 13 C26 19 38 22 37 17 C36 12 28 11 20 13Z" fill="#2dd4bf" opacity="0.6"/>
    <circle cx="20" cy="13" r="2.5" fill="#ede9e2"/>
  </svg>
  <span style="font-family:'Playfair Display',serif;font-weight:900;font-size:22px">
    <span style="color:#c8a84b">Butterfl</span><span style="color:#2dd4bf;font-style:italic">AI</span>
  </span>
</div>
""", unsafe_allow_html=True)

st.markdown(f"### {MODEL_NAME} · {len(CLASSES)}-class classifier")
st.caption(f"Job {JOB_ID} · Trained on **{DATASET_NAME}** · Best val accuracy: **{BEST_ACC*100:.1f}%**")

st.divider()

# ─── Upload ───────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1.2, 1])

with col1:
    uploaded = st.file_uploader(
        "Upload an image to classify",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        label_visibility="visible"
    )

    st.markdown("**Or try a sample URL:**")
    url_input = st.text_input("Image URL", placeholder="https://example.com/image.jpg", label_visibility="collapsed")

    if st.button("🔍 Classify", use_container_width=True, type="primary"):
        st.session_state["run_predict"] = True

with col2:
    st.markdown("**Model Info**")
    st.markdown(f"""
    <div class="metric-card" style="margin-bottom:8px">
      <div style="font-size:10px;color:#5c5855;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px">Architecture</div>
      <div style="font-size:14px;font-weight:600;color:#c8a84b">{MODEL_NAME}</div>
    </div>
    <div class="metric-card" style="margin-bottom:8px">
      <div style="font-size:10px;color:#5c5855;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px">Val Accuracy</div>
      <div style="font-size:18px;font-weight:700;color:#2dd4bf">{BEST_ACC*100:.1f}%</div>
    </div>
    <div class="metric-card">
      <div style="font-size:10px;color:#5c5855;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px">Classes</div>
      <div style="font-size:13px;font-weight:600">{len(CLASSES)}</div>
    </div>
    """, unsafe_allow_html=True)

# ─── Prediction ───────────────────────────────────────────────────────────────
if uploaded or (url_input and st.session_state.get("run_predict")):
    try:
        if uploaded:
            img = Image.open(uploaded)
        else:
            data = urllib.request.urlopen(url_input).read()
            img  = Image.open(io.BytesIO(data))

        st.image(img, caption="Input image", use_column_width=True)

        with st.spinner("Running inference…"):
            results = predict(img)

        top_class, top_prob = results[0]

        st.markdown(f"""
        <div class="result-box">
          <div style="font-size:12px;letter-spacing:2px;text-transform:uppercase;color:#9b9490;margin-bottom:8px">Prediction</div>
          <div style="font-family:'Playfair Display',serif;font-size:28px;font-weight:700;margin-bottom:4px">{top_class}</div>
          <div style="font-size:20px;font-weight:700;color:#2dd4bf">{top_prob*100:.1f}% confidence</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Top predictions:**")
        for cls, prob in results:
            pct = prob * 100
            is_top = cls == top_class
            st.markdown(f"""
            <div style="margin:6px 0">
              <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:3px">
                <span style="color:{'#c8a84b' if is_top else '#ede9e2'};font-weight:{'600' if is_top else '400'}">{cls}</span>
                <span style="color:#9b9490;font-family:'JetBrains Mono',monospace">{pct:.2f}%</span>
              </div>
              <div class="pred-bar-wrap">
                <div class="pred-bar-fill" style="width:{pct}%"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {{e}}")

    st.session_state["run_predict"] = False

# ─── Classes sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🦋 ButterflAI")
    st.caption(f"Job ID: `{JOB_ID}`")
    st.divider()
    st.markdown("**All classes this model knows:**")
    for i, c in enumerate(CLASSES):
        st.markdown(f"`{i:02d}` {c}")
    st.divider()
    st.caption("Built with ButterflAI · butterflai.app")
'''


def generate_streamlit_app(
    job_id: str,
    classes: list,
    model_name: str,
    image_size: int,
    best_acc: float,
    dataset_name: str,
) -> str:
    """Fill template and return the complete streamlit_app.py string."""
    return STREAMLIT_TEMPLATE.format(
        job_id=job_id,
        classes=repr(classes),
        model_name=model_name,
        image_size=image_size,
        best_acc=best_acc,
        dataset_name=dataset_name,
    )


if __name__ == "__main__":
    # Test generation
    code = generate_streamlit_app(
        job_id="JOB_TEST1234",
        classes=["cat", "dog", "bird"],
        model_name="efficientnet_b3",
        image_size=224,
        best_acc=0.9412,
        dataset_name="Animals-10 (Kaggle)",
    )
    with open("/tmp/streamlit_app_test.py", "w") as f:
        f.write(code)
    print("Generated /tmp/streamlit_app_test.py")
    print(f"Length: {len(code)} chars")
