import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import time
import base64

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Lung Nodule AI Platform",
    page_icon="🧠",
    layout="wide"
)

# =====================================
# CUSTOM MODERN CSS
# =====================================
st.markdown("""
<style>
body { background-color: #0f172a; }
.main { background-color: #0f172a; }
h1 { font-size: 46px !important; text-align:center; }
.subtitle { text-align:center; color:#94a3b8; }
.result-card {
    padding:25px;
    border-radius:20px;
    text-align:center;
    font-size:24px;
    font-weight:bold;
}
.positive { background: linear-gradient(135deg,#ff4b4b,#b91c1c); color:white; }
.negative { background: linear-gradient(135deg,#22c55e,#15803d); color:white; }
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# =====================================
# MODEL (MATCH TRAINING)
# =====================================
class AdvancedCNN(nn.Module):
    def __init__(self):
        super(AdvancedCNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

@st.cache_resource
def load_model():
    model = AdvancedCNN()
    model.load_state_dict(
        torch.load("best_model_advanced.pth", map_location=torch.device("cpu"))
    )
    model.eval()
    return model

model = load_model()

# =====================================
# HEADER
# =====================================
st.markdown("<h1>🧠 Lung Nodule Detection AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Advanced CNN-Based CT Scan Classification</p>", unsafe_allow_html=True)
st.divider()

# =====================================
# UPLOAD
# =====================================
uploaded_file = st.file_uploader("Upload CT Scan Image (.jpg / .png)", type=["jpg", "png"])

if uploaded_file is not None:

    col1, col2 = st.columns(2)

    image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    # ================= PREPROCESS =================
    img = np.array(image)
    img = cv2.resize(img, (64, 64))  # IMPORTANT: same as training
    img = img / 255.0
    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    # ================= PREDICTION =================
    with st.spinner("Analyzing CT scan using Advanced CNN model..."):
        time.sleep(1.5)
        with torch.no_grad():
            output = model(img)
            probs = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)

    prob_values = probs.numpy()[0]
    confidence_value = confidence.item()

    label = "NODULE DETECTED" if predicted.item() == 1 else "NO NODULE DETECTED"

    with col2:

        st.subheader("🔬 AI Analysis Result")

        if predicted.item() == 1:
            st.markdown(
                f'<div class="result-card positive">🚨 {label}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-card negative">✅ {label}</div>',
                unsafe_allow_html=True
            )

        st.write("")

        st.write("### Probability Distribution")
        st.progress(float(prob_values[0]))
        st.write(f"No Nodule: {prob_values[0]:.4f}")

        st.progress(float(prob_values[1]))
        st.write(f"Nodule: {prob_values[1]:.4f}")

        st.write("")
        st.metric("Final Confidence", f"{confidence_value:.4f}")

        st.info("For research and educational purposes only.")

        # ================= DOWNLOAD REPORT =================
        report_text = f"""
        Lung Nodule Detection Report
        ----------------------------------
        Prediction: {label}
        Confidence: {confidence_value:.4f}
        No Nodule Probability: {prob_values[0]:.4f}
        Nodule Probability: {prob_values[1]:.4f}
        """

        b64 = base64.b64encode(report_text.encode()).decode()
        href = f'<a href="data:file/txt;base64,{b64}" download="AI_Report.txt">📄 Download Report</a>'
        st.markdown(href, unsafe_allow_html=True)