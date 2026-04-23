import streamlit as st
import cv2
from ultralytics import YOLO
import pyttsx3
import numpy as np
import threading
import time
from datetime import datetime

# ================= PAGE CONFIG =================
st.set_page_config(page_title="AI Object Detection", layout="wide")

# ================= ICE BLUE UI =================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #EAF6FF, #D6ECFF);
}
h1 {
    color: #0A3D62;
    text-align: center;
    font-weight: 700;
}
.stButton>button {
    background: linear-gradient(90deg, #5DADE2, #3498DB);
    color: white;
    border-radius: 12px;
    font-weight: bold;
    border: none;
    padding: 8px 18px;
}
.custom-box {
    background: rgba(255,255,255,0.6);
    backdrop-filter: blur(10px);
    padding: 18px;
    border-radius: 15px;
    color: #0A3D62;
    font-weight: 600;
    box-shadow: 0 6px 18px rgba(0,0,0,0.1);
}
section[data-testid="stSidebar"] {
    background: #D6ECFF;
}
</style>
""", unsafe_allow_html=True)

st.title("🔍 Object Detection System (YOLO + Voice)")

# ================= MODEL =================
model = YOLO("yolov8n.pt")

# ================= VOICE =================
def speak(text):
    def run():
        engine = pyttsx3.init()   # हर बार नया engine
        engine.setProperty('rate', 150)
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run).start()

# ================= SIDEBAR =================
option = st.sidebar.selectbox("Choose Mode", ["Image", "Webcam"])
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)
voice_enabled = st.sidebar.checkbox("Enable Voice Feedback", True)

if st.sidebar.button("🔄 Refresh"):
    st.session_state.clear()
    st.rerun()

# ================= IMAGE =================
if option == "Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        results = model(img, conf=confidence_threshold)
        output = results[0].plot()

        names = results[0].names
        boxes = results[0].boxes

        detected = []
        table_data = []

        for box in boxes:
            cls_id = int(box.cls[0])
            name = names[cls_id]
            conf = float(box.conf[0])
            detected.append(name)
            table_data.append({"Object": name, "Confidence": f"{conf:.2f}"})

        text = "Detected: " + ", ".join(sorted(set(detected))) if detected else "No object detected"

        col1, col2 = st.columns([2,1])
        with col1:
            st.image(output, channels="BGR")
        with col2:
            st.markdown(f"""
            <div class="custom-box">
            {text} <br><br>
            🔢 Total Objects: {len(set(detected))}
            </div>
            """, unsafe_allow_html=True)
            if table_data:
                st.table(table_data)

        if voice_enabled and detected:
            speak(text)

# ================= WEBCAM =================
elif option == "Webcam":
    if "run" not in st.session_state:
        st.session_state.run = False

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start Camera"):
            st.session_state.run = True
    with col2:
        if st.button("⏹ Stop Camera"):
            st.session_state.run = False

    run = st.session_state.run
    FRAME_WINDOW = st.image([])
    info_box = st.empty()
    log_panel = st.expander("📜 Detection Logs")

    cap = cv2.VideoCapture(0)
    last_objects = set()

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not working")
            break

        results = model(frame, conf=confidence_threshold)
        output = results[0].plot()

        names = results[0].names
        boxes = results[0].boxes
        current_objects = set()
        table_data = []

        for box in boxes:
            cls_id = int(box.cls[0])
            name = names[cls_id]
            conf = float(box.conf[0])
            current_objects.add(name)
            table_data.append({"Object": name, "Confidence": f"{conf:.2f}"})

        if current_objects != last_objects:
            if current_objects:
                text = "Detected: " + ", ".join(sorted(current_objects))
                if voice_enabled:
                    speak(text)
                info_box.markdown(f"""
                <div class="custom-box">
                🟢 Camera Running <br><br>
                {text} <br><br>
                🔢 Total Objects: {len(current_objects)}
                </div>
                """, unsafe_allow_html=True)
                st.table(table_data)
                log_panel.write(f"{datetime.now().strftime('%H:%M:%S')} → {text}")
            else:
                info_box.warning("🔴 No object detected")
            last_objects = current_objects.copy()

        FRAME_WINDOW.image(output, channels="BGR")
        time.sleep(0.03)

    cap.release()
