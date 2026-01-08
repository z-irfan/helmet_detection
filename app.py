import os
import re
IS_CLOUD = True
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import time

def is_valid_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)



# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Helmet Detection System",
    page_icon="🪖",
    layout="wide"
)

st.markdown("### 📧 Alert Settings")
user_email = st.text_input(
    "Enter your email to receive alerts",
    placeholder="example@gmail.com"
)

st.title("🪖 Helmet Detection for Construction Sites")
st.write("Image, Video  Helmet Detection with Email Alert")

# ---------------- LOAD MODEL ----------------
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)


# ---------------- EMAIL FUNCTION ----------------
def send_email_alert(no_helmet_count, image_path, receiver_email):
    sender_email = st.secrets["EMAIL"]
    app_password = st.secrets["PASSWORD"]

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = "🚨 Helmet Safety Alert"

    body = f"""
ALERT!

{no_helmet_count} worker(s) detected WITHOUT helmet.

Please see the attached image.
"""
    msg.attach(MIMEText(body, "plain"))


    with open(image_path, "rb") as attachment:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())

    encoders.encode_base64(part)
    part.add_header(
        "Content-Disposition",
        f'attachment; filename="{os.path.basename(image_path)}"'
    )
    msg.attach(part)

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(sender_email, app_password)
    server.send_message(msg)
    server.quit()

# ---------------- SIDEBAR ----------------
st.sidebar.header("Settings")

options = ["Image", "Video"]
if not IS_CLOUD:
    options.append("Webcam")

input_type = st.sidebar.selectbox("Choose Input Type", options)

conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    0.1, 1.0, 0.4, 0.05
)

# ---------------- IMAGE MODE ----------------
if input_type == "Image":
    uploaded_file = st.file_uploader(
        "Upload an Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)

        results = model(img_array, conf=conf_threshold)
        annotated = results[0].plot()

        helmet_count = 0
        no_helmet_count = 0

        for box in results[0].boxes:
            cls_name = model.names[int(box.cls[0])].lower()
            if cls_name == "helmet":
                helmet_count += 1
            elif cls_name in ["head", "no_helmet"]:
                no_helmet_count += 1

        temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        cv2.imwrite(temp_img.name, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

        col1, col2 = st.columns(2)
        col1.image(image, caption="Original Image", use_container_width=True)
        col2.image(annotated, caption="Detection Result", use_container_width=True)

        st.success(f"🪖 Helmet: {helmet_count} | ❌ No Helmet: {no_helmet_count}")

        if user_email and is_valid_email(user_email):
            st.error("⚠️ Safety Alert: Worker without helmet is detected!")
            
            if user_email:
                send_email_alert(no_helmet_count, temp_img.name, user_email)
                st.success(f"📧 Alert sent to {user_email}")
            
            else:
                st.warning("⚠️ Enter email to receive alert")


# ---------------- VIDEO MODE ----------------
elif input_type == "Video":
    video_file = st.file_uploader(
        "Upload a Video",
        type=["mp4", "avi", "mov"]
    )

    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        email_sent = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=conf_threshold)
            annotated = results[0].plot()

            helmet_count = 0
            no_helmet_count = 0

            for box in results[0].boxes:
                cls_name = model.names[int(box.cls[0])].lower()
                if cls_name == "helmet":
                    helmet_count += 1
                elif cls_name in ["head", "no_helmet"]:
                    no_helmet_count += 1

            stframe.image(annotated, channels="BGR", use_container_width=True)

            if no_helmet_count > 0 and not email_sent:
                temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                cv2.imwrite(temp_img.name, annotated)
                
                if user_email and is_valid_email(user_email):
                    send_email_alert(no_helmet_count, temp_img.name, user_email)
                    st.success(f"📧 Alert sent to {user_email}")
                else:
                    st.warning("⚠️ Enter email to receive alert")
                
                email_sent = True


            time.sleep(0.03)

        cap.release()

# ---------------- WEBCAM MODE ----------------
elif input_type == "Webcam":
    st.warning("⚠️ Webcam works only when running locally")

    start = st.checkbox("Start Webcam")
    stop = st.checkbox("Stop Webcam")

    if not IS_CLOUD:
        cap = cv2.VideoCapture(0)

    stframe = st.empty()
    email_sent = False

    while start and not stop:
        ret, frame = cap.read()
        if not ret:
            st.error("Cannot access webcam")
            break

        results = model(frame, conf=conf_threshold)
        annotated = results[0].plot()

        helmet_count = 0
        no_helmet_count = 0

        for box in results[0].boxes:
            cls_name = model.names[int(box.cls[0])].lower()
            if cls_name == "helmet":
                helmet_count += 1
            elif cls_name in ["head", "no_helmet"]:
                no_helmet_count += 1

        stframe.image(annotated, channels="BGR", use_container_width=True)

        if no_helmet_count > 0 and not email_sent:
            temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            cv2.imwrite(temp_img.name, annotated)
            if user_email and is_valid_email(user_email):
                send_email_alert(no_helmet_count, temp_img.name, user_email)
                st.success(f"📧 Alert sent to {user_email}")
            elif no_helmet_count > 0:
                st.warning("⚠️ Enter email to receive alert")

            email_sent = True
            st.warning("📧 Email sent for webcam violation")

        time.sleep(0.03)

    cap.release()









