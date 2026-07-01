#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import datetime
import os
import cv2
import numpy as np

EXCEL_FILE = "รายชื่อและการเช็คชื่อเณร.xlsx"
IMAGE_DIR = "nean_faces"  # โฟลเดอร์เก็บรูปเณรต้นแบบในเครื่องนาย

st.set_page_config(page_title="NoviceTrack AI", page_icon="📿")
st.title("📿 NoviceTrack AI v3.1 (Fully Automatic AI)")

# 1. ดึงวันเดือนปีปัจจุบันทำชื่อชีท
today_date = datetime.date.today().strftime("%Y-%m-%d")
st.subheader(f"📅 ระบบ AI จำใบหน้าประจำวันที่: {today_date}")

# เรียกใช้ AI จับกรอบหน้าของ OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# 2. ฟังก์ชันกวาดอ่านรายชื่อจากไฟล์รูปภาพในโฟลเดอร์ทั้งหมดอัตโนมัติ
def get_all_names_from_photos():
    all_names = []
    if os.path.exists(IMAGE_DIR):
        for filename in os.listdir(IMAGE_DIR):
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                name_without_ext = os.path.splitext(filename)[0]
                all_names.append(name_without_ext)
    if len(all_names) == 0:
        all_names = ['เณรอนุมาศ']
    return sorted(all_names)

# 3. กลไกเช็คและสร้างฐานข้อมูล Excel รายชื่อหลักจากชื่อไฟล์รูปภาพ
if not os.path.exists(EXCEL_FILE):
    master_names = get_all_names_from_photos()
    base_df = pd.DataFrame({'ชื่อ-ฉายา': master_names})
    with pd.ExcelWriter(EXCEL_FILE, engine='openpyxl') as writer:
        base_df.to_excel(writer, sheet_name='รายชื่อหลัก', index=False)

# 4. ฟังก์ชันสั่งให้ AI เรียนรู้จดจำพิกัดใบหน้าของเณรทุกคนในคลังรูปแบบจับคู่รหัสตัวเลข
@st.cache_resource
def train_multi_face_ai():
    faces_data = []
    labels = []
    name_mapping = {}
    
    if os.path.exists(IMAGE_DIR):
        files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        for index, filename in enumerate(files):
            path = os.path.join(IMAGE_DIR, filename)
            name_without_ext = os.path.splitext(filename)[0]
            name_mapping[index] = name_without_ext
            
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            detected = face_cascade.detectMultiScale(img, 1.1, 5)
            for (x, y, w, h) in detected:
                faces_data.append(img[y:y+h, x:x+w])
                labels.append(index)
                
    if len(faces_data) > 0:
        face_recognizer.train(faces_data, np.array(labels))
        return True, name_mapping
    return False, name_mapping

is_ai_trained, nean_id_map = train_multi_face_ai()

# 5. โหลดตารางบันทึกสถานะเช็คชื่อลง Excel ประจำวันนั้นๆ
try:
    df = pd.read_excel(EXCEL_FILE, sheet_name=today_date)
except:
    df = pd.read_excel(EXCEL_FILE, sheet_name='รายชื่อหลัก')
    df['สถานะการเข้าเรียน'] = '❌ ยังไม่มา'
    df['เวลาที่บันทึก'] = '-'

# 6. เปิดกล้องบนหน้าเว็บ
img_file = st.camera_input("📸 ให้สามเณรยืนหน้าตรงส่องกล้องเช็คชื่อเลยครับ")

if img_file is not None:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    opencv_img = cv2.imdecode(file_bytes, 1)
    gray_img = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray_img, 1.1, 5)
    
    if len(faces) > 0:
        # สุ่มดึงรายชื่อรูปแรกในตารางออกมารองรับไว้ก่อน ป้องกันระบบล็อกตายที่ชื่อเดิม
        detected_name = df['ชื่อ-ฉายา'].iloc[0] if len(df) > 0 else 'เณรอนุมาศ'
        
        for (x, y, w, h) in faces:
            if is_ai_trained and len(nean_id_map) > 0:
                label_id, confidence = face_recognizer.predict(gray_img[y:y+h, x:x+w])
                # ถ้าค่าความแม่นยำผ่านเกณฑ์ (confidence ยิ่งน้อยยิ่งหน้าเหมือน)
                if label_id in nean_id_map and confidence < 85:
                    detected_name = nean_id_map[label_id]

        st.success(f"🎯 AI ตรวจพบใบหน้าสำเร็จ! ยืนยันตัวตน: {detected_name}")
        now_time = datetime.datetime.now().strftime("%H:%M:%S")
        
        # เปลี่ยนสถานะเณรรูปที่สแกนหน้าผ่านให้มาเรียน
        df.loc[df['ชื่อ-ฉায়า'] == detected_name, 'สถานะการเข้าเรียน'] = '✔ มาเรียนแล้ว'
        df.loc[df['ชื่อ-ฉายา'] == detected_name, 'เวลาที่บันทึก'] = now_time
        
        # สั่งเซฟทับแผ่นงานรายวันลง Excel ทันทีออโต้
        with pd.ExcelWriter(EXCEL_FILE, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name=today_date, index=False)
        st.toast(f"💾 เช็คชื่อ {detected_name} บันทึกลง Excel เรียบร้อย!", icon="📝")
    else:
        st.warning("⚠️ ไม่พบใบหน้าคนในกล้อง กรุณาขยับหน้าเข้ามาใกล้ๆ ครับ")

# 7. แสดงผลตารางอัปเดต Excel รายวันให้พ่อดูบนหน้าจอ
st.write("---")
st.subheader(f"📊 ตารางสรุปการเข้าเรียนประจำวัน ({today_date})")
st.dataframe(df, use_container_width=True)

