import cv2
import numpy as np
import pytesseract
import easyocr
import pandas as pd
from collections import Counter
from datetime import datetime
import time
import os
import warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*")


ROI_LEFT, ROI_TOP, ROI_RIGHT, ROI_BOTTOM = 0.05, 0.15, 0.95, 0.85
USE_CLAHE, USE_SHARPEN, USE_MORPH = True, True, True
SCAN_INTERVAL = 2.5
frames_for_vote = 3
min_chars_to_accept = 1
CONF_THRESHOLD = 80.0

reader = easyocr.Reader(['en'], gpu=False)
OUT_EXCEL = "ocr_results.xlsx"
SAVE_CAPTURED_IMAGES = True
IMAGES_DIR = "captures"
os.makedirs(IMAGES_DIR, exist_ok=True)

tess_config = r'--oem 3 --psm 6'

def preprocess_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if USE_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)
    proc = th
    if USE_MORPH:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        proc = cv2.morphologyEx(proc, cv2.MORPH_OPEN, kernel, iterations=1)
        proc = cv2.morphologyEx(proc, cv2.MORPH_CLOSE, kernel, iterations=1)
    if USE_SHARPEN:
        gaussian = cv2.GaussianBlur(proc, (0,0), 3)
        sharpened = cv2.addWeighted(proc, 1.5, gaussian, -0.5, 0)
        proc = np.clip(sharpened, 0, 255).astype(np.uint8)
    return proc

def ocr_tesseract(img):
    try:
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=tess_config)
        texts, confs = [], []
        for t, c in zip(data.get('text', []), data.get('conf', [])):
            t = t.strip()
            if t and str(c).strip() != '-1':
                texts.append(t)
                try:
                    confs.append(float(c))
                except:
                    pass
        full = " ".join(texts).strip()
        mean_conf = float(np.mean(confs)) if confs else -1.0
        return full, mean_conf
    except:
        return "", -1.0

def ocr_easyocr(img):
    try:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = reader.readtext(rgb, detail=1, paragraph=False)
        texts, confs = [], []
        for (bbox, text, conf) in results:
            if text.strip():
                texts.append(text.strip())
                confs.append(conf * 100.0)
        full = " ".join(texts).strip()
        mean_conf = float(np.mean(confs)) if confs else -1.0
        return full, mean_conf
    except:
        return "", -1.0

def ensemble_ocr(img):
    t_text, t_conf = ocr_tesseract(img)
    e_text, e_conf = ocr_easyocr(img)
    t_norm, e_norm = t_text.lower().strip(), e_text.lower().strip()
    if t_norm and e_norm and t_norm == e_norm:
        return t_text, max(t_conf, e_conf)
    if t_conf > e_conf:
        return t_text or e_text, t_conf
    else:
        return e_text or t_text, e_conf

def save_to_excel_row(row):
    df_row = pd.DataFrame([row])
    try:
        if os.path.exists(OUT_EXCEL):
            df_old = pd.read_excel(OUT_EXCEL)
            df = pd.concat([df_old, df_row], ignore_index=True)
        else:
            df = df_row
        df.to_excel(OUT_EXCEL, index=False)
    except PermissionError:
        backup = f"ocr_results_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        df_row.to_excel(backup, index=False)
        print(f"[Warning] File locked. Saved backup as {backup}")

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Automatic OCR running... press 'q' to stop.")
    last_scan_time = 0
    last_detected_text = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        H, W = frame.shape[:2]
        x1, y1, x2, y2 = int(W * ROI_LEFT), int(H * ROI_TOP), int(W * ROI_RIGHT), int(H * ROI_BOTTOM)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if time.time() - last_scan_time >= SCAN_INTERVAL:
            roi = frame[y1:y2, x1:x2]
            proc = preprocess_for_ocr(roi)
            text, conf = ensemble_ocr(proc)
            text = text.strip()
            if text and conf >= CONF_THRESHOLD and text != last_detected_text and len(text) >= min_chars_to_accept:
                last_detected_text = text
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] {text} (conf={conf:.1f})")
                row = {
                    "Detected Text": text,
                    "Confidence": round(float(conf), 2),
                    "Timestamp": timestamp
                }
                save_to_excel_row(row)
                if SAVE_CAPTURED_IMAGES:
                    fname = os.path.join(IMAGES_DIR, f"{timestamp.replace(':','-')}.png")
                    cv2.imwrite(fname, roi)
            last_scan_time = time.time()

        cv2.putText(frame, f"Last Detected: {last_detected_text[:50]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        cv2.imshow("Auto OCR (Wide Area)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done. Results saved to:", OUT_EXCEL)

if __name__ == "__main__":
    main()
