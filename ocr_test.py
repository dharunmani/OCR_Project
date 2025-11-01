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

CONF_THRESHOLD = 20.0
MIN_CHAR_LEN = 2

DETECTION_BUFFER_SIZE = 10
MAJORITY_THRESHOLD = 5

SAVE_CAPTURED_IMAGES = True
OUT_EXCEL = "ocr_results.xlsx"
IMAGES_DIR = "captures"
os.makedirs(IMAGES_DIR, exist_ok=True)

print("Loading EasyOCR reader... This may take a moment.")
reader = easyocr.Reader(['en'], gpu=False)
print("EasyOCR reader loaded.")

tess_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-'

def preprocess_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    smoothed = cv2.medianBlur(enhanced_gray, 3)
    
    # --- TUNE THESE PARAMETERS ---
    blockSize = 31
    C = 5
    
    thresh = cv2.adaptiveThreshold(
        smoothed,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize,
        C
    )
    
    kernel = np.ones((2, 2), np.uint8)
    proc = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    proc = cv2.morphologyEx(proc, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return proc

def ocr_tesseract(proc_img):
    try:
        data = pytesseract.image_to_data(proc_img, output_type=pytesseract.Output.DICT, config=tess_config)
        texts, confs = [], []
        
        for i, (t, c) in enumerate(zip(data.get('text', []), data.get('conf', []))):
            cleaned_t = ''.join(filter(lambda x: x.isalnum() or x == '-', t.strip())).upper()
            
            if len(cleaned_t) >= MIN_CHAR_LEN and str(c).strip() != '-1':
                try:
                    conf_val = float(c)
                    texts.append(cleaned_t)
                    confs.append(conf_val)
                except ValueError:
                    continue
                
        text = " ".join(texts).strip()
        mean_conf = np.mean(confs) if confs else -1
        return text, mean_conf
    except Exception as e:
        print(f"Tesseract OCR error: {e}")
        return "", -1

def ocr_easyocr(raw_img):
    try:
        rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        results = reader.readtext(rgb, detail=1, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-')
        texts, confs = [], []
        
        for (bbox, text, conf) in results:
            cleaned_t = text.strip().upper()
            if cleaned_t and len(cleaned_t) >= MIN_CHAR_LEN:
                texts.append(cleaned_t)
                confs.append(conf * 100)
                
        text = " ".join(texts)
        mean_conf = np.mean(confs) if confs else -1
        return text, mean_conf
    except Exception as e:
        print(f"EasyOCR error: {e}")
        return "", -1

def ensemble_ocr(proc_img, raw_img):
    t_text, t_conf = ocr_tesseract(proc_img)
    e_text, e_conf = ocr_easyocr(raw_img)
    
    if t_conf > e_conf:
        return t_text, t_conf
    else:
        return e_text, e_conf

def save_to_excel_row(row):
    df_row = pd.DataFrame([row])
    try:
        if os.path.exists(OUT_EXCEL):
            old = pd.read_excel(OUT_EXCEL)
            df = pd.concat([old, df_row], ignore_index=True)
        else:
            df = df_row
        
        df.to_excel(OUT_EXCEL, index=False)
    
    except PermissionError:
        backup = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        print(f"[Warning] Excel locked! Could not save to {OUT_EXCEL}.")
        print(f"Saving to backup file: {backup}")
        try:
            df_row.to_excel(backup, index=False)
        except Exception as e:
            print(f"[Error] Failed to save backup file: {e}")
            
    except Exception as e:
        print(f"[Error] Could not save to Excel: {e}")

def main():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Camera not found! Please check index (e.g., 0, 1, or 2).")
        return

    print("OCR Scanner Running LIVE... Press 'q' to exit.")
    last_saved_text = ""
    detection_buffer = []
    display_text = ""
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        H, W = frame.shape[:2]
        x1, y1, x2, y2 = int(W*ROI_LEFT), int(H*ROI_TOP), int(W*ROI_RIGHT), int(H*ROI_BOTTOM)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        
        roi = frame[y1:y2, x1:x2]
        
        frame_count += 1
        if frame_count % 2 == 0:
            proc = preprocess_for_ocr(roi)
            cv2.imshow("Preprocessed Image", proc)
            
            text, conf = ensemble_ocr(proc, roi)

            if text and len(text) >= MIN_CHAR_LEN and conf >= CONF_THRESHOLD:
                detection_buffer.append(text)
            else:
                detection_buffer.append(None)
                
            if len(detection_buffer) > DETECTION_BUFFER_SIZE:
                detection_buffer.pop(0)
            
            if len(detection_buffer) >= DETECTION_BUFFER_SIZE:
                vote_count = Counter(detection_buffer)
                
                if vote_count:
                    most_common_text, count = vote_count.most_common(1)[0]
                    
                    if most_common_text is not None and count >= MAJORITY_THRESHOLD:
                        display_text = most_common_text
                        
                        if most_common_text != last_saved_text:
                            last_saved_text = most_common_text
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            print(f"[{timestamp}] CONFIRMED: {most_common_text} (Seen {count}/{DETECTION_BUFFER_SIZE} times)")

                            row = {
                                "Detected Text": most_common_text, 
                                "Confidence": f"{conf:.1f}", 
                                "Timestamp": timestamp,
                                "Vote Count": count
                            }
                            save_to_excel_row(row) 
                            
                            if SAVE_CAPTURED_IMAGES:
                                filename = f"{timestamp.replace(':','-').replace(' ','_')}.png"
                                cv2.imwrite(os.path.join(IMAGES_DIR, filename), roi)
        
        status_text = f"Detected: {display_text}" if display_text else "Scanning..."
        cv2.putText(frame, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        
        buffer_status = f"Buffer: {len([x for x in detection_buffer if x is not None])}/{DETECTION_BUFFER_SIZE}"
        cv2.putText(frame, buffer_status, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1, cv2.LINE_AA)
        
        cv2.imshow("Auto OCR Scanner", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Results saved to {OUT_EXCEL}")

if __name__ == "__main__":
    main()