1️⃣ README.md
# 🔍 Real-Time OCR Detection using OpenCV, Tesseract, and EasyOCR

This project performs **real-time Optical Character Recognition (OCR)** using a live camera feed.  
It is designed for **industrial environments** to automatically detect and record product serial numbers, labels, or printed text.

---

## 🚀 Features
- 📷 Real-Time Camera OCR using OpenCV  
- 🧠 Hybrid Detection with Tesseract + EasyOCR  
- 📈 Confidence Filter (≥ 80%)  
- 🗂️ Automatic Excel Logging (text + timestamp + confidence)  
- 💾 Optional image saving of detected regions  
- 🔄 Designed for 24×7 continuous operation  
- ⚙️ Compatible with industrial cameras  

---

## 🧰 Tech Stack
| Component | Purpose |
|------------|----------|
| **OpenCV** | Camera access, ROI extraction, preprocessing |
| **Tesseract OCR** | Text recognition |
| **EasyOCR** | Deep learning OCR |
| **NumPy** | Image array processing |
| **Pandas** | Excel file handling |
| **Python** | Core language |

---

## 🧩 Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/OCR_Project.git
cd OCR_Project

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Install Tesseract OCR
Windows:

Download from Tesseract OCR (UB Mannheim)

and update the path in code if needed:

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

Linux:
sudo apt install tesseract-ocr

▶️ Run the Project
python camera_test.py


Camera window opens automatically.

Detected text with ≥ 80 % confidence appears in terminal.

Press q to exit.

📁 Output Files
File	Description
ocr_results.xlsx	Text + confidence + timestamp
captures/	Saved images (optional)
ocr_results_backup_YYYYMMDD_HHMMSS.xlsx	Backup if Excel is locked
🧠 How It Works

Capture video frames with OpenCV

Preprocess: grayscale → CLAHE → threshold → sharpen

Run both Tesseract and EasyOCR

Compare results → keep higher confidence

Save result if confidence ≥ 80 %

🏭 Industrial Deployment

For factory / production line setup:

Use industrial-grade camera (fixed mount)

Maintain constant lighting

Run continuously via loop:

:loop
python camera_test.py
timeout /t 5
goto loop


Rotate Excel logs daily/weekly

Optionally upload to cloud or DB

🛠️ requirements.txt
opencv-python
pytesseract
easyocr
pandas
numpy

📜 License

MIT License

👨‍💻 Author

Dharun.M
B.Tech – Artificial Intelligence and Machine Learning
Panimalar Engineering College


---

### ⚙️ **2️⃣ `requirements.txt`**
```txt
opencv-python
pytesseract
easyocr
pandas
numpy

🙈 3️⃣ .gitignore
# Python cache
__pycache__/
*.pyc

# Virtual environments
.venv/
env/
venv/

# Excel logs and data
ocr_results.xlsx
ocr_results_backup_*.xlsx

# Captured images
captures/

# IDE files
.vscode/
.idea/
.DS_Store