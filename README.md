# 🚀 ATS Tracking System

AI-powered **ATS Resume Analyzer** built using Streamlit and deployed on Hugging Face.

🔗 **Live App:**  
👉 https://huggingface.co/spaces/katarukondaashok143/ATS_Tracking_System

---

## 📌 Features

- 📄 Resume Review (AI-based analysis)
- 📊 ATS Percentage Match Score
- 🔍 Keyword Matching & Missing Skills Detection
- 🧠 Smart insights for improvement
- ⚡ Works even if API fails (Local ATS fallback)

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit  
- **Backend:** Python  
- **Libraries:** PyMuPDF, OpenAI SDK, Regex, NLP  
- **Deployment:** Hugging Face Spaces  

---

## 🧠 How It Works

1. Upload your resume (PDF)
2. Paste a job description
3. Click:
   - "Tell Me About the Resume"
   - "Percentage Match"
4. Get:
   - ATS Score
   - Strengths & Weaknesses
   - Missing Keywords

---

## ⚙️ Installation

```bash
git clone https://github.com/KatarukondaAshok/ATS-Tracking-System.git
cd ATS-Tracking-System
pip install -r requirements.txt
streamlit run app.py
