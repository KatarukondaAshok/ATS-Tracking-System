from dotenv import load_dotenv
import os
import re
from collections import Counter

import fitz  # PyMuPDF
import httpx
import streamlit as st
from openai import OpenAI

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()

raw_api_key = os.getenv("XAI_API_KEY", "")
API_KEY = raw_api_key.strip().strip('"').strip("'")

# ---------------------------
# Streamlit page config
# ---------------------------
st.set_page_config(page_title="ATS Resume Expert", layout="wide")
st.title("ATS Tracking System")

# ---------------------------
# xAI client setup
# ---------------------------
client = None
if API_KEY and not API_KEY.startswith("=") and API_KEY.startswith("gsk"):
    try:
        client = OpenAI(
            api_key=API_KEY,
            base_url="https://api.x.ai/v1",
            timeout=httpx.Timeout(120.0)
        )
    except Exception:
        client = None

# ---------------------------
# Utility functions
# ---------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#./\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_pdf_text(uploaded_file):
    if uploaded_file is None:
        raise FileNotFoundError("No file uploaded")

    pdf_bytes = uploaded_file.read()
    if not pdf_bytes:
        raise ValueError("Uploaded PDF is empty")

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()

    text = text.strip()
    if not text:
        raise ValueError("Could not extract text from the PDF")
    return text


def tokenize_keywords(text: str):
    text = clean_text(text)

    patterns = [
        r"\bpython\b", r"\bsql\b", r"\br\b", r"\btensorflow\b", r"\bpytorch\b",
        r"\bscikit learn\b", r"\bscikit-learn\b", r"\bkeras\b", r"\bpandas\b",
        r"\bnumpy\b", r"\bmatplotlib\b", r"\bseaborn\b", r"\bmachine learning\b",
        r"\bdeep learning\b", r"\bartificial intelligence\b", r"\bai\b", r"\bml\b",
        r"\bnlp\b", r"\btransformers?\b", r"\bllm\b", r"\brag\b", r"\bfaiss\b",
        r"\bchromadb\b", r"\bmongodb\b", r"\bmysql\b", r"\bfastapi\b", r"\bdocker\b",
        r"\baws\b", r"\bpower bi\b", r"\beda\b", r"\bfeature engineering\b",
        r"\bclassification\b", r"\bregression\b", r"\bclustering\b",
        r"\bdata preprocessing\b", r"\bdata analysis\b", r"\bdata science\b",
        r"\bdeployment\b", r"\bapi\b", r"\bvector database\b", r"\bann\b",
        r"\brnn\b", r"\bgru\b", r"\blstm\b", r"\bproblem solving\b",
        r"\banalytical skills\b", r"\bcommunication skills\b"
    ]

    found = set()
    for p in patterns:
        matches = re.findall(p, text)
        for m in matches:
            found.add(m.replace("-", " ").strip())

    words = [
        w for w in re.findall(r"\b[a-z][a-z0-9+#./-]{2,}\b", text)
        if w not in {
            "the", "and", "for", "with", "that", "this", "you", "your", "are", "from",
            "have", "has", "will", "into", "using", "used", "use", "our", "their",
            "they", "them", "his", "her", "she", "him", "its", "also", "good", "plus",
            "role", "job", "candidate", "candidates", "engineer", "experience", "skills",
            "required", "preferred", "qualification", "degree", "ability", "work",
            "team", "strong", "knowledge", "understanding", "familiarity", "basic"
        }
    ]

    freq = Counter(words)
    for token, count in freq.items():
        if count >= 2:
            found.add(token)

    return sorted(found)


def local_ats_analysis(job_description: str, resume_text: str):
    resume_clean = clean_text(resume_text)

    jd_keywords = tokenize_keywords(job_description)
    resume_keywords = set(tokenize_keywords(resume_text))

    matched = [kw for kw in jd_keywords if kw in resume_keywords or kw in resume_clean]
    missing = [kw for kw in jd_keywords if kw not in matched]

    score = round((len(matched) / len(jd_keywords)) * 100) if jd_keywords else 0

    strengths = []
    for kw in [
        "python", "sql", "tensorflow", "keras", "machine learning",
        "deep learning", "nlp", "rag", "transformers", "faiss",
        "mongodb", "mysql", "ann", "rnn", "gru", "lstm", "eda",
        "feature engineering", "data analysis"
    ]:
        if kw in resume_clean:
            strengths.append(kw)

    strengths = list(dict.fromkeys(strengths))[:10]

    if score >= 80:
        final_thoughts = "Excellent alignment. The resume matches most of the core technical expectations in the job description."
    elif score >= 65:
        final_thoughts = "Good alignment. The resume matches many required skills, but a few important keywords and role-specific terms are missing."
    elif score >= 50:
        final_thoughts = "Moderate alignment. The resume shows relevant background, but it should be tailored more strongly to the job description."
    else:
        final_thoughts = "Low alignment. The resume needs significant tailoring to match the role requirements more directly."

    return {
        "score": score,
        "matched": matched[:20],
        "missing": missing[:20],
        "strengths": strengths,
        "weaknesses": missing[:10],
        "final_thoughts": final_thoughts
    }


def local_resume_review(job_description: str, resume_text: str):
    analysis = local_ats_analysis(job_description, resume_text)

    summary = []
    summary.append("The resume is relevant for AI/ML roles and shows a good technical foundation in Python, machine learning, deep learning, and project-based learning.")

    if analysis["score"] >= 70:
        summary.append("Overall, the profile aligns well with the submitted job description.")
    else:
        summary.append("Overall, the profile is partially aligned with the submitted job description and would benefit from stronger tailoring.")

    if analysis["strengths"]:
        summary.append("Key strengths include: " + ", ".join(analysis["strengths"][:8]) + ".")

    if analysis["missing"]:
        summary.append("Important missing or underemphasized keywords include: " + ", ".join(analysis["missing"][:8]) + ".")

    summary.append("Recommendation: tailor the professional summary, skills, and project descriptions more closely to the job description and mirror the employer’s keywords where they honestly match your experience.")

    return "\n\n".join(summary)


def xai_resume_review(job_description: str, resume_text: str):
    if client is None:
        raise RuntimeError("xAI client unavailable")

    prompt = f"""
You are an experienced HR manager and ATS evaluator.

Review the resume against the job description.

Return your answer under these headings:
1. Overall Fit
2. Strengths
3. Weaknesses
4. Suggestions to Improve

Resume:
{resume_text}

Job Description:
{job_description}
""".strip()

    response = client.responses.create(
        model="grok-4.20-reasoning",
        input=[
            {"role": "system", "content": "You are a professional ATS resume evaluator and HR screening assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.output_text if hasattr(response, "output_text") else str(response)


def xai_ats_score(job_description: str, resume_text: str):
    if client is None:
        raise RuntimeError("xAI client unavailable")

    prompt = f"""
You are a skilled ATS scanner.

Evaluate the resume against the job description and return exactly in this format:

Percentage Match:
Missing Keywords:
Strengths:
Weaknesses:
Final Thoughts:

Resume:
{resume_text}

Job Description:
{job_description}
""".strip()

    response = client.responses.create(
        model="grok-4.20-reasoning",
        input=[
            {"role": "system", "content": "You are a professional ATS resume evaluator and HR screening assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.output_text if hasattr(response, "output_text") else str(response)


# ---------------------------
# UI
# ---------------------------
input_text = st.text_area("Job Description:", height=250, key="input")
uploaded_file = st.file_uploader("Upload your resume (PDF)...", type=["pdf"])

if uploaded_file is not None:
    st.success("PDF Uploaded Successfully")

col1, col2 = st.columns(2)
submit1 = col1.button("Tell Me About the Resume", use_container_width=True)
submit3 = col2.button("Percentage Match", use_container_width=True)

# ---------------------------
# Main actions
# ---------------------------
if submit1 or submit3:
    if uploaded_file is None:
        st.warning("Please upload the resume.")
        st.stop()

    if not input_text.strip():
        st.warning("Please enter the job description.")
        st.stop()

    try:
        pdf_content = extract_pdf_text(uploaded_file)
    except Exception as e:
        st.error(f"PDF reading error: {e}")
        st.stop()

    if submit1:
        with st.spinner("Analyzing resume..."):
            try:
                result = xai_resume_review(input_text, pdf_content)
                mode = "xAI"
            except Exception:
                result = local_resume_review(input_text, pdf_content)
                mode = "Local ATS"

        st.subheader("Resume Review")
        st.info(f"Mode: {mode}")
        st.write(result)

    if submit3:
        with st.spinner("Calculating ATS score..."):
            try:
                result = xai_ats_score(input_text, pdf_content)
                mode = "xAI"
            except Exception:
                analysis = local_ats_analysis(input_text, pdf_content)
                result = f"""
Percentage Match:
{analysis['score']}%

Missing Keywords:
{", ".join(analysis['missing']) if analysis['missing'] else "None"}

Strengths:
{", ".join(analysis['strengths']) if analysis['strengths'] else "Relevant technical foundation found in the resume"}

Weaknesses:
{", ".join(analysis['weaknesses']) if analysis['weaknesses'] else "No major weaknesses detected from keyword analysis"}

Final Thoughts:
{analysis['final_thoughts']}
""".strip()
                mode = "Local ATS"

        st.subheader("ATS Result")
        st.info(f"Mode: {mode}")
        st.write(result)
