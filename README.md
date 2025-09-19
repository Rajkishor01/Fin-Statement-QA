# 📊 Financial Report Q&A Chatbot

This Streamlit app allows you to upload **financial statements** (PDF/Excel) and interact with them through a **Q&A chatbot** powered by HuggingFace embeddings + a small local language model (SLM).

---

## 🚀 Features
- Upload **PDF, Excel (.xlsx, .xls), or CSV** financial reports.
- Automatic text chunking + embeddings (`all-MiniLM-L6-v2`).
- Local **Flan-T5** model (`google/flan-t5-base`) for Q&A (no API keys required).
- Conversational memory for multi-turn Q&A.
- Clean **Streamlit UI** with chat-like experience.

---

## 🛠️ Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/your-username/finreport-qa.git
cd finreport-qa
```
# Create venv (optional but recommended)
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate  # On Mac/Linux

# Install dependencies
pip install -r requirements.txt

# run the app using
```bash
python -m streamlit run "streamlit_app.py"
```
