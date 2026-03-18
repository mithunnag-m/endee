import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import PyPDF2


st.set_page_config(page_title="AI RAG Chatbot", layout="centered")

st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
        padding: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)


class VectorDB:
    def __init__(self):
        self.vectors = []
        self.texts = []

    def add(self, vector, text):
        self.vectors.append(vector)
        self.texts.append(text)

    def search(self, query_vector, top_k=3):
        similarities = []
        for i, vec in enumerate(self.vectors):
            sim = np.dot(query_vector, vec) / (
                np.linalg.norm(query_vector) * np.linalg.norm(vec)
            )
            similarities.append((sim, self.texts[i]))
        similarities.sort(reverse=True)
        return [text for _, text in similarities[:top_k]]


model = SentenceTransformer('all-MiniLM-L6-v2')
db = VectorDB()


default_docs = [
    "A database is a collection of structured data.",
    "SQL is used to manage and query databases.",
    "Machine learning is a subset of artificial intelligence.",
    "Python is a popular programming language.",
    "RAG combines retrieval and generation techniques."
]

for doc in default_docs:
    db.add(model.encode(doc), doc)


st.markdown("<h1 style='text-align:center; color:#4CAF50;'>🤖 AI RAG Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Upload PDF & Ask Smart Questions</p>", unsafe_allow_html=True)

st.info("💡 Tip: Upload a PDF for better answers or ask general questions.")


st.markdown("### 📄 Upload Document")
uploaded_file = st.file_uploader("", type=["pdf"])

if uploaded_file:
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text()

    chunks = text.split("\n")

    for chunk in chunks:
        if chunk.strip():
            db.add(model.encode(chunk), chunk)

    st.success("✅ PDF processed successfully!")


col1, col2 = st.columns([3,1])

with col1:
    query = st.text_input("💬 Ask a question:")

with col2:
    ask_button = st.button("🔍 Ask")


if ask_button:
    if query:

        q_vec = model.encode(query)
        results = db.search(q_vec)


        results = [r for r in results if any(word in r.lower() for word in query.lower().split())]
  
        unique_results = list(dict.fromkeys(results))


        short_results = unique_results[:2]


        st.markdown("### 📄 Retrieved Context")
        for r in short_results:
            st.info(r)

        
        st.markdown("### 🤖 Answer")

        if short_results:
            answer = " ".join(short_results)
            st.success(answer)

            st.markdown("### 📌 Key Points")
            for r in short_results:
                st.write("✔", r)
        else:
            st.warning("No relevant answer found.")


st.markdown("---")
st.markdown("<p style='text-align:center;'>Built using RAG + Endee Concept</p>", unsafe_allow_html=True)