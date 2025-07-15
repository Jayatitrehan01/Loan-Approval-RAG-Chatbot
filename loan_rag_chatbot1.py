# IMPORT LIBRARIES
import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import os

# LOAD DATASET
def load_csv_text(csv_path, limit=100):
    df = pd.read_csv(csv_path).fillna("Unknown")
    text_data = []
    for i, row in df.head(limit).iterrows():
        text = "\n".join([f"{col}: {row[col]}" for col in df.columns])
        text_data.append(text)
    return df, text_data

# VECTOR SEARCH
class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.chunks = []

    def build_index(self, chunks):
        self.chunks = chunks
        embeddings = self.model.encode(chunks)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings).astype('float32'))

    def search(self, query, top_k=3):
        query_embedding = self.model.encode([query])
        D, I = self.index.search(np.array(query_embedding).astype('float32'), top_k)
        return [self.chunks[i] for i in I[0] if i < len(self.chunks)]


# LLM
class RAGBot:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
        self.pipeline = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer)

    def generate(self, context, question):
        prompt = f"Using the following loan application data, answer the question.\n\nData:\n{context}\n\nQuestion: {question}\nAnswer:"
        out = self.pipeline(prompt, max_new_tokens=150)[0]["generated_text"]
        return out.strip()

# STREAMLIT APP
def main():
    st.set_page_config(page_title=" Loan Approval RAG Chatbot ", page_icon="🏦")
    st.title("Loan Approval RAG Chatbot")
  
    try:
        df, text_chunks = load_csv_text(csv_path)
        st.success(f" Dataset loaded with {len(df)} rows and {len(df.columns)} columns.")

        question = st.text_input(" Ask a question about the loan dataset:")
        with st.expander(" Example Questions "):
            st.markdown("""
            - What factors affect loan approval?
            - How does credit history impact loan status?
            - What is the average income of approved applicants?
            - Which property area has the highest approval rate?
            - How many applications were approved vs rejected?
            """)

        if question:
            with st.spinner("  Thinking..."):
                vs = VectorStore()
                vs.build_index(text_chunks)
                top_chunks = vs.search(question)
                context = "\n\n".join(top_chunks)

                bot = RAGBot()
                response = bot.generate(context, question)

                st.success(" Answer:")
                st.write(response)

    except Exception as e:
        st.error(f"Something went wrong: {e}")

# CSV FILE PATH
csv_path = r"C:\Users\jayat\OneDrive\Documents\loan approval pred\Training Dataset.csv"

if __name__ == "__main__":
    main()
