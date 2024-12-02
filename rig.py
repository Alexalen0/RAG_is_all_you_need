import streamlit as st
import PyPDF2
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

class PDFRAGSystem:
    def __init__(self):
        # Initialize embedding model
        self.embed_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        
        # Initialize Hugging Face Pipeline for LLM (updated with Llama model)
        hf_pipeline = pipeline("text-generation", model="meta-llama/Llama-3.2-1B-Instruct",max_length=512)
        self.llm = HuggingFacePipeline(pipeline=hf_pipeline)
        
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from uploaded PDF"""
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    
    def chunk_text(self, text, chunk_size=200, overlap=50):
        """Split text into manageable chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=overlap
        )
        chunks = text_splitter.split_text(text)
        return chunks
    
    def generate_embeddings(self, chunks):
        """Generate embeddings for text chunks"""
        embeddings = []
        with torch.no_grad():
            for chunk in chunks:
                inputs = self.tokenizer(chunk, return_tensors='pt', truncation=True, max_length=512)
                embedding = self.embed_model(**inputs).last_hidden_state.mean(dim=1)
                embeddings.append(embedding.numpy().flatten())
        
        return np.array(embeddings)
    
    def create_faiss_index(self, embeddings):
        """Create FAISS index for efficient similarity search"""
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index
    
    def retrieve_relevant_context(self, query, index, chunks, top_k=3):
        """Retrieve most relevant text chunks"""
        # Generate query embedding
        with torch.no_grad():
            query_inputs = self.tokenizer(query, return_tensors='pt', truncation=True, max_length=512)
            query_embedding = self.embed_model(**query_inputs).last_hidden_state.mean(dim=1).numpy()
        
        # Search similar chunks
        D, I = index.search(query_embedding, top_k)
        relevant_chunks = [chunks[i] for i in I[0]]
        return relevant_chunks
    
    def generate_answer(self, query, context):
        """Generate answer using retrieved context"""
        prompt = f"Context: {' '.join(context)}\n\nQuestion: {query}\n\nAnswer:"
        response = self.llm(prompt)
        return response
    
    def run_rag_pipeline(self, pdf_file, query):
        """Main RAG pipeline"""
        # Extract text from PDF
        pdf_text = self.extract_text_from_pdf(pdf_file)
        
        # Chunk text
        chunks = self.chunk_text(pdf_text)
        
        # Generate embeddings
        embeddings = self.generate_embeddings(chunks)
        
        # Create FAISS index
        index = self.create_faiss_index(embeddings)
        
        # Retrieve context
        context = self.retrieve_relevant_context(query, index, chunks)
        
        # Generate answer
        answer = self.generate_answer(query, context)
        
        return answer

def main():
    st.title("PDF RAG Question Answering System")
    
    # PDF Upload
    uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])
    
    # Query Input
    query = st.text_input("Enter your question")
    
    # Initialize RAG System
    rag_system = PDFRAGSystem()
    
    if uploaded_file and query:
        try:
            answer = rag_system.run_rag_pipeline(uploaded_file, query)
            st.write("Answer:", answer)
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
