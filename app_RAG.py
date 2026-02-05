import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
# Kör app_RAG.py i terminalen 

# 1. Konfigurera sidan
st.set_page_config(page_title="Gemini PDF-Chatt")
st.title("Chatta med PDF (via Google Gemini)")

# Hämta API-nyckel från sidofältet för säkerhet
api_key = st.sidebar.text_input("AIzaSyBOEsM0nE3iWo72EAz5b5QwO2UQqfsF9d0", type="password")

# 2. Filuppladdning
uploaded_file = st.file_uploader("Ladda upp en PDF", type="pdf")

if uploaded_file is not None and api_key:
    # Google kräver att denna miljövariabel sätts
    os.environ["GOOGLE_API_KEY"] = api_key

    # 3. Spara filen temporärt
    temp_file = "./temp.pdf"
    with open(temp_file, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.write("Bearbetar filen med Gemini...")

    # 4. Ladda och dela upp texten (Chunking i 500 och overlap till nästa 200)
    loader = PyPDFLoader(temp_file)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)

    # 5. Skapa Embeddings (Google-versionen)
    # Använder modellen "models/embedding-001" som är gratis/billig
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Skapa vektordatabasen
    vector_store = FAISS.from_documents(chunks, embeddings)

    st.success("Klar! Fråga på.")

    # 6. Skapa kedjan (Google-versionen)
    # Använder "gemini-1.5-flash" som är snabb
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )

    # 7. Chat-interface
    query = st.text_input("Din fråga:")

    if query:
        with st.spinner("Gemini tänker..."):
            response = qa_chain.invoke(query)
            st.write("Svar:")
            st.write(response["result"])

elif not api_key:
    st.warning("Du måste ange en Google API-nyckel i menyn till vänster för att starta.")