# ==========================
# Import necessary libraries
# ==========================
import streamlit as st  # To build the web app interface
from langchain_groq import ChatGroq  # For using Groq's LLM (like ChatGPT)
from langchain_text_splitters import RecursiveCharacterTextSplitter  # To split large text into smaller parts
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # To create embeddings using Google's AI
from langchain_community.vectorstores import FAISS  # Fast tool to search similar text chunks
import pymupdf  # Used to read PDF files
import pymupdf4llm  # Used to extract clean markdown text from PDF
import os  # To work with environment variables
from dotenv import load_dotenv  # To load secrets from .env file

# ==========================
# Load API Key from .env file
# ==========================
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")  # Load your Google API key

# ==========================
# Set up the Streamlit app
# ==========================
st.set_page_config(page_title="🤖 Student Assistant", page_icon="🏥")
st.title("🤖 Student Assistant 🏥")

# If the key is missing, show error and stop
if not google_api_key:
    st.error("Google API Key not found! Please make sure it's in a .env file.")
    st.stop()

# Keep chat history stored in session
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your personalized assistant. Upload a PDF to get started or ask me a general question. 💊"}
    ]

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =====================================
# Initialize the Models
# =====================================
llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # AI model name
    temperature=0.2  # How creative AI should be (low = more focused)
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",  # Embedding model name
    google_api_key=google_api_key
)

# =====================================
# Sidebar: File uploader for PDF
# =====================================
uploaded_file = st.sidebar.file_uploader(
    "Upload a document (PDF)", type="pdf"  # Upload PDF only
)

# =========================================================================
# Function to process PDF and create vector store (with caching)
# =========================================================================
@st.cache_resource(show_spinner="Processing your document...")
def create_vector_store_from_pdf(file_bytes, _embeddings):
    """
    1. Convert PDF to markdown text
    2. Split text into chunks
    3. Convert chunks to vectors (embeddings)
    4. Store in FAISS for fast search
    """
    doc = pymupdf.open(stream=file_bytes, filetype="pdf")  # Load PDF from memory
    text = pymupdf4llm.to_markdown(doc)  # Convert PDF to clean text

    # Break large text into smaller parts
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.split_text(text)  # Split text

    # Convert each chunk to vector and store in FAISS
    vector_store = FAISS.from_texts(docs, _embeddings)
    return vector_store

# =====================================
# Process uploaded file
# =====================================
vector_store = None  # Default value if no file is uploaded
if uploaded_file:
    file_bytes = uploaded_file.getvalue()  # Read file as bytes
    vector_store = create_vector_store_from_pdf(file_bytes, embeddings)  # Process it
    st.sidebar.success("Document processed! You can now ask questions about it.")

# =====================================
# Main Chat Logic (User asks question)
# =====================================
if prompt := st.chat_input("Ask your question..."):
    # Store user's message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # If PDF is uploaded, use it to find answers
            if vector_store:
                # Find 3 most similar chunks from the PDF
                relevant_docs = vector_store.similarity_search(prompt, k=3)
                context = "\n\n".join([doc.page_content for doc in relevant_docs])

                # Build prompt using context and user question
                full_prompt = (
                    "You are a helpful student assistant. Answer the user's question "
                    "based *only* on the following context provided from their document. "
                    "If the answer is not in the context, clearly state that you cannot find the answer in the document.\n\n"
                    f"Context:\n{context}\n\n"
                    f"Question: {prompt}\nAnswer:"
                )
                response = llm.invoke(full_prompt)  # Ask AI to answer using context
            else:
                # If no PDF, give general answer
                system_prompt = (
                    "You are a helpful student's personalized assistant to help them with their studies. "
                    "Since no document is uploaded, answer the question generally."
                )
                full_prompt = f"{system_prompt}\n\nQuestion: {prompt}"
                response = llm.invoke(full_prompt)

            # Show AI's response
            response_content = response.content
            st.markdown(response_content)
    
    # Save assistant's response to chat
    st.session_state.messages.append(
        {"role": "assistant", "content": response_content}
    )
