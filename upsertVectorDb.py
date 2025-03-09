from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import pickle
import os
from dotenv import load_dotenv

# Your API keys (make sure to secure them in production!)
load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
STREAMLIT_INDEX_NAME = os.getenv('STREAMLIT_INDEX_NAME')
KNOWLEDGE_FILE_PATH = os.getenv("KNOWLEDGE_FILE_PATH")

# Initialize Pinecone and SentenceTransformer
pc = Pinecone(api_key=PINECONE_API_KEY)
model = SentenceTransformer('all-mpnet-base-v2')

# Load or generate embeddings

print(f"knowledge file path: {KNOWLEDGE_FILE_PATH}")
def load_embeddings():
    if os.path.exists("embeddings.pkl"):
        with open("embeddings.pkl", "rb") as f:
            pdf_pages, embeddings = pickle.load(f)
        print("Embeddings loaded from file.")
    else:
        pdf_pages = load_pdf(KNOWLEDGE_FILE_PATH)
        print(f"Loaded {len(pdf_pages)} pages from the PDF.")
        embeddings = [generate_embedding(page) for page in pdf_pages]
        with open("embeddings.pkl", "wb") as f:
            pickle.dump((pdf_pages, embeddings), f)
        print("Embeddings generated and saved.")
    return pdf_pages, embeddings

def load_pdf(filepath):
    with open(filepath, 'rb') as file:
        reader = PdfReader(file)
        pages = [page.extract_text() for page in reader.pages]
    return pages

def generate_embedding(text):
    return model.encode(text).tolist()

pdf_pages, embeddings = load_embeddings()

# Prepare Pinecone
if not pc.has_index(STREAMLIT_INDEX_NAME):
    pc.create_index(
        name=STREAMLIT_INDEX_NAME,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

vectors = [
    {
        'id': f'page_{i}',
        'values': embedding,
        'metadata': {'page_number': i, 'text': pdf_pages[i]}
    }
    for i, embedding in enumerate(embeddings)
]

index = pc.Index(STREAMLIT_INDEX_NAME)
index.upsert(vectors)
print("upserted vectors:", vectors)