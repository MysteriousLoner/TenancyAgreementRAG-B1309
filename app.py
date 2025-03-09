import streamlit as st
from pinecone.grpc import PineconeGRPC as Pinecone
from sentence_transformers import SentenceTransformer
from google import genai
import os
import asyncio
# from dotenv import load_dotenv

# Your API keys (make sure to secure them in production!)
# load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
STREAMLIT_INDEX_NAME = os.getenv('STREAMLIT_INDEX_NAME')

# Initialize Pinecone and SentenceTransformer
pc = Pinecone(api_key=PINECONE_API_KEY)
model = SentenceTransformer('all-mpnet-base-v2')

def generate_embedding(text):
    return model.encode(text).tolist()

index = pc.Index(STREAMLIT_INDEX_NAME)

# Streamlit UI
st.title("Tenancy Agreement Assistant")
st.write("Ask me questions about the tenancy agreement!")

query = st.text_input("Enter your query:")
if query:
    query_embedding = generate_embedding(query)
    result = index.query(vector=query_embedding, top_k=5, include_metadata=True)
    context = "\n".join([match["metadata"]["text"] for match in result['matches']])

    # Generate response
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"Context: {context}\n\nPrompt: {query}"
    )
    st.write(f"**Response:** {response.text}")

async def main():
    # placeholder
    pass

if __name__ == "__main__":
    asyncio.run(main())