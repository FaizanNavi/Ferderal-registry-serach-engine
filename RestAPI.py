import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB client with my_vectordb
chroma_client = chromadb.PersistentClient(path="my_vectordb")
collection = chroma_client.get_or_create_collection(
    name="abstract_embeddings",
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
)

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def search_query(query_text):
    """
    Query the vector database for similar entries to the given text.
    """
    # Encode the query into embeddings
    query_embedding = model.encode(query_text).tolist()

    # Query the database
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    if results:
        ids = results.get("ids", [[]])
        documents = results.get("documents", [[]])
        metadatas = results.get("metadatas", [[]])

        output = ""
        for idx in range(len(ids[0])):
            output += f"\n### Result {idx + 1}\n"
            output += f"**Document ID:** {ids[0][idx]}\n\n"
            if idx < len(documents[0]):
                output += f"**Information:** {documents[0][idx]}\n\n"
            if idx < len(metadatas[0]):
                output += "**Other Info:**\n"
                for key, value in metadatas[0][idx].items():
                    output += f"- **{key}:** {value}\n"
            output += "\n---\n"
        return output
    else:
        return "No results found for this query."

# Streamlit UI
st.title("Vector Database Query Interface")
st.write("Search for information in the vector database using a text query.")

# Query input
query_text = st.text_input("Enter your query:", placeholder="Type something to search...")

# Search button
if st.button("Search"):
    if query_text:
        st.write("Processing your query...")
        results = search_query(query_text)
        st.markdown(results, unsafe_allow_html=True)
    else:
        st.error("Please enter a query.")
