import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import pandas as pd
import streamlit as st

# Load the CSV data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Load a pre-trained transformer model
def load_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

# Generate embeddings from abstracts
def generate_embeddings(model, abstracts):
    embeddings = model.encode(abstracts, convert_to_tensor=True)
    return embeddings

# Initialize ChromaDB client and create a collection
def initialize_chromadb():
    chroma_client = chromadb.PersistentClient(path="local_chromadb_storage")
    collection_name = "abstract_embeddings"
    try:
        collection = chroma_client.create_collection(name=collection_name)
    except chromadb.errors.UniqueConstraintError:
        st.write(f"Collection '{collection_name}' already exists. Fetching the existing collection.")
        collection = chroma_client.get_collection(name=collection_name)
    return collection

# Add data to ChromaDB collection
def add_to_collection(collection, abstracts, embeddings, metadata):
    collection.add(
        embeddings=embeddings.tolist(),
        documents=abstracts,
        metadatas=metadata,
        ids=[str(i) for i in range(len(abstracts))]
    )

# Query the collection
def query_collection(collection, model, query, top_n=5):
    query_embedding = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_n
    )
    return results

# Streamlit app
def main():
    st.title("ChromaDB Query Interface")

    data_file = "combined_data.csv"  # Update path if necessary

    st.write("Loading data...")
    df = load_data(data_file)
    st.write("Data loaded successfully.")

    st.write("Loading pre-trained model...")
    model = load_model()
    st.write("Model loaded successfully.")

    abstracts = df['abstract'].tolist()

    st.write("Generating embeddings for abstracts...")
    embeddings = generate_embeddings(model, abstracts)
    st.write("Embeddings generated.")

    st.write("Initializing ChromaDB client...")
    collection = initialize_chromadb()

    metadata = df.drop(columns=['abstract']).to_dict(orient='records')
    st.write("Adding data to ChromaDB collection...")
    add_to_collection(collection, abstracts, embeddings, metadata)
    st.write("Data added to collection.")

    query = st.text_input("Enter your query:")
    if query:
        st.write("Querying the collection...")
        results = query_collection(collection, model, query)
        st.write("Results:")

        for i, doc in enumerate(results['documents'][0]):
            st.write(f"Result {i+1}: {doc}")
            st.write("Metadata:")
            for key, value in results['metadatas'][0][i].items():
                st.write(f"  {key}: {value}")

if __name__ == "__main__":
    main()
