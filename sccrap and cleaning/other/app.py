import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd
import streamlit as st

st.title("Fedral Registry")

data_file = "combined_data.csv" 
st.write("Loading data...")
df = pd.read_csv(data_file)
st.write("Data loaded successfully.")

st.write("Loading pre-trained model and generating embeddings...")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['abstract'].tolist(), convert_to_tensor=True)
st.write("Embeddings generated.")

st.write("Initializing ChromaDB...")
chroma_client = chromadb.PersistentClient(path="local_chromadb_storage")
collection_name = "abstract_embeddings"
try:
    collection = chroma_client.create_collection(name=collection_name)
except chromadb.errors.UniqueConstraintError:
    st.write(f"Collection '{collection_name}' already exists. Using the existing collection.")
    collection = chroma_client.get_collection(name=collection_name)

# Step 4: Add data to the collection
st.write("Adding data to ChromaDB...")
metadata = df.drop(columns=['abstract']).to_dict(orient='records')
collection.add(
    embeddings=embeddings.tolist(),
    documents=df['abstract'].tolist(),
    metadatas=metadata,
    ids=[str(i) for i in range(len(df))]
)
st.write("Data added to collection.")

# Step 5: Query the collection
query = st.text_input("Enter a search query:")
if query:
    st.write("Searching collection...")
    query_embedding = model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=5)

    # Display results
    st.write("Search Results:")
    for i, doc in enumerate(results['documents'][0]):
        st.write(f"Result {i+1}: {doc}")
        st.write("Metadata:")
        for key, value in results['metadatas'][0][i].items():
            st.write(f"  {key}: {value}")



# Simplified Process Explanation:
# 1. Load data from a CSV file containing abstracts and metadata.
# 2. Use a pre-trained transformer model to generate embeddings for abstracts.
# 3. Initialize or retrieve a ChromaDB collection.
# 4. Add embeddings, abstracts, and metadata to the ChromaDB collection.
# 5. Accept a query from the user and search the collection using generated embeddings.
# 6. Display the search results and associated metadata interactively in Streamlit.
