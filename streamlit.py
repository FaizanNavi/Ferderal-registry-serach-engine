import streamlit as st
import chromadb
from chromadb.utils import embedding_functions

# Initialize ChromaDB client and embedding function
chroma_client = chromadb.PersistentClient(path="my_vectordb")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")

# Create or get the collection
collection = chroma_client.get_or_create_collection(name="federal", embedding_function=sentence_transformer_ef)

def search_query(query_text):
    # Query the collection
    results = collection.query(
        query_texts=[query_text],
        n_results=3,
    )

    if results:
        ids = results.get("ids", [[]])
        documents = results.get("documents", [[]])
        metadatas = results.get("metadatas", [[]])

        output = ""
        for idx in range(len(ids[0])):
            output += f"\nResult {idx + 1}\n"
            output += f"\nDocument ID: \n {ids[0][idx]}\n\n"
            if idx < len(documents[0]):
                output += f"Information: {documents[0][idx]}\n\n"
            if idx < len(metadatas[0]):
                output += "Other info:\n"
                for key, value in metadatas[0][idx].items():
                    output += f"- {key}: {value}\n"
            output += "\n---\n"
        return output
    else:
        return "No search result for this query"

st.title("US Federal Registry")
st.write("Enter a query to search the vector database and retrieve matching results.")

query_text = st.text_input("Query", placeholder="Enter your search term here...")

if st.button("Search"):
    if query_text:
        results = search_query(query_text)
        st.markdown(results, unsafe_allow_html=True)
    else:
        st.error("Please enter a query.")
