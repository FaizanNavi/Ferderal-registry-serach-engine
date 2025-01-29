import chromadb
from chromadb.config import Settings
import csv

# Initialize ChromaDB client with the new configuration
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",  # Use DuckDB with Parquet
    persist_directory="./chroma_db"  # Directory for persistent storage
))

# Create or retrieve a collection
collection_name = "my_collection"
collection = client.get_or_create_collection(name=collection_name)  # Updated to match the new API

# Load data from CSV
csv_file_path = "combined_data.csv"
with open(csv_file_path, encoding="utf-8") as file:
    lines = csv.DictReader(file)  # Read CSV with headers
    
    # Initialize lists for documents, metadata, and IDs
    documents = []
    metadatas = []
    ids = []
    
    # Populate lists
    for row in lines:
        documents.append(row["abstract"])  # Extract abstract column
        ids.append(row["document_number"])  # Extract document_number column
        # Extract metadata (exclude abstract and document_number)
        metadata = {key: value for key, value in row.items() if key not in ["abstract", "document_number"]}
        metadatas.append(metadata)

# Add data to the ChromaDB collection
collection.upsert( 
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

print(f"Data successfully added to collection '{collection_name}'.")

query_results = collection.query(
    query_texts=["example query"], 
    n_results=3  
)

print("Query Results:", query_results)
