import chromadb
from chromadb.utils import embedding_functions
import csv


# Load sample data

with open('combined_data.csv', encoding='utf-8') as file:
    lines = csv.DictReader(file)  # Use DictReader to work with column names

    # Initialize arrays
    documents = []
    metadatas = []
    ids = []

    # Loop through each line and populate the arrays
    for row in lines:
        # Add the "abstract" column to the documents list
        documents.append(row["abstract"])

        # Use "document_number" as the ID
        ids.append(row["document_number"])

        # Use all other columns as metadata (excluding "abstract" and "document_number")
        metadata = {key: value for key, value in row.items() if key not in ["abstract", "document_number"]}
        metadatas.append(metadata)


chroma_client = chromadb.PersistentClient(path="my_vectordb")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")

collection = chroma_client.get_or_create_collection(name = "federal", embedding_function=sentence_transformer_ef)

results = collection.query(
    query_texts=  ["tax"],
    n_results=3,
)
# Organize and print results completely
if results:
    ids = results.get("ids", [[]])
    documents = results.get("documents", [[]])
    metadatas = results.get("metadatas", [[]])

    # Ensure all IDs, documents, and metadata are grouped and displayed properly
    for idx in range(len(ids[0])):  # Loop through the results based on IDs
        print(f"Result {idx + 1}:")
        
        # Print ID
        print(f"ID:{ids[0][idx]}")

        # Print Document
        if idx < len(documents[0]):
            print(f"Document:{documents[0][idx]}")

        # Print Metadata
        if idx < len(metadatas[0]):
            print("  Metadata:")
            for key, value in metadatas[0][idx].items():
                print(f"{key}: {value}")

        print("-" * 80)  # Separator for readability
else:
    print("No results found.")