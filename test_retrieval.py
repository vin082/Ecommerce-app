from flipkart.data_ingestion import DataIngestor
import pandas as pd

def test_retrieval():
    print("Initializing DataIngestor...")
    try:
        data_ingestor = DataIngestor()
        vector_store = data_ingestor.ingest(load_existing=True)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        query = "Review for BoAt Rockerz 235v2"
        print(f"Querying: '{query}'")
        docs = retriever.invoke(query)
        
        print("\n--- Retrieved Documents ---")
        for i, doc in enumerate(docs):
            print(f"\nDocument {i+1}:")
            print(f"Content: {doc.page_content}")
            print(f"Metadata: {doc.metadata}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_retrieval()
