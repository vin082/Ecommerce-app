from flipkart.data_ingestion import DataIngestor

def reingest():
    print("Starting re-ingestion process...")
    try:
        ingestor = DataIngestor()
        # triggering ingestion with load_existing=False to force adding new docs
        ingestor.ingest(load_existing=False)
        print("Data successfully re-ingested with Product Names included!")
    except Exception as e:
        print(f"Error during ingestion: {e}")

if __name__ == "__main__":
    reingest()
