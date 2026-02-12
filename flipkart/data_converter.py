import pandas as pd
from langchain_core.documents import Document

class DataConverter:
    def __init__(self , file_path:str):
        self.file_path = file_path

    def convert(self):
        df = pd.read_csv(self.file_path)[["product_id", "product_title", "rating", "summary", "review"]]

        docs = [
            Document(
                id=str(row["product_id"]),          # AstraDB upserts by this ID — safe for incremental loads
                page_content=(
                    f"Product Name: {row['product_title']}\n"
                    f"Rating: {row['rating']}/5\n"
                    f"Summary: {row['summary']}\n"
                    f"Review: {row['review']}"
                ),
                metadata={
                    "product_id": str(row["product_id"]),
                    "product_name": str(row["product_title"]),
                    "rating": str(row["rating"]),
                    "summary": str(row["summary"]),
                }
            )
            for _, row in df.iterrows()
        ]

        return docs

