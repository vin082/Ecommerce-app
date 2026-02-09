import pandas as pd
df = pd.read_csv("data/flipkart_product_review.csv")
print(df['product_title'].head().values)
