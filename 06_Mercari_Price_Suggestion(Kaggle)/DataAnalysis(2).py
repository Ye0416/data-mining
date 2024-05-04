import pandas as pd
import plotly.express as px


train_df = pd.read_table("train.tsv")
print(train_df.shape)

fig = px.histogram(train_df, x="price", nbins=100, log_y=True)
fig.update_layout(title="Price Distribution", xaxis_title="Price", yaxis_title="Count (Log Scale)")
fig.show()

fig = px.box(train_df, x="item_condition_id", y="price", log_y=True)
fig.update_layout(title="Price vs. Item Condition", xaxis_title="Item Condition", yaxis_title="Price (Log Scale)")
fig.show()

fig = px.box(train_df, x="shipping", y="price", log_y=True)
fig.update_layout(title="Price vs. Shipping", xaxis_title="Shipping (0: Buyer pays, 1: Seller pays)", yaxis_title="Price (Log Scale)")
fig.show()

top_20_brands = train_df["brand_name"].value_counts().nlargest(20).index
train_df_top_brands = train_df[train_df["brand_name"].isin(top_20_brands)]

fig = px.box(train_df_top_brands, x="brand_name", y="price", log_y=True)
fig.update_layout(title="Price vs. Top 20 Brand Names", xaxis_title="Brand Name", yaxis_title="Price (Log Scale)")
fig.update_xaxes(categoryorder="total descending")
fig.show()
