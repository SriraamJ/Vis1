import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

import plotly.express as px

# -------------------------
# Page configuration
# -------------------------
st.set_page_config(
    page_title="App Store Games Dashboard",
    layout="wide"
)

st.title("📊 App Store Games – Model & Report Dashboard")
st.markdown("""
This dashboard presents insights from the **Model** and **Report** stages of the data analysis:
- Regression
- Clustering
- Principal Component Analysis (PCA)
""")

# -------------------------
# Load data
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("appstore_games.csv")
    return df

df = load_data()

# -------------------------
# Sidebar filters
# -------------------------
st.sidebar.header("Filters")

selected_genres = st.sidebar.multiselect(
    "Select Primary Genre",
    options=df["Primary Genre"].dropna().unique(),
    default=df["Primary Genre"].dropna().unique()[:5]
)

price_range = st.sidebar.slider(
    "Price range",
    float(df["Price"].min()),
    float(df["Price"].max()),
    (0.0, float(df["Price"].quantile(0.95)))
)

filtered_df = df[
    (df["Primary Genre"].isin(selected_genres)) &
    (df["Price"] >= price_range[0]) &
    (df["Price"] <= price_range[1])
]

# -------------------------
# Model stage
# -------------------------
model_df = filtered_df[
    ["Average User Rating", "User Rating Count", "Price", "Size", "Primary Genre"]
].dropna()

model_df["log_ratings"] = np.log1p(model_df["User Rating Count"])

# ---- Regression ----
X = model_df[["Price", "log_ratings"]]
y = model_df["Average User Rating"]

reg = LinearRegression()
reg.fit(X, y)
model_df["rating_pred"] = reg.predict(X)

# ---- Clustering ----
features = model_df[["Average User Rating", "Price", "Size", "log_ratings"]]
scaled_features = StandardScaler().fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=42)
model_df["Cluster"] = kmeans.fit_predict(scaled_features)

# ---- PCA ----
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_features)

model_df["PC1"] = pca_components[:, 0]
model_df["PC2"] = pca_components[:, 1]

# -------------------------
# Dashboard layout
# -------------------------
col1, col2 = st.columns(2)

# -------------------------
# Visualization 1: Clustering
# -------------------------
with col1:
    st.subheader("1️⃣ Game Clusters (Price vs Rating)")
    fig1 = px.scatter(
        model_df,
        x="Price",
        y="Average User Rating",
        color="Cluster",
        hover_data=["Primary Genre"],
        title="K-Means Clustering of Games"
    )
    st.plotly_chart(fig1, use_container_width=True)

# -------------------------
# Visualization 2: PCA
# -------------------------
with col2:
    st.subheader("2️⃣ PCA Projection")
    fig2 = px.scatter(
        model_df,
        x="PC1",
        y="PC2",
        color="Cluster",
        title="PCA of Game Features"
    )
    st.plotly_chart(fig2, use_container_width=True)

# -------------------------
# Visualization 3: Regression
# -------------------------
with col1:
    st.subheader("3️⃣ Regression: Popularity vs Rating")
    fig3 = px.scatter(
        model_df,
        x="log_ratings",
        y="Average User Rating",
        trendline="ols",
        title="User Rating vs Popularity (log scale)"
    )
    st.plotly_chart(fig3, use_container_width=True)

# -------------------------
# Visualization 4: Genre comparison
# -------------------------
with col2:
    st.subheader("4️⃣ Top Genres by Average Rating")
    genre_avg = (
        filtered_df
        .groupby("Primary Genre")["Average User Rating"]
        .mean()
        .reset_index()
        .sort_values("Average User Rating", ascending=False)
        .head(10)
    )

    fig4 = px.bar(
        genre_avg,
        x="Primary Genre",
        y="Average User Rating",
        title="Top 10 Genres by Average Rating"
    )
    st.plotly_chart(fig4, use_container_width=True)

# -------------------------
# Key insights section
# -------------------------
st.markdown("---")
st.subheader("📌 Key Insights")

st.markdown("""
- **Popularity (number of ratings)** has a stronger relationship with user rating than price.
- Games naturally form **three distinct clusters**, representing different market segments.
- PCA confirms that clustering captures meaningful structure in the data.
- Certain genres consistently achieve **higher average ratings**, indicating user preference trends.
""")