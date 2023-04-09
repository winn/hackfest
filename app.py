import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def kmeans_clustering(url, k):
    # read the CSV file from the URL into a Pandas DataFrame
    df = pd.read_csv(url)

    # select only the columns that have a numeric data type
    numeric_cols = df.select_dtypes(include=["int", "float"]).columns.tolist()


    # create a new DataFrame with only the selected columns
    df = df[numeric_cols]
    df = df.dropna()

    # perform K-Means clustering on the data with K equals to k
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df)

    # get the cluster labels for each data point
    labels = kmeans.labels_

    # add the cluster labels as a new column to the DataFrame
    df['cluster'] = labels

    # perform PCA on the selected features to reduce the dimensions to two
    pca = PCA(n_components=2)
    pca.fit(df[numeric_cols])
    features_2d = pca.transform(df[numeric_cols])

    # add the two PCA dimensions to the DataFrame
    df['feature_1'] = features_2d[:, 0]
    df['feature_2'] = features_2d[:, 1]

    # return the DataFrame with the cluster labels and PCA dimensions
    return df

# create the Streamlit app
def app():
    st.title("K-Means Clustering App")

    # get the URL of the CSV file from the user
    url = st.text_input("Enter the URL of the CSV file")

    # get the value of K from the user
    k = st.slider("Select the value of K", min_value=2, max_value=10)

    # run the K-Means clustering function on the data
    if st.button("Run Clustering"):
        result_df = kmeans_clustering(url, k)

        # display the result as a table in Streamlit
        st.write(result_df)

        # visualize the clusters in a 2D scatter plot
        fig, ax = plt.subplots()
        for label in result_df['cluster'].unique():
            cluster_df = result_df[result_df['cluster'] == label]
            ax.scatter(cluster_df['feature_1'], cluster_df['feature_2'], label=f'Cluster {label}')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title(f'K-Means Clustering (K={k})')
        ax.legend()
        st.pyplot(fig)

# call the Streamlit app
if __name__ == "__main__":
    app()
