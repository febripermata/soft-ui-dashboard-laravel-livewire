import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.impute import KNNImputer
from scipy.stats import zscore
import copy

# Set Seaborn style
sns.set()

# Title for the Streamlit app
st.title('Student Clustering Analysis')

# File uploader
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

if uploaded_file is not None:
    # Get the list of sheets
    excel_file = pd.ExcelFile(uploaded_file)
    sheet_names = excel_file.sheet_names
    
    # Select sheet
    sheet = st.selectbox('Select the sheet', sheet_names)

    # Read the selected sheet
    df = pd.read_excel(uploaded_file, sheet_name=sheet)
    
    # Display the raw dataset
    st.subheader('Raw Dataset')
    st.dataframe(df)

    # Drop unnecessary columns and handle missing values using KNNImputer
    data_clean = df.drop(["No", "Nama"], axis=1)
    imputer = KNNImputer(n_neighbors=5)
    x_imputed = imputer.fit_transform(data_clean)
    data_clean = pd.DataFrame(x_imputed, columns=data_clean.columns)
    
    # Display the cleaned data
    st.subheader('Cleaned Data')
    st.dataframe(data_clean.head())

    # Pairplot of the cleaned data
    st.subheader('Pairplot of Cleaned Data')
    pairplot_fig = sns.pairplot(data_clean)
    st.pyplot(pairplot_fig.fig)

    # Scale the data
    scaled_preprocessing = scale(data_clean)
    scaled = data_clean.apply(zscore)
    
    # Elbow method for finding the optimal number of clusters
    st.subheader('Elbow Method for Optimal Number of Clusters')
    wcss = []
    cluster_range = range(1, 10)
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(scaled)
        wcss.append(kmeans.inertia_)
    
    fig, ax = plt.subplots()
    ax.plot(cluster_range, wcss, marker='x')
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("WCSS")
    ax.set_title("Elbow Method for Optimal Number of Clusters")
    st.pyplot(fig)

    # Clustering with the optimal number of clusters
    optimal_clusters = 4
    kmeans = KMeans(n_clusters=optimal_clusters)
    kmeans.fit(scaled)
    centroids = kmeans.cluster_centers_
    centroid_df = pd.DataFrame(centroids, columns=list(scaled.columns))

    # Display centroids
    st.subheader('Cluster Centroids')
    st.dataframe(centroid_df)

    # Assign clusters
    clusters = scaled.copy()
    clusters['cluster_pred'] = kmeans.fit_predict(scaled)
    scaled["labels"] = clusters['cluster_pred']
    
    # Pairplot with clusters
    st.subheader('Pairplot with Cluster Labels')
    pairplot_with_labels_fig = sns.pairplot(scaled, hue='labels')
    st.pyplot(pairplot_with_labels_fig.fig)

    # Make a deep copy of the scaled data to avoid modification
    data_copy = copy.deepcopy(scaled)
    data_copy['Cluster'] = kmeans.labels_

    # Replace cluster indices with sorted centroid indices
    sorted_indices = centroid_df.index.values
    data_copy['Cluster'] = data_copy['Cluster'].replace(dict(zip(range(optimal_clusters), sorted_indices)))

    # Replace cluster labels with 'LOW', 'MEDIUM', and 'HIGH'
    label_mapping = {0: 'LOW', 2: 'MEDIUM', 1: 'HIGH'}  # Adjusted to the sorted centroid indices
    data_copy['Cluster'] = data_copy['Cluster'].map(label_mapping)

    # Join student names from the original dataframe based on index
    data_copy_with_names = data_copy.join(df[['Nama']], how='inner')

    # Move the 'Nama' column to the first position
    columns_reordered = ['Nama'] + [col for col in data_copy_with_names.columns if col != 'Nama']
    data_copy_with_names = data_copy_with_names[columns_reordered]

    # Separate students into different DataFrames based on the cluster
    low_students = data_copy_with_names[data_copy_with_names['Cluster'] == 'LOW']
    medium_students = data_copy_with_names[data_copy_with_names['Cluster'] == 'MEDIUM']
    high_students = data_copy_with_names[data_copy_with_names['Cluster'] == 'HIGH']

    # Concatenate students from all categories
    all_students = pd.concat([low_students, medium_students, high_students], axis=0)

    # Display the list of all students
    st.subheader('All Students with Cluster Labels')
    st.dataframe(all_students)

    # Scatter plot for each cluster
    st.subheader('Distribution of Students Across Clusters')
    cluster_colors = {'LOW': 'blue', 'MEDIUM': 'green', 'HIGH': 'red'}
    fig, ax = plt.subplots(figsize=(12, 6))
    for cluster, color in cluster_colors.items():
        students_in_cluster = data_copy_with_names[data_copy_with_names['Cluster'] == cluster]
        ax.scatter(students_in_cluster.index, students_in_cluster['Cluster'], c=color, label=cluster)
    
    ax.set_title('Distribution of Students Across Clusters')
    ax.set_xlabel('Student Index')
    ax.set_ylabel('Cluster')
    ax.legend()
    st.pyplot(fig)

    # Optional: Calculate and display total variance within clusters
    # Calculate the squared distances of each sample to its nearest centroid
    squared_distances = ((scaled.values - kmeans.cluster_centers_[kmeans.labels_]) ** 2).sum(axis=1)
    
    # Calculate cluster variances by summing the squared distances within each cluster
    cluster_variances = pd.DataFrame(squared_distances, columns=['squared_distances']).groupby(kmeans.labels_).sum()
    
    # Calculate the total variance within all clusters
    total_variance = cluster_variances.sum().sum()
    
    # Display the total variance
    st.subheader('Total Variance within All Clusters')
    st.write("Total variance within all clusters:", total_variance)
