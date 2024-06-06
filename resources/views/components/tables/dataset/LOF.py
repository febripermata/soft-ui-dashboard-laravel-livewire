import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor, LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

# Title for the Streamlit app
st.title('Student Clustering and Anomaly Detection')

# File uploader
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

if uploaded_file is not None:
    # Reading the dataset
    row_data = pd.read_excel(uploaded_file, sheet_name='Sheet1')
    
    # Display the raw dataset
    st.subheader('Raw Dataset')
    st.dataframe(row_data)

    # Create a binary target column "Prediksi" based on the absence score
    threshold_absen = row_data['absen'].mean()
    row_data['Prediksi'] = (row_data['absen'] > threshold_absen).astype(int)
    st.write('Threshold for absence score:', threshold_absen)
    st.dataframe(row_data.head())

    # Drop unnecessary columns and handle missing values using KNNImputer
    data_clean = row_data.drop(["No", "Nama"], axis=1)
    imputer = KNNImputer(n_neighbors=5)
    x_imputed = imputer.fit_transform(data_clean)
    data_clean = pd.DataFrame(x_imputed, columns=data_clean.columns)
    st.subheader('Cleaned Data')
    st.dataframe(data_clean.head())

    X = data_clean.drop(columns=['Prediksi']).values
    y = data_clean['Prediksi'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Initializing LOF model
    n_neighbors_lof = 20
    contamination_lof = 0.1
    lof = LocalOutlierFactor(n_neighbors=n_neighbors_lof, contamination=contamination_lof)

    # Calculating anomaly scores using LOF
    anomaly_score_lof = lof.fit_predict(X)

    # Adding LOF anomaly scores to the dataframe
    row_data['Anomaly_Score_LOF'] = anomaly_score_lof

    # Identifying outliers based on LOF
    outliers_lof = row_data[row_data['Anomaly_Score_LOF'] == -1]

    st.subheader('Details of Outliers Detected by LOF')
    st.dataframe(outliers_lof[['Nama', 'absen', 'Anomaly_Score_LOF']])

    # Visualizing LOF anomaly scores with outlier markers
    st.subheader('Anomaly Detection using LOF')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(range(len(anomaly_score_lof)), anomaly_score_lof, c=anomaly_score_lof, cmap='coolwarm', label='Anomaly Score LOF')
    ax.scatter(outliers_lof.index, outliers_lof['Anomaly_Score_LOF'], c='red', marker='x', label='Outliers LOF')
    ax.set_title('Anomaly Detection using LOF')
    ax.set_xlabel('Data Point')
    ax.set_ylabel('Anomaly Score LOF')
    plt.colorbar(ax.collections[0], label='Anomaly Score LOF')
    ax.legend()
    st.pyplot(fig)

    # Initialize labels to 0 for normal instances
    row_data['Anomaly_Label'] = 0

    # Adjust the anomaly score threshold
    anomaly_score_threshold = 12

    # Create a binary column indicating anomalies based on the adjusted threshold
    row_data['Anomaly_Label'] = (row_data['Anomaly_Score_LOF'] == -1)

    # Set a threshold for well grades
    well_grades_threshold = 12  # Adjust this threshold based on your criteria

    # Define a function to classify students based on specific criteria
    def classify_students(row):
        if row['Anomaly_Label'] == True:
            return 'Not Well-Graded'
        else:
            return 'Well-Graded'

    # Apply the classification function to each row and create a new 'Class' column
    row_data['Class'] = row_data.apply(classify_students, axis=1)

    # Identify outliers among 'Well-Graded' students
    well_graded_outliers = row_data[(row_data['Class'] == 'Well-Graded') & row_data['Anomaly_Label']]
    st.subheader('Details of Well-Graded Outliers')
    st.dataframe(well_graded_outliers[['Nama', 'absen', 'Anomaly_Score_LOF', 'Class', 'Anomaly_Label']].head())

    # Identify outliers among 'Not Well-Graded' students
    not_well_graded_outliers = row_data[(row_data['Class'] == 'Not Well-Graded') & row_data['Anomaly_Label']]
    st.subheader('Details of Not Well-Graded Outliers')
    st.dataframe(not_well_graded_outliers[['Nama', 'absen', 'Anomaly_Score_LOF', 'Class', 'Anomaly_Label']].head())
