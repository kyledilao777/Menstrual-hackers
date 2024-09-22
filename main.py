import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io  # For in-memory file operations

# Function to preprocess data and train the model
def train_model(data):
    # Split features (X) and target (y)
    X = data.drop(['Sample ID', 'Confidence Level of Viability (%)'], axis=1)  # Features
    y = data['Confidence Level of Viability (%)']  # Target: Confidence Level of Viability

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Evaluate the model on test data
    y_pred = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return rf_model, X_test, y_test, y_pred, mse, r2

# Function to logicle transform biomarkers
def logicle_transform(data):
    data = np.array(data)
    log_part = np.log10(data[data >= 0] + 1)
    linear_part = data[data < 0]
    transformed = np.zeros_like(data)
    transformed[data >= 0] = log_part
    transformed[data < 0] = linear_part
    return transformed

# Load the synthetic stem cell dataset
data = pd.read_csv('./Stem_Cell_Synthetic_Dataset.csv')

# Streamlit UI
st.title("Stem Cell Viability Prediction Portal")

# Allow user input for biomarkers
st.header("Input Biomarker Values")
biomarker_values = {}
for column in data.columns[2:-1]:  # Skip 'Sample ID' and 'Confidence Level'
    biomarker_values[column] = st.number_input(f"{column} Value", min_value=0.0, step=0.1)

# Run the model when the button is clicked
if st.button("Run Model"):
    # Display input values as a DataFrame
    user_input = pd.DataFrame([biomarker_values])
    st.write("### Biomarker Input")
    st.write(user_input)

    # Train the model and get results
    rf_model, X_test, y_test, y_pred, mse, r2 = train_model(data)

    # Predict the viability confidence for the user input
    user_prediction = rf_model.predict(user_input)[0]

    # Display the predicted viability confidence
    st.subheader("Predicted Viability Confidence Level")
    st.write(f"Based on the input biomarkers, the predicted viability confidence level is **{user_prediction:.2f}%**.")

    # Provide a description of the results
    st.write("""
    ### Model Results Explanation
    - **Actual**: The true confidence levels from the test dataset.
    - **Predicted**: The confidence levels predicted by the model.
    - The model uses the input biomarkers to predict the viability confidence level of stem cells.
    """)

    # Display results
    st.subheader("Model Performance on Test Data")
    results_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    st.write(results_df)

    # Display evaluation metrics
    st.write(f"**Mean Squared Error (MSE)**: {mse:.2f}")
    st.write(f"**R-squared (R² Score)**: {r2:.2f}")

    # Add an option to export the results to Excel
    st.subheader("Export Results")
    # Create a button to download the results as an Excel file
    output = io.BytesIO()
    excel_writer = pd.ExcelWriter(output, engine='xlsxwriter')
    results_df.to_excel(excel_writer, index=False, sheet_name='Sheet1')
    excel_writer.save()
    excel_data = output.getvalue()

    st.download_button(
        label="Download Results as Excel",
        data=excel_data,
        file_name='stem_cell_predictions.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

# Additional visualizations (optional)
if st.checkbox("Show Residuals Plot"):
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, color='blue')
    plt.title('Residuals Distribution (Actual - Predicted)')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    st.pyplot(plt)

if st.checkbox("Show Feature Importance Plot"):
    feature_importances = pd.Series(rf_model.feature_importances_, index=X_test.columns)
    plt.figure(figsize=(10, 6))
    feature_importances.nlargest(10).plot(kind='barh', color='skyblue')
    plt.title('Top 10 Important Biomarkers')
    plt.xlabel('Importance Score')
    plt.ylabel('Biomarker')
    st.pyplot(plt)
