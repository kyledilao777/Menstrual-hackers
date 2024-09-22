import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io  # For in-memory file operations
from fpdf import FPDF

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
for column in data.columns[1:-1]:  # Skip 'Sample ID' and 'Confidence Level'
    biomarker_values[column] = st.number_input(f"{column} Value", min_value=0.0, step=0.1)

# Run the model when the button is clicked
model_run = st.button("Run Model")

if model_run:
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
    results_df = pd.DataFrame({
        "Sample ID": data['Sample ID'][y_test.index],  # Adding Sample ID
        "Actual": y_test,
        "Predicted": y_pred
    })
    st.write(results_df)

    # Display evaluation metrics
    st.write(f"**Mean Squared Error (MSE)**: {mse:.2f}")
    st.write(f"**R-squared (R² Score)**: {r2:.2f}")

    # Add average predicted viability
    average_viability = results_df['Predicted'].mean()
    st.write(f"**Average Predicted Viability Confidence Level**: {average_viability:.2f}%")

    # Add an option to export the results to Excel
    st.subheader("Export Results")
    # Create a button to download the results as an Excel file
    output = io.BytesIO()
    
    # Use pandas to write to an Excel file with xlsxwriter as the engine
    with pd.ExcelWriter(output, engine='xlsxwriter') as excel_writer:
        results_df.to_excel(excel_writer, index=False, sheet_name='Sheet1')
        excel_writer.close()  
    
    # Ensure the data is available in the buffer
    output.seek(0)
    
    # Streamlit download button for exporting the data
    st.download_button(
        label="Download Results as Excel",
        data=output,
        file_name='stem_cell_predictions.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

    # Visualizations in a 2x2 grid
    st.subheader("Visualizations")

    # Create a 2x2 grid for the visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Step 1: Histogram of a Single Marker (e.g., CD10)
    sns.histplot(data['CD10'], kde=True, color='blue', ax=axes[0, 0])
    axes[0, 0].set_title('Distribution of CD10 Expression')
    axes[0, 0].set_xlabel('CD10 Expression Level')
    axes[0, 0].set_ylabel('Frequency')

    # Step 2: Density Plot (CD44 vs. CD73)
    sns.kdeplot(x=data['CD44'], y=data['CD73'], cmap='Blues', shade=True, ax=axes[0, 1])
    axes[0, 1].set_title('Density Plot of CD44 vs. CD73')
    axes[0, 1].set_xlabel('CD44 Expression Level')
    axes[0, 1].set_ylabel('CD73 Expression Level')

    # Step 3: Residuals Plot
    residuals = y_test - y_pred
    sns.histplot(residuals, kde=True, color='blue', ax=axes[1, 0])
    axes[1, 0].set_title('Residuals Distribution (Actual - Predicted)')
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')

    # Step 4: Feature Importance Plot
    feature_importances = pd.Series(rf_model.feature_importances_, index=X_test.columns)
    feature_importances.nlargest(10).plot(kind='barh', color='skyblue', ax=axes[1, 1])
    axes[1, 1].set_title('Top 10 Important Biomarkers')
    axes[1, 1].set_xlabel('Importance Score')
    axes[1, 1].set_ylabel('Biomarker')

    plt.tight_layout()
    st.pyplot(fig)

    # Re-import necessary libraries for the scatter plot
    cd44 = np.random.normal(loc=50, scale=10, size=1000)
    cd73 = np.random.normal(loc=50, scale=15, size=1000)
    cd44_logicle = logicle_transform(cd44)
    cd73_logicle = logicle_transform(cd73)

    # Logicle Scatter Plot of CD44 vs CD73
    plt.figure(figsize=(8, 6))
    plt.scatter(cd44_logicle, cd73_logicle, alpha=0.5, c=cd44_logicle, cmap='coolwarm')
    plt.title('Logicle Scatter Plot: CD44 vs. CD73 (Color by CD44 Expression)')
    plt.xlabel('CD44 (Logicle Transformed)')
    plt.ylabel('CD73 (Logicle Transformed)')
    plt.colorbar(label='CD44 Expression Level')
    plt.grid(True)
    st.pyplot(plt)

    # Function to save plots as images and export them to PDF
def export_to_pdf(mse, r2, average_viability):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Add title
    pdf.cell(200, 10, txt="Stem Cell Viability Prediction Report", ln=True, align='C')
    
    # Add evaluation metrics
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Mean Squared Error (MSE): {mse:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"R-squared (R² Score): {r2:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Average Predicted Viability Confidence Level: {average_viability:.2f}%", ln=True)

    # Save each plot as an image in memory
    img_buffer = io.BytesIO()

    # Plot 1: Distribution of CD10 Expression
    plt.figure(figsize=(6, 4))
    sns.histplot(data['CD10'], kde=True, color='blue')
    plt.title('Distribution of CD10 Expression')
    plt.xlabel('CD10 Expression Level')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)  # Rewind buffer to beginning
    pdf.image(img_buffer, x=10, y=None, w=100)  # Adjust the x, y, and width as needed
    img_buffer.truncate(0)
    img_buffer.seek(0)  # Clear buffer

    # Plot 2: Density Plot of CD44 vs. CD73
    plt.figure(figsize=(6, 4))
    sns.kdeplot(x=data['CD44'], y=data['CD73'], cmap='Blues', shade=True)
    plt.title('Density Plot of CD44 vs. CD73')
    plt.xlabel('CD44 Expression Level')
    plt.ylabel('CD73 Expression Level')
    plt.tight_layout()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    pdf.image(img_buffer, x=10, y=None, w=100)
    img_buffer.truncate(0)
    img_buffer.seek(0)

    # Plot 3: Residuals Distribution
    residuals = y_test - y_pred
    plt.figure(figsize=(6, 4))
    sns.histplot(residuals, kde=True, color='blue')
    plt.title('Residuals Distribution (Actual - Predicted)')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    pdf.image(img_buffer, x=10, y=None, w=100)
    img_buffer.truncate(0)
    img_buffer.seek(0)

    # Plot 4: Feature Importance Plot
    feature_importances = pd.Series(rf_model.feature_importances_, index=X_test.columns)
    plt.figure(figsize=(6, 4))
    feature_importances.nlargest(10).plot(kind='barh', color='skyblue')
    plt.title('Top 10 Important Biomarkers')
    plt.xlabel('Importance Score')
    plt.ylabel('Biomarker')
    plt.tight_layout()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    pdf.image(img_buffer, x=10, y=None, w=100)
    img_buffer.truncate(0)
    img_buffer.seek(0)

    # Save the PDF to a buffer
    pdf_output = io.BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)

    # Return the PDF for download
    return pdf_output

    # Button to generate PDF report
    if st.button("Export Report as PDF"):
        pdf_output = export_to_pdf(mse, r2, average_viability)
        
        # Download button for the PDF
        st.download_button(
            label="Download PDF Report",
            data=pdf_output,
            file_name="viability_report.pdf",
            mime="application/pdf"
        )

