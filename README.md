# 🤖 AI Data Profiler & Cleaner

An intelligent, local data pipeline built with Python, Dask, and Gradio. This tool automatically ingests tabular data, profiles it for errors, and uses Machine Learning (K-Nearest Neighbors) to intelligently predict and fill missing values. 

## ✨ Features
* **Universal File Support:** Upload `.csv`, `.xlsx`, `.json`, `.parquet`, `.xml`, `.html`, or paste a direct web link.
* **Advanced PDF Parsing:** Automatically extracts structured data tables from `.pdf` files. If no tables are found, it gracefully falls back to extracting unstructured text for line-by-line analysis.
* **Smart Deduplication:** Automatically finds and removes duplicate rows.
* **AI-Powered Imputation:** Uses `sklearn.impute.KNNImputer` to analyze the relationships between your columns and predictively fill missing numeric data. 
* **Data Visualization:** Generates a real-time Plotly graph mapping exactly where your data quality issues are located.
* **Secure & Local:** Everything runs locally on your machine. No data is sent to external cloud APIs for processing.

## 🛠️ Technology Stack
* **UI Framework:** Gradio
* **Data Processing:** Dask & Pandas
* **Machine Learning:** Scikit-Learn
* **PDF Extraction:** pdfplumber
* **Visualization:** Plotly Express

## 🚀 How to Run Locally

**1. Clone the repository**
```bash
git clone [https://github.com/YOUR_USERNAME/ai-data-profiler.git](https://github.com/YOUR_USERNAME/ai-data-profiler.git)
cd ai-data-profiler
