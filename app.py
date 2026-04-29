import os
import io
import tempfile
from pathlib import Path

import requests
import pandas as pd
import dask.dataframe as dd
import gradio as gr
import plotly.express as px
import pdfplumber
from sklearn.impute import KNNImputer

class DataPipeline:
    def __init__(self, missing_threshold=0.3):
        self.missing_threshold = missing_threshold

    def ingest(self, source):
        source_str = str(source).strip()
        if source_str.startswith(('http://', 'https://')):
            try:
                response = requests.get(source_str)
                response.raise_for_status()
                try:
                    df = pd.DataFrame(response.json())
                except ValueError:
                    df = pd.read_csv(io.StringIO(response.text))
                return dd.from_pandas(df, npartitions=1)
            except Exception:
                return None

        path = Path(source)
        ext = path.suffix.lower()
        
        try:
            if ext == '.csv':
                return dd.read_csv(path, assume_missing=True)
            elif ext == '.parquet':
                return dd.read_parquet(path)
            elif ext in ['.xls', '.xlsx']:
                return dd.from_pandas(pd.read_excel(path), npartitions=1)
            elif ext == '.json':
                return dd.from_pandas(pd.read_json(path), npartitions=1)
            elif ext == '.xml':
                return dd.from_pandas(pd.read_xml(path), npartitions=1)
            elif ext in ['.html', '.htm']:
                return dd.from_pandas(pd.read_html(path)[0], npartitions=1)
            elif ext == '.pdf':
                all_tables = []
                with pdfplumber.open(path) as pdf:
                    for page in pdf.pages:
                        table = page.extract_table()
                        if table:
                            cleaned_table = [row for row in table if any(cell is not None for cell in row)]
                            all_tables.extend(cleaned_table)
                
                if all_tables:
                    pdf_df = pd.DataFrame(all_tables[1:], columns=all_tables[0])
                    return dd.from_pandas(pdf_df, npartitions=1)
                else:
                    all_text_lines = []
                    with pdfplumber.open(path) as pdf:
                        for page in pdf.pages:
                            text = page.extract_text()
                            if text:
                                lines = [line.strip() for line in text.split('\n') if line.strip()]
                                all_text_lines.extend(lines)
                    
                    if all_text_lines:
                        pdf_df = pd.DataFrame(all_text_lines, columns=["unstructured_text"])
                        return dd.from_pandas(pdf_df, npartitions=1)
                    return None
            return None
        except Exception:
            return None

    def profile(self, ddf):
        if ddf is None:
            return {"error": "No data"}
        
        total_rows = ddf.shape[0].compute()
        missing_counts = ddf.isnull().sum().compute()
        ddf_unique = ddf.drop_duplicates()
        unique_rows = ddf_unique.shape[0].compute()
        
        return {
            "total_rows": int(total_rows),
            "dtypes": ddf.dtypes.astype(str).to_dict(),
            "missing_pct": (missing_counts / total_rows).to_dict(),
            "duplicates": int(total_rows - unique_rows)
        }

    def validate(self, profile_report):
        if "error" in profile_report:
            return False, ["Failed to load data."]
        
        issues = []
        if profile_report.get("duplicates", 0) > 0:
            issues.append(f"Found {profile_report['duplicates']} duplicate rows.")
            
        for col, pct in profile_report["missing_pct"].items():
            if pct > self.missing_threshold:
                issues.append(f"Column '{col}' exceeds missing threshold ({pct*100:.1f}%).")
                
        return len(issues) == 0, issues

    def clean(self, ddf, profile_report):
        action_log = []
        
        if profile_report.get("duplicates", 0) > 0:
            ddf = ddf.drop_duplicates()
            action_log.append(f"Dropped {profile_report['duplicates']} duplicate rows.")
            
        df_clean = ddf.compute()
        knn_model = KNNImputer(n_neighbors=5, weights="distance")
        
        for col, dtype in profile_report.get("dtypes", {}).items():
            if profile_report["missing_pct"].get(col, 0) == 0:
                continue
            
            if 'int' in dtype or 'float' in dtype:
                numeric_cols = df_clean.select_dtypes(include=['number']).columns
                df_clean[numeric_cols] = knn_model.fit_transform(df_clean[numeric_cols])
                action_log.append(f"Imputed missing values in '{col}' using KNN.")
            elif 'object' in dtype or 'string' in dtype:
                df_clean[col] = df_clean[col].fillna("UNKNOWN")
                action_log.append(f"Filled missing text in '{col}' with 'UNKNOWN'.")
                
        if not action_log:
            action_log.append("No cleaning actions were required.")
            
        return dd.from_pandas(df_clean.reset_index(drop=True), npartitions=1), action_log

def process_data(file_input, url_input):
    pipeline = DataPipeline(missing_threshold=0.3)
    source = url_input if url_input.strip() else (file_input.name if file_input else None)
    
    if not source:
        return {"error": "No input"}, "Upload a file or paste a link.", 0, 0, None, "No data provided.", None, None, None

    ddf = pipeline.ingest(source)
    if ddf is None:
        return {"error": "Bad source"}, "Ingestion failed.", 0, 0, None, "Ingestion failed.", None, None, None
        
    cols = list(ddf.columns)
    seen = {}
    new_cols = []
    for c in cols:
        c_str = str(c) if pd.notna(c) and str(c).strip() else "Unnamed"
        if c_str in seen:
            seen[c_str] += 1
            new_cols.append(f"{c_str}_{seen[c_str]}")
        else:
            seen[c_str] = 0
            new_cols.append(c_str)
            
    ddf.columns = new_cols

    report = pipeline.profile(ddf)
    total_rows = report.get("total_rows", 0)
    dupe_count = report.get("duplicates", 0)
    missing_dict = report.get("missing_pct", {})
    total_missing = sum(missing_dict.values())
    
    if total_missing > 0:
        fig = px.bar(
            x=list(missing_dict.keys()),
            y=[v * 100 for v in missing_dict.values()],
            labels={'x': 'Columns', 'y': '% Missing Data'},
            title="Missing Data Percentage by Column",
            color_discrete_sequence=['#ff4b4b']
        )
    else:
        fig = px.bar(
            x=["All Columns"], 
            y=[0],
            title="No Missing Values Detected",
            color_discrete_sequence=['#00cc96']
        )
        
    raw_preview = ddf.head(50).reset_index(drop=True)
    
    def highlight_errors(val):
        return 'background-color: rgba(255, 50, 50, 0.4); font-weight: bold;' if pd.isna(val) else ''
    
    try:
        styled_raw_df = raw_preview.style.map(highlight_errors)
    except AttributeError:
        styled_raw_df = raw_preview.style.applymap(highlight_errors)

    is_critical_pass, issues = pipeline.validate(report)
    
    if total_missing == 0 and dupe_count == 0:
        status_msg = "Dataset is clean. No errors or duplicates found."
        ddf_final = ddf
        audit_text = "No cleaning actions were required."
    else:
        if is_critical_pass:
            status_msg = f"Minor issues found ({dupe_count} duplicates). Auto-cleaning applied."
        else:
            status_msg = f"Issues detected: {'; '.join(issues)}. Auto-cleaning applied."
            
        ddf_final, action_log = pipeline.clean(ddf, report)
        audit_text = "\n".join(action_log)
        
    out_path = os.path.join(tempfile.gettempdir(), "cleaned_data.csv")
    ddf_final.to_csv(out_path, single_file=True, index=False)
    
    return report, status_msg, total_rows, dupe_count, fig, audit_text, styled_raw_df, ddf_final.head(50), out_path

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Data Profiler and Cleaner")
    gr.Markdown("Inspects datasets and uses K-Nearest Neighbors to impute missing values.")
    
    with gr.Row():
        with gr.Column(scale=1):
            file_in = gr.File(label="Upload File")
            url_in = gr.Textbox(label="Or Paste Web Link")
            run_btn = gr.Button("Inspect & Clean", variant="primary")
            status_out = gr.Textbox(label="Pipeline Status", lines=2)
            
            with gr.Row():
                rows_out = gr.Number(label="Total Rows Scanned", interactive=False)
                dupes_out = gr.Number(label="Duplicates Removed", interactive=False)
            
        with gr.Column(scale=1):
            graph_out = gr.Plot(label="Missing Data Visualization")
            gr.Markdown("### Audit Log")
            audit_out = gr.Textbox(label="Cleaning actions applied:", lines=5, interactive=False)
            
    with gr.Accordion("Advanced Report", open=False):
        report_out = gr.JSON()
            
    gr.Markdown("### Data Preview")
    with gr.Row():
        raw_df_out = gr.Dataframe(label="Raw Data", interactive=False)
        clean_df_out = gr.Dataframe(label="Cleaned Data", interactive=False)
        
    download_out = gr.File(label="Download Cleaned CSV")

    run_btn.click(
        fn=process_data,
        inputs=[file_in, url_in],
        outputs=[
            report_out, status_out, rows_out, dupes_out, 
            graph_out, audit_out, raw_df_out, clean_df_out, download_out
        ]
    )

if __name__ == "__main__":
    demo.launch()
