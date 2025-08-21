import os
import time
import json
import tempfile
import streamlit as st
import re
from dotenv import load_dotenv
from openai import OpenAI
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import pandas as pd

# Load environment variables
load_dotenv()
mistral_api_key = os.getenv("MISTRAL_API_KEY")

# Check if the API key is set
if not mistral_api_key:
    st.error("Mistral API key not found in environment variables.")
    st.stop()

# Initialize Mistral client
client = OpenAI(api_key=mistral_api_key, base_url="https://api.mistral.ai/v1")

# ----------- File Processing -----------
def extract_text_from_pdf(file):
    """Extract text from PDF using PyPDF2 and OCR fallback with pdf2image + Tesseract."""
    text = ""
    try:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    except Exception:
        text = ""

    if not text.strip():
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        images = convert_from_path(tmp_path)
        for img in images:
            text += pytesseract.image_to_string(img)
        os.remove(tmp_path)

    return text.strip()

def extract_text_from_image(file):
    """Extract text from image using Tesseract."""
    image = Image.open(file)
    return pytesseract.image_to_string(image).strip()

# ----------- AI Chat -----------
def chat_with_mistral(text, selected_fields, model="mistral-small-latest"):
    # Build dynamic prompt using user-selected + custom fields
    fields_list = "\n".join([f"- {f}" for f in selected_fields])
    example_json = "{\n" + ",\n".join([f'  "{f}": "value"' for f in selected_fields]) + "\n}"
    
    prompt = f"""
        You are an information extraction system.
        
        Extract ONLY these fields from the given invoice text:
        {fields_list}
        
        Return the result as **valid JSON only**, without explanations, markdown, or any extra text. 
        If data is missing, use null.
        
        Example:
        {example_json}
        
        Now extract from this text:
        {text}
        """

    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return completion.choices[0].message.content

def parse_response_to_table(response, selected_fields):
    # Normalization helper (reuse the same one)
    def normalize_field_name(name: str) -> str:
        return " ".join(word.capitalize() for word in name.strip().split())

    try:
        # Remove code fences if present
        response_clean = re.sub(r"^```json|```$", "", response.strip(), flags=re.MULTILINE).strip()
        data = json.loads(response_clean)

        if isinstance(data, dict):
            data = [data]  # wrap single object in list

        df = pd.DataFrame(data)

        # --- Normalize column names to match selected_fields ---
        col_map = {col: normalize_field_name(col) for col in df.columns}
        df = df.rename(columns=col_map)

        # Ensure DataFrame has all selected fields in order
        for f in selected_fields:
            if f not in df.columns:
                df[f] = None
        df = df[selected_fields]

        return df

    except Exception:
        st.warning("⚠️ Failed to parse JSON. Falling back to raw text parsing.")
        # Fallback: try to interpret as Markdown-like table
        lines = response.split("\n")
        rows = []
        for line in lines[1:]:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) == len(selected_fields):
                rows.append(dict(zip(selected_fields, parts)))
        return pd.DataFrame(rows)

def save_file(df, file_format="excel"):
    if file_format == "excel":
        output_file = "extracted_data.xlsx"
        df.to_excel(output_file, index=False)
    else:
        output_file = "extracted_data.csv"
        df.to_csv(output_file, index=False)
    return output_file

# ----------- Streamlit UI -----------
def main():
    st.set_page_config(page_title="Exxelify - Convert PDF to Excel", page_icon="📄", layout="wide")
    st.sidebar.title("Excelify - Convert PDF/Images to Excel")
    st.sidebar.caption("Explore Excelify, a cutting-edge tool revolutionizing document conversions. Instantly transform PDFs and images into editable Excel files, streamlining data extraction and analysis. Simplify workflows and enhance productivity with our AI-powered solution, ensuring accuracy and efficiency in data management tasks.")
    st.sidebar.markdown("Made with ❤️ in India")

    st.header("📄 Excelify with Mistral: Extract Invoices/Bills into Excel/CSV")

    # Field selection
    default_fields = ["Invoice Number", "Invoice Date", "Company Name", "Company GST Number",
                      "Customer Name", "Customer GST Number", "Total Quantity", "Total Amount"]
    selected_fields = st.multiselect("Select fields to extract:", default_fields, default=default_fields)

    # Let user add custom fields
    custom_fields_input = st.text_input("Add custom fields (comma separated):", "")

    if custom_fields_input:
        custom_fields = [f.strip() for f in custom_fields_input.split(",") if f.strip()]
        selected_fields.extend(custom_fields)
        selected_fields = list(dict.fromkeys(selected_fields))  # remove duplicates
    
    # --- Normalize fields ---
    def normalize_field_name(name: str) -> str:
        # Trim whitespace, lowercase everything, then capitalize each word
        return " ".join(word.capitalize() for word in name.strip().split())
    
    # Apply normalization + deduplicate while preserving order
    normalized_fields = []
    for f in selected_fields:
        nf = normalize_field_name(f)
        if nf not in normalized_fields:
            normalized_fields.append(nf)
    
    selected_fields = normalized_fields

    file_option = st.radio("Upload Mode", ["Single", "Multiple"], index=0, horizontal=True)

    all_text = ""
    dfs = []

    if file_option == "Single":
        file = st.file_uploader("Upload your file (PDF/Image)", type=["pdf", "jpeg", "jpg", "png"])
        if file:
            ext_text = extract_text_from_pdf(file) if file.type == "application/pdf" else extract_text_from_image(file)
            if ext_text:
                response = chat_with_mistral(ext_text, selected_fields)
                df = parse_response_to_table(response, selected_fields)
                st.subheader("Preview Extracted Data")
                st.dataframe(df)
                dfs.append(df)
    else:
        files = st.file_uploader("Upload multiple files", accept_multiple_files=True, type=["pdf", "jpeg", "jpg", "png"])
        if files:
            for file in files:
                st.write(f"Processing {file.name} ...")
                ext_text = extract_text_from_pdf(file) if file.type == "application/pdf" else extract_text_from_image(file)
                if ext_text:
                    response = chat_with_mistral(ext_text, selected_fields)
                    df = parse_response_to_table(response, selected_fields)
                    dfs.append(df)
            if dfs:
                final_df = pd.concat(dfs, ignore_index=True)
                st.subheader("Preview Extracted Data")
                st.dataframe(final_df)
                dfs = [final_df]

    # Download buttons
    if dfs:
        final_df = dfs[0] if len(dfs) == 1 else pd.concat(dfs, ignore_index=True)
        col1, col2 = st.columns(2)
        with col1:
            excel_file = save_file(final_df, "excel")
            with open(excel_file, "rb") as f:
                st.download_button("Download Excel", f, file_name="extracted_data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with col2:
            csv_file = save_file(final_df, "csv")
            with open(csv_file, "rb") as f:
                st.download_button("Download CSV", f, file_name="extracted_data.csv", mime="text/csv")

if __name__ == "__main__":
    main()
