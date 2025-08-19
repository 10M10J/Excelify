import os
import time
import json
import tempfile
import streamlit as st
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
    prompt = f"""
    Extract the following fields from the provided invoice/bill text: {', '.join(selected_fields)}.
    Respond ONLY in strict JSON array of objects where each object represents one invoice.
    Example format:
    [
      {{
        "Invoice Number": "12345",
        "Invoice Date": "2024-01-15",
        "Company Name": "ABC Ltd",
        ...
      }}
    ]
    Text to extract from:
    {text}
    """

    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return completion.choices[0].message.content

def parse_response_to_table(response, selected_fields):
    try:
        data = json.loads(response)
        if isinstance(data, dict):
            data = [data]
        df = pd.DataFrame(data)
    except Exception:
        st.warning("Failed to parse JSON properly, falling back to raw text parsing.")
        lines = response.split("\n")
        rows = []
        for line in lines[1:]:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) == len(selected_fields):
                rows.append(dict(zip(selected_fields, parts)))
        df = pd.DataFrame(rows)
    return df

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
    st.sidebar.title("Excelify - Convert PDF/Images to Excel")
    st.sidebar.markdown("Made with ❤️ by [StirPot](https://stirpot.in/)")

    st.header("📄 Excelify with Mistral: Extract Invoices/Bills into Excel/CSV")

    # Field selection
    default_fields = ["Invoice Number", "Invoice Date", "Company Name", "Company GST Number",
                      "Customer Name", "Customer GST Number", "Total Quantity", "Total Amount"]
    selected_fields = st.multiselect("Select fields to extract:", default_fields, default=default_fields)

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
