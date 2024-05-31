import time
import streamlit as st
from dotenv import load_dotenv
import os
from groq import Groq
from PyPDF2 import PdfReader
import pytesseract
from PIL import Image
import pandas as pd
import tabula 

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

# Check if the API key is set
if not groq_api_key:
    st.error("GROQ API key not found in environment variables.")
    st.stop()

# Create a Groq client
client = Groq(api_key=groq_api_key)

# File upload and processing function
def process_file(file):
    file_name, file_extension = os.path.splitext(file.name)
    ext_text = ""
    if file_extension.lower() == ".pdf":
        file_reader = PdfReader(file)
        for page in file_reader.pages:
            page_text = page.extract_text()
            if page_text:
                ext_text += page_text
    elif file_extension.lower() in [".jpeg", ".jpg", ".png"]:
        image = Image.open(file)
        ext_text = pytesseract.image_to_string(image)
    else:
        st.write(f"File '{file.name}' is not a supported format.")
    return ext_text

def save_to_excel(table_data, file_name="extracted_data.xlsx"):
    # Assuming table_data is a list of dictionaries or a similar structured format
    df = pd.DataFrame(table_data)
    df.to_excel(file_name, index=False)
    # convert PDF into CSV
    # tabula.convert_into(table_data, file_name, output_format="csv", pages='all')
    return file_name

def convert_response_to_table_data(response):
    # Split the response string into lines
    lines = response.split('\n')
    
    # Initialize an empty list to store table data
    table_data = []

    # Extract column names from the first line
    columns = [col.strip() for col in lines[0].split('|') if col.strip()]

    # Iterate over the remaining lines to extract data
    for line in lines[2:]:
        # Split each line into values
        values = [value.strip() for value in line.split('|') if value.strip()]
        # Create a dictionary mapping column names to values for each row
        row_data = dict(zip(columns, values))
        # Append the row data to the table_data list
        table_data.append(row_data)
    return table_data

def chat_with_groq(client, model, text):
    prompt = f'''
        Extract Invoice number, Invoice Date, Company Name, Company GST number, Customer Name, Customer GST number, 
        Total Quantity of boxes and Total amount. Arrange these details in a table format with same column names 
        and reply only with table (no need for anything else like explanations, etc..). 
        Use information from the following text:
        {text}
    '''
    
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return completion.choices[0].message.content

def main():
    # Sidebar contents
    with st.sidebar:
        st.title('Excelify - Convert PDF/Images to Excel')
        st.markdown('''
        ## About
        Explore Excelify, a cutting-edge tool revolutionizing document conversions. 
        Instantly transform PDFs and images into editable Excel files, streamlining data extraction and analysis.
        Simplify workflows and enhance productivity with our AI-powered solution, 
        ensuring accuracy and efficiency in data management tasks.
        ''')
        st.write('Made with :sparkling_heart: by [StirPot](https://stirpot.in/).')
        model = "llama3-8b-8192"

    st.header("Unlock Data: Convert PDFs/Images to Excel with Excelify", divider='rainbow')

    file_option = st.radio('Select file upload option below:', ('Multiple', 'Single'), index=1, horizontal=True)

    ext_text = ""
    response = ""

    # Upload file(s)
    if file_option == 'Single':
        file = st.file_uploader("Upload your file (pdf or image only) ", type=['pdf', 'jpeg', 'jpg', 'png'])
        if file is not None:
            ext_text = process_file(file)
    #        st.write("Extracted Text:")
    #        st.write(ext_text)
            if ext_text and ext_text.strip():  # Ensure ext_text is not None and is not empty
                response = chat_with_groq(client, model, ext_text)
                st.write("AI Response:")
                st.write(response)
            else:
                st.write("No text extracted from the file.")
    elif file_option == 'Multiple':
        multi_file = st.file_uploader("Upload your file(s) (pdf or image only) ", accept_multiple_files=True, type=['pdf', 'jpeg', 'jpg', 'png'])
        if multi_file:
            st.write(f"{len(multi_file)} file(s) uploaded:")
            for file in multi_file:
                ext_text += process_file(file)
    #        st.write("Extracted Text:")
    #        st.write(ext_text)
            if ext_text and ext_text.strip():  # Ensure ext_text is not None and is not empty
                response = chat_with_groq(client, model, ext_text)
                st.write("AI Response:")
                st.write(response)
            else:
                st.write("No text extracted from the files.")

    if response :
        table_data = convert_response_to_table_data(response)
        excel_file_name = save_to_excel(table_data)
        st.markdown("#")
        with open(excel_file_name, "rb") as file:
            btn = st.download_button(
                    label= "Download Excel file",
                    data=file,
                    file_name=excel_file_name,
                    help="Download your converted data in Excel Format",
                    type="primary",
                    mime="text/csv"
                )
    # else:
    #     st.write("No Response for Excel")


if __name__ == '__main__':
    main()
