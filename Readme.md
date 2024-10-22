# PDF Answering AI

## Project Overview

This project aims to develop an AI model that can answer questions based on the content of a PDF document. By leveraging advanced natural language processing (NLP) techniques, this system can interpret and provide accurate responses to user queries about the text within the PDF.

## Features

- Upload a PDF document and input questions.
- Fine-tuned BERT model for context-aware question answering.
- Supports text extraction and processing from PDF files.
- Interactive interface for user queries and responses.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)

## Installation

To set up the project environment, follow these steps:

1. **Clone the Repository**
   ```sh
   git clone https://github.com/nyatiraghav/PDF_Answering_AI.git
   cd PDF_Answering_AI
    ```
2. **Create and Activate Virtual Environment**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3. **Install Dependencies**
   ```sh
   pip install -r requirements.txt
    ```

## Usage

1. **Navigate to the Code Directory**
   ```sh
   cd Code
    ```
2. **Run the python file for fine tuning the model**
   ```sh
   python process_model.py
    ```
3. **Run the application**
   ```sh
   streamlit run main.py
    ```
**Interact with the System**

- Upload a PDF document.
- Enter your question related to the PDF content.
- Receive the answer generated by the AI model.
