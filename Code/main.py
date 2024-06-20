import streamlit as st
from PyPDF2 import PdfReader
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# Function to read PDF
def read_pdf(file):
    pdfReader = PdfReader(file)
    count = len(pdfReader.pages)
    all_page_text = ""
    for i in range(count):
        page = pdfReader.pages[i]
        all_page_text += page.extract_text()
    return all_page_text

# Function to handle document upload and processing
def doc_uploader():
    document = None
    uploaded_file = st.sidebar.file_uploader('Choose a PDF file', type=['pdf'])
    if uploaded_file is not None:
        document = read_pdf(uploaded_file)
    return document

# Sidebar for NLP tasks
st.sidebar.title('NLP Tasks')

# Document upload and processing
document = doc_uploader()

if document:
    st.text_area("Document Text", document, height=300)

    # Load the fine-tuned model
    model_name = "./fine_tuned_model"
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

    st.write("### Ask a question based on the document:")
    question = st.text_area(label='Write your question')
    if question:
        QA_input = {
            'question': question,
            'context': document
        }
        res = nlp(QA_input)
        st.write(f"Answer: {res['answer']}")
