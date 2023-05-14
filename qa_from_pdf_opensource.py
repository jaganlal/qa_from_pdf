import os
import pandas as pd
import matplotlib.pyplot as plt

from transformers import GPT2TokenizerFast

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding

import textract

from dotenv import load_dotenv
import gradio as gr

class PDFQA():

    def __init__(self) -> None:
        super().__init__()

        os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv('HUGGINGFACEHUB_API_TOKEN')

        self.source_folder = './data/'
        self.output_folder = './output/'

    def loading_pdf():
        return 'Loading...'

    def pdf_changes(self, pdf_doc, repo_id):
        # # Simple method - Split by pages 
        # loader = PyPDFLoader(pdf_doc.name)
        # pages = loader.load_and_split()

        # # SKIP TO STEP 2 IF YOU'RE USING THIS METHOD
        # chunks = pages

        if pdf_doc is None:
            return

        doc = textract.process(pdf_doc.name)

        # Step 2: Save to .txt and reopen (helps prevent issues)
        with open(self.output_folder+'test.txt', 'w') as f:
            f.write(doc.decode('utf-8'))

        with open(self.output_folder+'test.txt', 'r') as f:
            text = f.read()

        # Step 3: Create function to count tokens
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

        def count_tokens(text: str) -> int:
            return len(tokenizer.encode(text))

        # Step 4: Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size = 512,
            chunk_overlap  = 24,
            length_function = count_tokens,
        )

        chunks = text_splitter.create_documents([text])

        embeddings = HuggingFaceEmbeddings()
        # embed_model = LangchainEmbedding(embeddings)

        # Create vector database
        db = FAISS.from_documents(chunks, embeddings)
        retriever = db.as_retriever()

        flan_ul2 = HuggingFaceHub(repo_id=repo_id, model_kwargs={'temperature': 0.1})
        # chain = load_qa_chain(flan_ul2, chain_type='stuff')
        self.chain = RetrievalQA.from_chain_type(llm=flan_ul2, chain_type='stuff', retriever=retriever, return_source_documents=True)
        return 'Ready'

    def add_text(self, history, text):
        history = history + [(text, None)]
        return history, ''

    def bot(self, history):
        response = self.infer(history[-1][0])
        history[-1][1] = response['result']
        return history

    def infer(self, question):
        
        query = question
        result = self.chain({'query': query})

        return result

if __name__ == '__main__':
    load_dotenv()

    pdf_qa = PDFQA()

    # query = 'What is the copay for Diagnostic test?'
    # query = 'what is my out of pocket limit?'
    # query = 'what is the radiology cost for having a baby'
    # query = how much will be my radiology cost for new born baby?

    css='''
    #col-container {max-width: 700px; margin-left: auto; margin-right: auto;}
    '''

    title = '''
    <div style='text-align: center;max-width: 700px;'>
        <h1>Upload a PDF document and start asking questions from the document</h1>
        <p style='text-align: center;'>Upload a pdf document from your computer, click the 'Process PDF' button, <br />
        once ready (check for the <i>Ready</i> status), you can start asking questions about the uploaded pdf document</p>
    </div>
    '''

    with gr.Blocks(css=css) as demo:
        with gr.Column(elem_id='col-container'):
            gr.HTML(title)
            
            with gr.Column():
                pdf_doc = gr.File(label='Load a pdf', file_types=['.pdf'], type='file')
                repo_id = gr.Dropdown(label='LLM', choices=['google/flan-t5-large', 'bigscience/mt0-large', 'microsoft/prophetnet-large-uncased', 'bigscience/T0_3B'], value='google/flan-t5-large')
                with gr.Row():
                    langchain_status = gr.Textbox(label='Status', placeholder='', interactive=False)
                    load_pdf = gr.Button('Process PDF')
            
            chatbot = gr.Chatbot([], elem_id='chatbot').style(height=350)
            question = gr.Textbox(label='Question', placeholder='Ask your question and hit enter')
            submit_btn = gr.Button('Send message')
    
        repo_id.change(pdf_qa.pdf_changes, inputs=[pdf_doc, repo_id], outputs=[langchain_status], queue=False)
        load_pdf.click(pdf_qa.pdf_changes, inputs=[pdf_doc, repo_id], outputs=[langchain_status], queue=False)
        question.submit(pdf_qa.add_text, [chatbot, question], [chatbot, question]).then(
            pdf_qa.bot, chatbot, chatbot
        )
        submit_btn.click(pdf_qa.add_text, [chatbot, question], [chatbot, question]).then(
            pdf_qa.bot, chatbot, chatbot
        )

    demo.launch()