import os
import sys
import constants

os.environ["OPENAI_API_KEY"] = constants.APIKEY

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain


from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(temperature=0.2,model_name="gpt-3.5")

# Data Ingestion
from langchain.document_loaders import DirectoryLoader
pdf_loader = DirectoryLoader('./data/', glob="**/*.pdf")
excel_loader = DirectoryLoader('./data/', glob="**/*.xlsx")
word_loader = DirectoryLoader('./data/', glob="**/*.docx")
text_loader = DirectoryLoader('./data/', glob="**/*.txt")
loaders = [pdf_loader, excel_loader,word_loader,text_loader]
documents = []
for loader in loaders:
    documents.extend(loader.load())

# Chunk and Embeddings
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)

# Initialise Langchain - Conversation Retrieval Chain
qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.2), vectorstore.as_retriever())

# Front end web app
import gradio as gr
with gr.Blocks(css=".gradio-container {background: 'black')}") as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Send a message")
    clear = gr.Button("Clear")
    chat_history = []
    
    def user(user_message, history):
        # Convert Gradio's chat history format to LangChain's expected format
        langchain_history = [(msg[1], history[i+1][1] if i+1 < len(history) else "") for i, msg in enumerate(history) if i % 2 == 0]
        # Get response from QA chain
        response = qa({"question": user_message, "chat_history": langchain_history})

        # Append user message and response to chat history
        history.append((user_message, response["answer"]))
        return gr.update(value=""), history
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False)
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(debug=True)
