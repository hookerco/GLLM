from langchain.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import HuggingFaceHub
from glob import glob
import streamlit as st
import os
import textwrap

os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["huggingface_token"]
pdf_data_dir = 'pdfs'     # same as in train_pipeline.py
vector_db_dir = "faiss_index"


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


embeddings = HuggingFaceEmbeddings()

# if a vector database was already created, just load it instead of creating a new one
if os.path.exists(vector_db_dir):
    db = FAISS.load_local(vector_db_dir, embeddings)
else:
    filepaths = glob(pdf_data_dir + '/**/*.pdf', recursive=True)

    # Load data
    loaders = [PyPDFLoader(filepath) for filepath in filepaths]
    documents = []
    for loader in loaders:
        documents.extend(loader.load())

    # Split documents to fit in context window of model
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(documents)

    # Store chunks (after embedding) in vector database
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(vector_db_dir)

# Retrieve mechanism (return relevant Documents given a string query using similarity search)
# Tip: you can also try different types of retrievers
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# Example usage:
# retrieved_docs = retriever.invoke("Sample query")
# print(retrieved_docs[0].page_content)

# todo: replace later with finetuned model
llm = HuggingFaceHub(repo_id='HuggingFaceH4/zephyr-7b-beta',
                     model_kwargs={"temperature": 0.9, "max_length": 500})

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.

{context}

Question: {question}

Helpful Answer:"""

custom_rag_prompt = PromptTemplate.from_template(template)

# Chain takes a question, retrieves relevant documents,
# constructs a prompt, passes that to a model, and parses the output
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)

# todo next steps: add chat history https://python.langchain.com/docs/use_cases/question_answering/chat_history

# Testing
prompt_without_rag = PromptTemplate.from_template("""Answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.

Question: {question}

Helpful Answer:""")

sample_q = "Was ist der HELLER Lernfabrik?"

chain_without_rag = {"question": RunnablePassthrough()} | prompt_without_rag | llm | StrOutputParser()
plain_a = chain_without_rag.invoke(sample_q)
# textwrap just to limit number of words in each line
rag_a = '\n'.join(textwrap.wrap(rag_chain.invoke(sample_q), 100))

print(f'Q: {sample_q}\n')
print(f'A (with RAG): {rag_a}\n')
print(f'A (without RAG): {plain_a}\n')
