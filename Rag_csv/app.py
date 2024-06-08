from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
# Create OpenAI LLM model
llm = OpenAI(temperature=0.6)
# print(llm.invoke("What is capital of India?"))
# # Initialize instructor embeddings using the OllamaEmbeddings
embeddings = OllamaEmbeddings()
vectordb_file_path = "D:\\Daval\\Rag_csv"

def create_vector_db():
    # Load data from FAQ sheet
    loader = CSVLoader(file_path='source_faqs.csv', source_column="prompt")
    data = loader.load()

    # Create a FAISS instance for vector database from 'data'
    vectordb = FAISS.from_documents(documents=data,
                                 embedding=OllamaEmbeddings())

    # Save vector database locally
    vectordb.save_local(vectordb_file_path)


def get_qa_chain():
    # Load the vector database from the local folder
    vectordb = FAISS.load_local(vectordb_file_path, embeddings, allow_dangerous_deserialization=True)

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain

if __name__ == "__main__":
    create_vector_db()
    chain = get_qa_chain()
    print(chain("Do you have javascript course?"))