import os
import time
import streamlit as st
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from langchain_pinecone import PineconeVectorStore

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


azure_embeddings = AzureOpenAIEmbeddings(
    azure_deployment="embedding",
    openai_api_version="2023-05-15",
    azure_endpoint="https://varuny.openai.azure.com/",
    api_key=os.getenv("OPENAI_API_PATH"), chunk_size=1
)

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_api_key
pc = Pinecone(api_key=pinecone_api_key)


llm = AzureChatOpenAI(deployment_name='Test1',
                      model_name='gpt-35-turbo',
                      openai_api_version='2023-07-01-preview',
                      openai_api_key='d4f878c740d749deb907a6ebc9929c0d',
                      azure_endpoint="https://varuny.openai.azure.com/")

# loading documents
def load_documents_from_directory(directory_path):
    for file in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file)

        if file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file.endswith('.docx') or file.endswith('.doc'):
            loader = Docx2txtLoader(file_path)
        else:
            continue 

        loaded_docs = loader.load()
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
        docs=text_splitter.split_documents(loaded_docs)
        
    return docs


# List of directories to process
directories = [
    "C:\\Users\\shamb\\Downloads\\DocPrompt\\pdf-tables",
    "C:\\Users\\shamb\\Downloads\\DocPrompt\\chicago-principal-agreement"
]


def get_vector_store(docs,directory):
    pinecone = PineconeVectorStore.from_documents(
    docs,embedding=azure_embeddings, index_name=get_index_name(directory)
    )
    print("\nembeddings created and stored in pinecone vectorstore\n")
    return pinecone


def get_index_name(directory):
    index_name = os.path.basename(directory).replace(' ', '_').lower()
    # print(f"index name = {index_name}\n")
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    # print(f"existing = {existing_indexes}\n")

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
    
    return index_name

prompt_template = """

Human: Construction companies use unions to hire workers. They have their own set of rules for break off which contains 'Meal Breaks' and 'Rest Periods'. They also consist of Overtime Pay rules - 'Overtime Pay on Single Shifts' and 'Daily Overtime Pay'. They also consist of 'Show up' also called as 'Minimum Hours Pay' rules. The following pieces of context belong to the contracts of union workers. Use these union contracts to extract below rules
break 
overtime (overtime pay of single shift and daily overtime pay)
show up also called as minimum hours pay
Which rule needs to be extracted is mentioned in the question at the end. Please extract as many relevant rules as you can. Refer to the tables if present in the documents and extract all necessary data. Give the answer in points and also number the points. Dont make any sub-headings. Also mention the source or name of the document which you have used to give the answer, separately. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the word limit minimum 180-200 words. If you dont have enough information, use what you have but dont repeat the same point again to satisfy the word limit.
{context}
</context

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_response_llm(llm,vectorstore_pinecone,query):
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_pinecone.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer=qa.invoke({"query":query})
    return answer['result']


def main():
    for directory in directories:
        vectorstore = get_vector_store(load_documents_from_directory(directory),directory)
        print(f"The following information has been taken from {get_index_name(directory)}.\n")
        print(get_response_llm(llm,vectorstore,query="Give me the break rules for workers"))


if __name__ == "__main__":
    main()