import os
import time
import streamlit as st
import json  
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

processed_docs_file = "processed_docs.json"

if os.path.exists(processed_docs_file):
    with open(processed_docs_file, "r") as f:
        processed_docs = json.load(f)
else:
    processed_docs = []

azure_embeddings = AzureOpenAIEmbeddings(
    azure_deployment="embedding",
    openai_api_version="2023-05-15",
    azure_endpoint="https://varuny.openai.azure.com/",
    api_key=os.getenv("OPENAI_API_PATH"), chunk_size=1
)

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

llm = AzureChatOpenAI(deployment_name='Test1',
                      model_name='gpt-35-turbo',
                      openai_api_version='2023-07-01-preview',
                      openai_api_key='d4f878c740d749deb907a6ebc9929c0d',
                      azure_endpoint="https://varuny.openai.azure.com/")

def extract_rules_from_text(text):
    rules = {
        "Break Rules": [],
        "Overtime Rules": [],
        "Show Up Rules": []
    }
    
    current_rule_type = None
    lines = text.splitlines()

    for line in lines:
        if "Break Rules:" in line:
            current_rule_type = "Break Rules"
        elif "Overtime Rules:" in line:
            current_rule_type = "Overtime Rules"
        elif "Show-Up Rules:" in line or "Show Up Rules:" in line or "Show-up Rules:" in line or "Show up Rules:" in line:
            current_rule_type = "Show Up Rules"
        elif current_rule_type and line.strip():
            rules[current_rule_type].append(line.strip())

    return rules

def save_rules_to_mvl(rules_data, output_file):
    with open(output_file, 'w') as f:
        json.dump(rules_data, f, indent=4)


def load_documents_from_directory(directory_path, processed_docs):
    new_docs = []
    files = processed_docs[:]  # Start with the already processed files

    for file in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file)

        if file in processed_docs:
            print(f"\nSkipping {file} as it has already been processed.\n")
            continue

        if file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file.endswith('.docx') or file.endswith('.doc'):
            loader = Docx2txtLoader(file_path)
        else:
            continue 

        loaded_docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        docs = text_splitter.split_documents(loaded_docs)
        
        new_docs.extend(docs) 
        files.append(file)  # contains all the files that have either been processed or are new
        processed_docs.append(file)  

    return new_docs, files
 

prompt_template = """

Human: You are a helpful assistant. Construction companies use unions to hire workers. You will be provided with a query and some documents. Construction companies' unions have their own set of rules which include 'Meal Breaks', 'Rest Periods', 'Overtime Pay on Single Shifts', 'Daily Overtime Pay', 'Show up and Minimum Hours Pay', 'Hours of Work' and 'Holiday' rules. The following context contains information from the contracts of union workers. Your task is to extract and list as many detailed rules as possible related to the query from the provided documents. Extract the following rules:

Rest Periods : Include rules related to Rest Periods.
Meal Breaks : Include rules related to Meal Breaks.
Overtime Rules: Include rules related to Overtime Pay and OVERTIME PAY ON SINGLE SHIFTS. There can be rules for different types of workers. Give me rules for all types of workers in detail.
Hours of Work : Include rules related to Hours of work including Workday, Work Shift and Workweek and Pay period.
Holiday Rules : Include rules related to Holidays, and how much pay is to be given for work on these holidays and how the holiday payment is calculated. For example, the employee will be paid for the number of hours obtained by dividing their normal
number of scheduled weekly work hours by five (5) days for each observed holiday. This is just one example, but there could be similar rules to calculate holiday payment.
Show up Pay and Minimum Hours Rules : Include rules related to Show up Pay and Minimum Hours Rules.

Make sure to extract and include all necessary data from the contract documents as specified. This data could also be available in tabular format in the provided documents. Provide the answer in numbered points. Use separate headings for each set of rules. If you don't know the answer, just state that you don't know; do not attempt to fabricate an answer. Keep the word limit to a minimum of 300 words for each set of rules. If you do not have enough information, use what you have without repeating points to meet the word limit. 
{context}
</context

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_vector_store(docs, directory):
    pinecone = PineconeVectorStore.from_documents(
        docs, embedding=azure_embeddings, index_name=get_index_name(directory)
    )
    print("\nEmbeddings created and stored in Pinecone VectorStore.\n")
    return pinecone

def get_index_name(directory):
    index_name = os.path.basename(directory).replace(' ', '_').lower()
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

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

def get_response_llm(llm, vectorstore_pinecone, query):
    retriever = vectorstore_pinecone.as_retriever(
        search_type="similarity", search_kwargs={"k": 13}
    )
    # retrieved_docs = retriever.get_relevant_documents(query)
    # for i, doc in enumerate(retrieved_docs, start=1):
    #     print(f"{i}:\n")
    #     print(f"  Text: {doc.page_content}")
    #     if hasattr(doc, 'metadata'):
    #         print(f"  Metadata: {doc.metadata}")
    #     if hasattr(doc, 'score'):
    #         print(f"  Score: {doc.score}")
    #     print("\n") 

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    answer = qa.invoke({"query": query})
    return answer['result']

def main():
    processed_docs_file = "processed_docs.json"
    output_mvl_file = "rules.mvl"

    if os.path.exists(processed_docs_file):
        with open(processed_docs_file, "r") as f:
            processed_docs = json.load(f)
    else:
        processed_docs = []

    directories = [
        "C:\\Users\\shamb\\Downloads\\DocPrompt\\pdf-tables",
        "C:\\Users\\shamb\\Downloads\\DocPrompt\\chicago-principal-agreement"
    ]

    all_rules_data = []
    i=0
    for directory in directories:
        i+=1
        docs, files = load_documents_from_directory(directory, processed_docs)
        if not docs:
            vectorstore = PineconeVectorStore.from_documents([], embedding=azure_embeddings, index_name=get_index_name(directory))
        else:
            vectorstore = get_vector_store(docs, directory)

        response = get_response_llm(llm, vectorstore, query="Give me 'Hours of Work' rules including Workday, Work Shift and Workweek and Pay period; 'meal breaks' rules, 'rest periods' rules, 'Holiday Pay' calculation rules, 'Show up Pay and Minimum Hours' rules; and 'Overtime Pay' Rules, including overtime pay on single shifts for workers. Give as many rules as you can in detail.")
        print(response)
        source = files[i-1]
        print(f"\nSource: {source}\n")
        
        rules = extract_rules_from_text(response)
        all_rules_data.append({
                "source": source,
                "rules": rules
            })

    save_rules_to_mvl(all_rules_data, output_mvl_file)
    print(f"Rules saved to '{output_mvl_file}'")
        

if __name__ == "__main__":
    main()