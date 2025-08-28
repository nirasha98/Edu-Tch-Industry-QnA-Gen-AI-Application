from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os

from dotenv import load_dotenv

load_dotenv() # read API key from .env file

# Creating Google Palm Model
llm = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.1)

# Initialize Instructor Embeddings using HuggingFace model
instructure_embeddings= HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large" )
vectorDB_file_path = "faiss_index"

def create_vector_DB():

    #Load data from csv
    loader= CSVLoader(file_path='codebasics_faqs.csv',source_column="prompt")
    data = loader.load()

    #create a FAISS instance for vector database from 'data'
    vectorDB= FAISS.from_documents(documents=data,embedding=instructure_embeddings)

    #save vector database to local disk
    vectorDB.save_local(vectorDB_file_path)


def get_QA_Chain():

    # Load the vector database from local disk
    vectorDB= FAISS.load_local(vectorDB_file_path,instructure_embeddings)

    #create a retriever for querying the vector database
    retriever = vectorDB.as_retriever(score_threshold=0.7)

    prompt_template= """ Given the following context and a question,generate an answer based on this context only.
     In the answer try to provide as much text as possible from "response" section in the source document context without making much changes
       If the answer is not found in the context, kindly state "I don't know.Please put a email wityour Question for further details. Email= abc@gmail.com " Don't try to make up an answer.
    
    CONTEXT: {context}

    QUESTION : {question}  """

    Prompt = PromptTemplate(template = prompt_template,input_variable=["context","question"])

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type= "stuff",
                                        input_key="query",
                                        return_source_documents= True,
                                        chain_type_kwargs={"prompt": Prompt}
                                        )
    return chain

if __name__=="__main__":
    create_vector_DB()
    chain= get_QA_Chain()
    print(chain("Do you have ML Course?"))



# from langchain_community.vectorstores import FAISS
# from langchain_community.llms import GooglePalm
# from langchain.document_loaders.csv_loader import CSVLoader
# from langchain_community.embeddings import HuggingFaceInstructEmbeddings
# from langchain.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# import os
# from dotenv import load_dotenv

# load_dotenv()  # read API key from .env file

# # Creating Google Palm Model
# llm = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.1)

# # Initialize Instructor Embeddings using HuggingFace model
# instructure_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
# vectorDB_file_path = "faiss_index"

# def create_vector_DB():
#     # Load data from csv
#     loader = CSVLoader(file_path='codebasics_faqs.csv', source_column="prompt")
#     data = loader.load()

#     # Create a FAISS instance for vector database from 'data'
#     vectorDB = FAISS.from_documents(documents=data, embedding=instructure_embeddings)

#     # Save vector database to local disk
#     vectorDB.save_local(vectorDB_file_path)

# def get_QA_Chain():
#     # Load the vector database from local disk
#     vectorDB = FAISS.load_local(vectorDB_file_path, instructure_embeddings)

#     # Create a retriever for querying the vector database
#     retriever = vectorDB.as_retriever(score_threshold=0.7)

#     prompt_template = """Given the following context and a question, generate an answer based on this context only.
#     In the answer try to provide as much text as possible from the "response" section in the source document context without making much changes.
#     If the answer is not found in the context, kindly state "I don't know. Please put an email with your question for further details. Email= abc@gmail.com." Don't try to make up an answer.

#     CONTEXT: {context}

#     QUESTION: {question}"""

#     Prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

#     chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         input_key="query",
#         return_source_documents=True,
#         chain_type_kwargs={"prompt": Prompt}
#     )
#     return chain

# if __name__ == "__main__":
#     create_vector_DB()
#     chain = get_QA_Chain()
#     print(chain("Do you have ML Course?"))
