from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader, PyPDFLoader
from langchain_community.document_loaders.word_document import Docx2txtLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os

def load_resumes():
    print("Loading resumes")
    documents = []  
    #doc_loader = PyPDFDirectoryLoader("resumes/")

    for file in os.listdir("resumes/"):
        #if file.endswith(".pdf"):
        doc_loader = PyPDFLoader(f"resumes/{file}")
        docs = doc_loader.load()
        document_text = ""
        for document in docs:
            #print(document)
            document_text += document.page_content.replace("\n", " ")

        documents.append(document_text)
    return documents

def load_job_description():
    with open("job_description.txt", "r") as f:
        job_description = f.read()
    return job_description

resumes = load_resumes()
job_desc = load_job_description()

llm = ChatOllama(model="llama3.1",temperature=0.5)


resume_review_prompt = ChatPromptTemplate.from_messages(
        [
        ("system", "You are assisting a company in recruiting. You are thoughtful in your responses"),
        ("user", """
        You will be given a resume and the company's job description. 
         
        Here is the resume:
        {context}

        ========================================
         
        Here is the job description:
        {job_description}
         
        ========================================
        You must say whether the resume is a good fit for the job. Only answer 'yes' or 'no' 
        UNLESS the user asks an additional question below. Then you must answer both the yes or no question
         AND the user's question. Saying No will not have any negative consequences for the candidate.
        {q}
        """)
        ]
        )

print(len(resumes))
print(resumes[0])



#llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)


while True:
    q = input("Enter a question: ")
    chain = (
    {"context": lambda x: resumes[0], "job_description": lambda y: job_desc, 'q':RunnablePassthrough()}
    | resume_review_prompt
    | llm
    | StrOutputParser()
    )  
    print(chain.invoke(q))