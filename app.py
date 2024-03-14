import streamlit as st
from dotenv import load_dotenv
import streamlit as st
from langchain_community.vectorstores import Qdrant # used for creating the Qdrant Vector Object
from langchain_community.embeddings.openai import OpenAIEmbeddings # used for embeddings 
from langchain.text_splitter import CharacterTextSplitter
from qdrant_client.http import models
from langchain_core.runnables import RunnablePassthrough , RunnableParallel
from langchain.schema.runnable import RunnablePassthrough
from langchain.llms import OpenAI
from qdrant_client import QdrantClient
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.chains import RetrievalQA
from langchain.schema import retriever
import qdrant_client
import os


load_dotenv()  # take environment variables from .env.

OPENAI_API_KEY=st.secrets["OPENAI_API_KEY"]
llm = ChatOpenAI(temperature=0,api_key=OPENAI_API_KEY)

def get_vector_store():
    
    client = qdrant_client.QdrantClient(
        os.getenv("QDRANT_HOST"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    
    QDRANT_COLLECTION_NAME= "Personas"

    embeddings = OpenAIEmbeddings()


    vector_store = Qdrant(
    client=client,
    collection_name="Personas",
    embeddings=embeddings,
)
    
    return vector_store

# create vector store
vector_store = get_vector_store()

retriever = vector_store.as_retriever()

template = """"You are an strategizing AI agent for industrial projects 
you provide insights which are categorized based on the personas provided to you. 
Your insights should reflect proper industry experience of the persona
 Please include the personas employed in the team composition.
     {context}

Question: {question}
"""

 
prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI()

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)


def main():
    load_dotenv()
    
    st.set_page_config(page_title="Ask Question")
    st.header("Ask your query 💬")
    
 
    # create chain 
    chain = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )



    # show user input
    user_question = st.text_input("Ask a question about your text:")
    if user_question:
        st.write(f"Question: {user_question}")
        answer = chain.run(user_question)
        st.write(f"Answer: {answer}")
    
        
if __name__ == '__main__':
    main()