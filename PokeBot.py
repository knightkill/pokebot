from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
import streamlit as st

import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')


def init_database(database: str) -> SQLDatabase:
    db_uri = f'sqlite:///{database}'

    return SQLDatabase.from_uri(db_uri)


def get_sql_chain(db):
    template = """
	You are a Head of Database Administration. 
	Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.    

    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Rule 1: Respond with only SQL query.
    Rule 2: No backticks.   
    Rule 3: No code wrapping.
    
    Your turn:
    
    Question: {question}
    SQL Query:
    """
    template = """
        You are assisting a user querying a Pokemon training database. The user may ask about various attributes of Pokemons stored in the database. Provide SQL queries that answer the user's question based on the provided schema.

        Database Schema: {schema}

        Previous User Queries: {chat_history}

        User Question: "{question}"
        Write an SQL query to retrieve the relevant data:
        """

    template = """
        -- Generate an SQL query based on the user question and schema information.
        Schema: {schema}
        User Question: "{question}"
        SQL Query:
        """

    template = """
        Schema: {schema}
        Question: "{question}"
        -- Write the SQL query below this line
        """

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(model="gpt-4-0125-preview")
    # llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
    # llm = GoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=GEMINI_API_KEY, temperature=0)

    # llm = Ollama(model="phi3:latest")

    def get_schema(_):
        return db.get_table_info()

    result = (
            RunnablePassthrough.assign(schema=get_schema)
            | prompt
            | llm
            | StrOutputParser()
    )
    return result


def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)

    template = """
	You are a data analyst at a pokemon training center. You are interacting with a pokemon trainer who is asking you questions about the pokemons's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
    Please use the exact syntax as shown in the example below:
    
	---
	
	<SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}
    
    ---
    """
    # print("User Query: ", sql_chain)
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4-0125-preview", temperature=0.000000001)
    # llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.000000001)
    # llm = Ollama(model="phi3:latest")
    # llm = GoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=GEMINI_API_KEY, temperature=0)
    result = RunnablePassthrough.assign(query=sql_chain).assign(
        schema=lambda _: db.get_table_info(),
        response=lambda vars: db.run(vars["query"]),
    )
    print(result)
    chain = (
            result
            | prompt
            | llm
            | StrOutputParser()
    )
    result2 = {
        "question": user_query,
        "chat_history": chat_history,
    }
    return chain.invoke(result2)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm a Pokemon Analyst. Ask me anything about your pokemons."),
    ]

db = init_database("pokedex.sqlite")
st.session_state.db = db

# Database Config
with st.sidebar:
    st.subheader("Settings")
    st.write("This is a simple chat application using SQLite. Connect to the database and start chatting.")

    st.text_input("Database", value="pokedex.sqlite", key="Database")

    if st.button("Connect"):
        with st.spinner("Connecting to database..."):
            db = init_database(
                st.session_state["Database"]
            )
            st.session_state.db = db
            st.success("Connected to database!")

# Seperate AI and human messages
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

user_query = st.chat_input("Type a message...")

if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
        st.markdown(response)

    st.session_state.chat_history.append(AIMessage(content=response))
