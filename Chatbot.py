import os
import streamlit as st
from dotenv import load_dotenv

# --- Core LangChain Imports ---
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
# Import the specific Streamlit chat message histories
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables.history import RunnableWithMessageHistory

#  1. SETUP and CONFIGURATION
# -----------------------------
# Loading the API keys from the environment variable
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = "sql-rag"

# Main Streamlit Page
st.set_page_config(page_title="SQL Tutor", page_icon="🤖")
st.title("🤖 SQL Tutor ")
st.write("Ask me any questions about the SQL concepts from Emory's Big Data course notes!")

# 2. SESSION STATE INITIALIZATION FOR DISPLAY MESSAGES
# -----------------------------------------------------

# `st.session_state` is a special Streamlit dictionary that stores variables
# across different runs of your script. This ensures data (like chat history)
# doesn't disappear every time the user interacts with the UI such as asking
# a new question in the chat.

# 'messages' holds all chat messages for displaying in the Streamlit UI.
# If it doesn't exist yet (first run), initialize it with a welcome message from the assistant.
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your SQL Tutor. How can I help you today?"}]

# 3. CACHED RAG COMPONENTS
# -------------------------
# `@st.cache_resource` is a decorator that tells Streamlit to run the decorated
# function only once and store its result. When the function is called again with
# the same arguments, Streamlit returns the stored result instead of re-running the function.
# This saves time and resources.

@st.cache_resource
def get_retriever():
    """Initializes and caches the Pinecone vector store retriever.
    This connects to the Pinecone vector db to search for relevant SQL notes """
    if not PINECONE_API_KEY or not OPENAI_API_KEY:
        st.error("API keys for Pinecone or OpenAI are not set. Please check your .env file.")
        st.stop()
    # The embeddings are numerical representations of the text using the OpenAI model.
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
    vectorstore = PineconeVectorStore.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)
    # This converts the vector store into a retriever that can find the top 4 most relevant documents.
    return vectorstore.as_retriever(search_kwargs={"k": 4})

@st.cache_resource
def get_llm():
    """Initializes and caches the ChatOpenAI language model."""
    return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.6, api_key=OPENAI_API_KEY)

@st.cache_resource
def get_memory():
    """
    Initializes and caches the StreamlitChatMessageHistory object.
    This `key="langchain_chat_history"` component of langchain
    connects directly to Streamlit's `st.session_state`
    telling it where to store the history.
    """
    return StreamlitChatMessageHistory(key="langchain_chat_history")

def format_docs(docs: list[Document]) -> str:
    """Helper function to combine a list of documents (your SQL notes) into one single text string.
    This string will be given to the AI as "context"."""
    return "\n\n".join(doc.page_content for doc in docs)

# 4. USE THE CACHED RAG COMPONENTS AND BUILD THE CHAIN
# ----------------------------------------------------
retriever = get_retriever()
llm = get_llm()
memory = get_memory()  # Get the cached memory instance

# Defining the prompt outside the chain for clarity.
# This is like a "blueprint" for the message
# that will be sent to the AI. It includes:
# - A "system" instruction (what the AI should act like - a helpful SQL tutor).
# - A `MessagesPlaceholder` for `chat_history`: This is where the ongoing conversation
#   will be automatically inserted by LangChain.
# - A "human" instruction: This is where the retrieved "context" and the "question"
#   will be placed.

prompt_template = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful SQL tutor. Use only the provided context. If the answer is not in the context, say you don't know."),
    # This `variable_name` must match `history_messages_key` in RunnableWithMessageHistory
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Context:\n{context}\n\nQuestion: {question}")
])

# Define the "base" RAG chain. This chain processes the question and gets the context.
# `RunnablePassthrough.assign` does this:
# It takes the input it receives (which will already contain the user's "question"
# and the "chat_history" from `RunnableWithMessageHistory`) and then *adds* the
# "context" obtained from the retriever. It ensures all necessary parts are
# present for the `prompt_template`.
base_rag_chain = (
    RunnablePassthrough.assign(
        context=lambda x: format_docs(retriever.invoke(x["question"]))
    )
    | prompt_template
    | llm
    | StrOutputParser()
)

# Then we wrap the base_rag_chain with RunnableWithMessageHistory for session management.
# 1. Loading the chat history from the `memory` object before the `base_rag_chain` runs.
# 2. Passing the correct input ('question') to the `base_rag_chain`.
# 3. Saving the user's question and the AI's response back into the `memory` object after the chain runs.
conversational_rag_chain = RunnableWithMessageHistory(
    base_rag_chain,
    lambda session_id: memory, # This lambda function provides the message history object
    input_messages_key="question", # This key specifies which part of the input dict is the new user message
    history_messages_key="chat_history", # This must match the variable_name in MessagesPlaceholder
)

# 5. DISPLAY CHAT HISTORY
# -----------------------
# # Loop through all messages currently stored in `st.session_state.messages`
# and display them in the chat interface.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. HANDLE USER INPUT
# ---------------------
# This section manages what happens when a user types something into the chat box.

if user_prompt := st.chat_input("Ask a question about your SQL notes..."):
    # Append user message to display history meaning add the user's question to `st.session_state.messages`
    # and display in streamlit UI.
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.spinner("Thinking..."):
        # This is the order of steps when the bot is thinking:
        # 1. `RunnableWithMessageHistory` gets the current chat history from `memory`.
        # 2. It combines the `user_prompt` (as 'question') and the `chat_history`.
        # 3. This combined input goes to `base_rag_chain`.
        # 4. `base_rag_chain` retrieves relevant 'context' from Pinecone.
        # 5. All three (question, history, context) are sent to the LLM via `prompt_template`.
        # 6. The LLM generates the `response`.
        # The `session_id` is required by LangChain for history management; "test_session"
        # works fine for a single-user app like this.
        response = conversational_rag_chain.invoke(
            {"question": user_prompt},
            config={"configurable": {"session_id": "test_session"}}
        )

    # Once the AI's `response` is received:
    # 1. Create a dictionary for the assistant's message
    assistant_message = {"role": "assistant", "content": response}
    # 2. Add it to `st.session_state.messages` for display.
    st.session_state.messages.append(assistant_message)
    # 3. Display the assistant's response in the Streamlit UI.
    with st.chat_message("assistant"):
        st.markdown(response)
