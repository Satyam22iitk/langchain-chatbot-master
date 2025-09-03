import streamlit as st
from streamlit.logger import get_logger
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.llms import HuggingFaceHub

logger = get_logger('Langchain-Chatbot')

# decorator
def enable_chat_history(func):
    # to clear chat history after switching chatbot
    current_page = func.__qualname__
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = current_page
    if st.session_state["current_page"] != current_page:
        try:
            st.cache_resource.clear()
            del st.session_state["current_page"]
            del st.session_state["messages"]
        except:
            pass

    # to show chat history on ui
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

    def execute(*args, **kwargs):
        func(*args, **kwargs)
    return execute

def display_msg(msg, author):
    """Method to display message on the UI

    Args:
        msg (str): message to display
        author (str): author of the message - user/assistant
    """
    st.session_state.messages.append({"role": author, "content": msg})
    st.chat_message(author).write(msg)

def configure_llm():
    available_llms = ["flan-t5-base (HF)", "mistral-7b (HF)"]
    llm_opt = st.sidebar.radio(
        label="Select HuggingFace Model",
        options=available_llms,
        key="SELECTED_HF_MODEL"
    )

    if llm_opt == "flan-t5-base (HF)":
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-base",
            task="text2text-generation",
            huggingfacehub_api_token=st.secrets["HUGGINGFACEHUB_API_TOKEN"],
            model_kwargs={"temperature": 0.7, "max_length": 512}
        )
    elif llm_opt == "mistral-7b (HF)":
        llm = HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            task="text-generation",
            huggingfacehub_api_token=st.secrets["HUGGINGFACEHUB_API_TOKEN"],
            model_kwargs={"temperature": 0.7, "max_new_tokens": 512}
        )
    else:
        st.error("Please select a model")
        st.stop()

    return llm



def print_qa(cls, question, answer):
    log_str = "\nUsecase: {}\nQuestion: {}\nAnswer: {}\n" + "------"*10
    logger.info(log_str.format(cls.__name__, question, answer))

@st.cache_resource
def configure_embedding_model():
    embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return embedding_model

def sync_st_session():
    for k, v in st.session_state.items():
        st.session_state[k] = v
