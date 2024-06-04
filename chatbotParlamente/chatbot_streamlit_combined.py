import streamlit as st
import os
import OpenAI_utility
import torch
import time
import gc
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from streamlit.delta_generator import DeltaGenerator  # Import DeltaGenerator for updating messages

#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#progress_bar
#main_placeholder = st.empty()
#def main_place(message="The Task Is Finished !!!!"):
  # main_placeholder.text(message)

# Memory management functions
def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()

def wait_until_enough_gpu_memory(min_memory_available, max_retries=10, sleep_time=5):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(torch.cuda.current_device())

    for _ in range(max_retries):
        info = nvmlDeviceGetMemoryInfo(handle)
        if info.free >= min_memory_available:
            break
        print(f"Waiting for {min_memory_available} bytes of free GPU memory. Retrying in {sleep_time} seconds...")
        time.sleep(sleep_time)
    else:
        raise RuntimeError(f"Failed to acquire {min_memory_available} bytes of free GPU memory after {max_retries} retries.")

def main():
    # Call memory management functions before starting Streamlit app
    #min_memory_available = 1 * 1024 * 1024 * 1024  # 1GB
    clear_gpu_memory()
    #wait_until_enough_gpu_memory(min_memory_available)

    #NOTA: l'ordine delle chiamate nel main ci garantisce che display_document_embedding_page() 
    #venga eseguita e completata prima che display_chatbot_page() venga chiamata.
    #Siamo dunque certi che il vector sia gia presente o che venga creato prima di interagire col chatbot
    display_document_embedding_page()
    
    display_chatbot_page()
   



def display_chatbot_page():

    st.title("Multi Source Chatbot")

    # CSS to right-align user messages
    st.markdown(
        """
        <style>
        .user-message-container {
            display: flex;
            flex-direction: row-reverse; /* Reverse the order of elements */
            align-items: flex-end; /* Align items to the bottom */
            margin-bottom: 10px;
        }
        .user-message {
            text-align: right;
            background-color: #dcf8c6;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
        }
        .assistant-message {
            text-align: left;
            background-color: #f1f0f0;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Prepare the LLM model
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    VECTOR_STORE = "first_CameraDep"
    TEMPERATURE = 0.5
    MAX_LENGTH = 300

    st.session_state.conversation = OpenAI_utility.prepare_rag_llm(
        VECTOR_STORE, TEMPERATURE, MAX_LENGTH
    )

    # Chat history
    if "history" not in st.session_state:
        st.session_state.history = []

    # Source documents
    if "source" not in st.session_state:
        st.session_state.source = []

    # Display chats
    for message in st.session_state.history:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(f'<div class="user-message-container"><div class="user-message">{message["content"]}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-message">{message["content"]}</div>', unsafe_allow_html=True)

    # Ask a question
    if question := st.chat_input("Fammi una domanda"):
        # Append user question to history
        st.session_state.history.append({"role": "user", "content": question})
        # Add user question
        with st.chat_message("user"):
            st.markdown(f'<div class="message-container"><div class="user-message">{question}</div></div>', unsafe_allow_html=True)
            
            
        # Answer the question
        answer, doc_source = OpenAI_utility.generate_answer(question)
        #risposta intera
        #with st.chat_message("assistant"):
            #st.write(answer)

        # Display the answer one word at a time
        with st.chat_message("assistant") as assistant_message:
            words = answer.split()
            message_placeholder = st.empty()
            for i in range(len(words)):
                message_placeholder.markdown(" ".join(words[:i+1]))
                time.sleep(0.06)  # Adjust the sleep time to control the speed of word display


        # Append assistant answer to history
        st.session_state.history.append({"role": "assistant", "content": answer})

        # Append the document sources
        st.session_state.source.append({"question": question, "answer": answer, "document": doc_source})


    # Source documents
    with st.expander("Chat History and Source Information"):
        st.write(st.session_state.source)



def display_document_embedding_page():

    #se il vector_store nominato first_CameraDep esiste gia, non lo ricreo, se invece non esiste allora lo creo con lo stesso nome
    if not os.path.exists("vector store/first_CameraDep"):

        print("assente")

        #richiamo read_pdf
        combined_content = OpenAI_utility.read_pdf("merged_RegistroCmeraDeputati.pdf")
        #splitto il documento con chunk-size=330 e overlapping=50
        split = OpenAI_utility.split_doc(combined_content, 330, 50)
        #creo il vector_store
        OpenAI_utility.embedding_storing(split, True, "first_CameraDep", "first_CameraDep")


if __name__ == "__main__":
    main()
