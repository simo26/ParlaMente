import streamlit as st
import os
import OpenAI_utility
import torch
import time
import gc
import streamlit.components.v1 as components
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from streamlit.delta_generator import DeltaGenerator  # Import DeltaGenerator for updating messages

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#progress_bar
#main_placeholder = st.empty()
#def main_place(message="The Task Is Finished !!!!"):
  # main_placeholder.text(message)

# Memory management functions
def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()

#utility per la gestione della memoria
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
    
    #funzione per la visualizzazione del chatbot, da chiamare SOLO dopo la creazione del vectore store
    display_chatbot_page()
   
def load_css(file_name): #funzione per caricare il tema .css adatto
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def display_chatbot_page():

    #per configurare logo e titolo della pagina nel browser
    st.set_page_config(
        page_title="ParlaMente",
        page_icon="https://raw.githubusercontent.com/alessiamanna/text_mining/main/immagine.png"
    )

    #sono stati creati due temi, coffee theme con una palette di colori caldi, e custom theme con una palette di colori più classica
    coffee_theme_css = 'coffee_theme.css'
    custom_theme_css = 'custom_theme.css' 

    #caricamento del tema di default
    if 'theme' not in st.session_state:
        st.session_state.theme = coffee_theme_css


    load_css(st.session_state.theme)


    #link di riferimento per le icone
    bot = "https://raw.githubusercontent.com/alessiamanna/text_mining/main/immagine.png"
    user = "https://raw.githubusercontent.com/alessiamanna/text_mining/main/user.png"
    doc_logo = "https://raw.githubusercontent.com/alessiamanna/text_mining/main/doc.png"

   
    #url da sostituire con collegamento a github per la documentazione
    url = 'https://github.com/alessiamanna/text_mining/blob/main/ParlaMente_Documentazione_foobar.pdf'

    #container per gestire più agevolmente i tasti per la documentazione e per lo switch dei temi
    container_2 = st.container()
    with container_2:
        buttonDoc = f'''
        <a href="{url}">
            <button class = "customBtn">
                <img src="{doc_logo}" alt = "logo" style="width:50px; position: relative;">
            </button>
        </a>
        '''
        st.markdown(buttonDoc, unsafe_allow_html=True)
        # Add a button to switch themes
        if st.button('Switch Theme'):
            if st.session_state.theme == coffee_theme_css:
                st.session_state.theme = custom_theme_css
            else:
                st.session_state.theme = coffee_theme_css
            # Reload CSS after theme change
            load_css(st.session_state.theme)

    #titolo della pagina
    st.markdown(f"""<h1 style='text-align:center;' className='stTitle'> ParlaMente <img src="{bot}" alt="logo" style="width:65px; position: relative; bottom: 5px;">
    </h1>
    """, unsafe_allow_html=True)

    #sottotitolo della pagina
    st.markdown("""
    <div style='text-align: center; margin-bottom: 30px;'>
        <p class="customHdr" style='font-family: "Roboto", sans-serif; font-size: 1.2em;'>
           Il Tuo Deputato Digitale, Sempre a Disposizione!
        </p>
    </div>
    """, unsafe_allow_html=True)
    

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
        display_message(message["role"], message["content"], user if message["role"] == "user" else bot)
            
    # Ask a question
    if question := st.chat_input("Fammi una domanda"):
        # Append user question to history
        st.session_state.history.append({"role": "user", "content": question})
        # Add user question
        #with st.chat_message("user", avatar = user):
            #st.markdown(question)  

        display_message("user", question, user) #funzione per mostrare i messaggi
        # Answer the question
        answer, doc_source = OpenAI_utility.generate_answer(question)
        #risposta intera
        #with st.chat_message("assistant"):
            #st.write(answer)

        # Display the answer one word at a time
        words = answer.split()
        message_placeholder = st.empty()
        partial_answer = ""
        for i in range(len(words)):
            partial_answer += words[i] + " "
            message_placeholder.markdown(f""" 
                <div class="assistant-message">
                    <img src="{bot}" class="avatar">
                    <div class="chat-bubble assistant-bubble">
                        {partial_answer.strip()}
                    </div>
                </div>
            """, unsafe_allow_html=True)
            time.sleep(0.06)  # Adjust the sleep time to control the speed of word display
        # Append assistant answer to history
        st.session_state.history.append({"role": "assistant", "content": answer})

        # Append the document sources
        st.session_state.source.append({"question": question, "answer": answer, "document": doc_source})

    #aggiungi sidebar con informazioni sulla sessione e sullo storico delle chat
    with st.sidebar:
        container = st.container()
        with container:
            st.write("<style> .lower-text { margin-top: 10px; } </style>", unsafe_allow_html=True)
            st.write('<div class="lower-text">Storico delle chat e info sulla sessione</div>', unsafe_allow_html=True)
            st.write(st.session_state.get('source', 'No data available'))


def display_message(role, content, avatar_url):
    message_class = "user-message" if role == "user" else "assistant-message"
    bubble_class = "user-bubble" if role == "user" else "assistant-bubble"
    if role == "user":
        st.markdown(f"""         
            <div class="{message_class}">
                <div class="chat-bubble {bubble_class}">
                    {content}
                </div>
                <img src="{avatar_url}" class="avatar">
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="{message_class}">
                <img src="{avatar_url}" class="avatar">
                <div class="stChatMessage {bubble_class}">
                    {content}
                </div>
            </div>
        """, unsafe_allow_html=True)


def display_document_embedding_page():

    #se il vector_store nominato first_CameraDep esiste gia, non lo ricreo, se invece non esiste allora lo creo con lo stesso nome
    if not os.path.exists("chatbotParlamente/vector store/first_CameraDep"):

        print("assente")

        #richiamo read_pdf
        combined_content = OpenAI_utility.read_pdf("chatbotParlamente/merged_RegistroCmeraDeputati.pdf")
        #splitto il documento con chunk-size=520 e overlapping=80
        split = OpenAI_utility.split_doc(combined_content, 520, 80)
        #creo il vector_store
        OpenAI_utility.embedding_storing(split, True, "first_CameraDep", "first_CameraDep")


if __name__ == "__main__":
    main()
