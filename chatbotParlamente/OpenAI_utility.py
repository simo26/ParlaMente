import os
import openai
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
#Streamlit è una libreria Python per creare rapidamente applicazioni web interattive e data-driven. 
import streamlit as st
#PdfReader è una classe della libreria  pypdf che consente di leggere e manipolare file PDF
from pypdf import PdfReader
from pypdf import PdfMerger
#RecursiveCharacterTextSplitter è uno strumento per suddividere testi lunghi in segmenti più piccoli in modo ricorsivo, mantenendo la coerenza delle frasi. È utile per pre-processare testi per analisi successive.
from langchain.text_splitter import RecursiveCharacterTextSplitter
#FAISS (Facebook AI Similarity Search) è una libreria per la ricerca efficiente e il clustering di grandi collezioni di vettori. Viene utilizzata per eseguire ricerche di similarità su grandi dataset di embedding.
from langchain_community.vectorstores import FAISS
#ConversationalRetrievalChain è una catena di elaborazione che combina retrieval di informazioni e generazione di risposte in un contesto conversazionale, utile per chatbot e assistenti virtuali.
from langchain.chains import ConversationalRetrievalChain
#ConversationBufferWindowMemory è una struttura di memoria che memorizza le ultime N interazioni di una conversazione. È utile per mantenere il contesto durante una conversazione lunga, migliorando la coerenza delle risposte.
from langchain.memory import ConversationBufferWindowMemory

openai.api_key = "sk-proj-CwBtRD9D21FWFK8tFJf6T3BlbkFJn510lJk6oJlyxez3tnp1"

#Questa funzione merge_pdfs prende come input il percorso della cartella che contiene i PDF (folder_path) 
#e il percorso dove salvare il PDF unito (output_path). La funzione utilizza os.walk per attraversare 
#tutti i file nella cartella e unisce solo quelli che terminano con .pdf (ignorando maiuscole/minuscole).
def merge_pdfs(folder_path, output_path):
    merger = PdfMerger()

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                file_path = os.path.join(root, file)
                merger.append(file_path)
    
    merger.write(output_path)
    merger.close()


#funzione che consente di leggere i pdf e ritornarne il contenuto testuale tramite la libreria PdfReader; 
#sostanzialmente legge le pagine del pdf attraverso un ciclo: per ogni pagina all'interno 
#dell'oggetto reader, generato istanziando la funzione PdfReader a cui viene passato il file ovvero il suo
#specifico path, permette di estrarre il testo e di ritornare il documento concatenando (+=) il testo 
#di tutto il pdf. Quindi il risultato finale di questa funzione è che ritorna il testo del pdf
#concatenando tutti le stringhe appartenenti ad ogni pagina del pdf; 
#Inoltre sostituisco il next-line + "-" con #, in modo da realizzare sono uno splitting in chunks più 
#consistente del documento;
def read_pdf(file):
    document = ""

    reader = PdfReader(file)
    for page in reader.pages:
        document += page.extract_text()
        document = document.replace("\n-", "\n#")

    return document



#Splittiamo i documenti di grande dimensioni in chunks, poichè il modello non può gestire input più grandi della sua context Length che nella maggior parte dei casi è 2048 token!
#Ovviamente vogliamo realizzare uno split del documento che favorisca il recupero corretto delle informazioni.
#Generiamo chunks fino a 200 token(o caratteri) attraverso separatori come double new line o space applicati sul documento;
#Una volta individuato un chunk, realizziamo un secondo chunk non partendo dalla fine di quest'ultimo, ma tornando indietro di tot
#caratteri al fine non rischiare di perdere informazioni con lo split; cioè da un lato può portare a rindondanza, ma assicuro maggiormente di preservare il significato semanatico nei chunk prodotti dal documento.

#funzione per realizzare il recursive character text splitter; 
#In tale funzione vengono passati i parametri settati a priori chunk size e chunk overlapping ottimali per i testi forniti; 
#viene istanziato l'oggetto "splitter" a partire dalla classe RecursiveCharacterTextSplitter in cui vengono passati i parametri precedenti;
#su splitter viene richiamata la funzione split_text con cui viene eseguito uno split del documento ritornato da read_text sulla base del conteggio dei caratteri; viene dunque ritornato il documento splittato;
#su splitter viene richiamata la funzione create_documents che crea documenti, a partire dal testo splittato creato prima, in un formato per il processing del documento;
#al termine di tutto, la funzione dunque ritorna i chunk del documento originale;
def split_doc(document, chunk_size, chunk_overlap):

    splitter = RecursiveCharacterTextSplitter( ##aggiungi parte Split dell'nb!   
        separators="#", 
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    split = splitter.split_text(document)
    split = splitter.create_documents(split)

    return split


#codice per vedere i chunk generati
#result = split_doc(read_pdf("merged_RegistroCmeraDeputati.pdf"), 330, 50)
#print(result[0])
#print(result[1])
#print(result[2])



#una volta creati i chunks, realizziamo l'embedding del documento e il salvataggio in un vector store per un futuro retrieval;
#tale funzione prendere in ingresso i chunks generati con split_doc e dei parametri di controllo nel caso in cui si sta creando un nuovo vector store, con un nuovo nome, o se se ne sta ulitizzando creato in precedenza;
def embedding_storing( split, create_new_vs, existing_vector_store, new_vs_name):


        #Viene usato un pretrained sentence transformer model per generare l'embedding;
        #In tal caso si sfruttano i modelli di Embedding offerti da OpenAI (senza specificare il modello di default si utilizza text-embedding-ada-002)
        
        instructor_embeddings = OpenAIEmbeddings()

        #Viene generato un vs tramite la classe FAISS per realizzare la similarity search,
        #in particolare, applicando il metodo from_documents che a partire dai chunks dal modello di embedding precedente
        #ho dunque generato il database;
        #new db = FAISS.from_documents(split, instructor_embeddings)
        db = FAISS.from_documents(split, instructor_embeddings)

        #Spiega perchè FAISS invece di Chroma!!!

        #Se è stato creato un nuovo vs, lo salvo in una CACHE definita a partire dal nome del nuovo vs;
        #ciò mi permetterà successivamente, dopo aver istanziato almeno la prima volta il vs, di poterlo richiamare subito
        #senza dover generare di nuovo gli embedding ed inserirli nel db;
        #quindi, nel caso il vs esiste già, viene recuperato dalla cache, si ha il merge dei due db e risalvato in cache;
        if create_new_vs == True:
            # Save db
            db.save_local("vector store/" + new_vs_name)
        else:
            # Load existing db
            load_db = FAISS.load_local(
                "vector store/" + existing_vector_store,
                instructor_embeddings,
                allow_dangerous_deserialization=True
            )
            # Merge two DBs and save
            load_db.merge_from(db)
            load_db.save_local("vector store/" + new_vs_name)
 
        
        
#funzione per inizializzare e preparare il RAG conversational model con una data configurazione
def prepare_rag_llm(vector_store_list, temperature, max_length):


    #istanzio nuovamente il modello di embedding precedente;
    instructor_embeddings = OpenAIEmbeddings()

    # carico il mio db attraverso load_local della classe FAISS passando il file path del vector_store su cui lavorerà il RAG,
    #il modello di embedding, ed una serie di parametri (vedi a cosa serve allow_dangerous_deserialization)
    loaded_db = FAISS.load_local(
        f"vector store/{vector_store_list}", instructor_embeddings, allow_dangerous_deserialization=True
    )

    #istanzio l'LLM richiamando il modello di GPT-3.5-turbo attraverso la classe per l'integrazione 
    #di modelli OpenaAI, settando la temperature e la massima lungezza della risposta che viene data in output; 
    #L'API_KEY viene ottenuta direttamente a partire dalle variabili d'ambiente;
    #NOTA: TEMPERATURA = parametro di entropia che va da 0 a 1 e che determina quanto la risposta del chatbot sarà creativa (andando verso l'1), ovvero una risposta più o meno deterministica.
    llm = ChatOpenAI(
        model="gpt-3.5-turbo", #Il modello GPT-3.5-turbo può gestire fino a 4096 token per richiesta. Questo limite include sia i token di input che quelli di output generati dal modello. Ad esempio, se invii 2000 token come input, il modello può generare fino a 2096 token in output.
        temperature= temperature, 
        max_tokens = max_length
    )

    #setto ConversationBuffer che consente al chat bot di recuperare informazioni a partire dalla storia
    #della chat
    memory = ConversationBufferWindowMemory(
        k=2, #recupera info a partire dalle ultime 2 interazioni;
        memory_key="chat_history", #tali info verrare usate in streamlit
        output_key="answer", #tali info verrare usate in streamlit
        return_messages=True,
    )

    # Create the chatbot:
    #setto la question-answer chain attraverso la chain ConversationalRetrievalChain
    #fornita dal LangChain, ovvero un metodo che recupera l'LLM definito prima, il db
    #settato in modalità as_retriever, ovvero pronto a cercare i documenti necessari,
    #e settando il parametro di memoria precedente;
    qa_conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff", #refine #il parametro chain_type="stuff" indica che i tre documenti più attinenti recuperati verranno combinati insieme e passati al modello di linguaggio in una singola richiesta per generare una risposta alla query
        retriever=loaded_db.as_retriever(search_kwargs={"k": 3}), #Sfrutto i top 3 documenti più attinenti alla query
        return_source_documents=True, #serve a indicare se i documenti sorgente utilizzati per generare la risposta devono essere restituiti insieme alla risposta stessa.
        memory=memory,
    )

    #alla fine viene ritornata la chain finale di question_answer
    return qa_conversation

#funzione per generare risposte alle query dell'utente usando un il conversational model realizzato prima:
#prima c'era token
def generate_answer(question):
    answer = "Si è verificato un errore" #Risposta di Default in caso di errore
    response = st.session_state.conversation({"question": question}) #Viene processata la domanda attraverso il conversational model
    answer = response.get("answer").split("Helpful Answer:")[-1].strip() #Viene estratta la risposta da response
    explanation = response.get("source_documents", [])
    doc_source = [d.page_content for d in explanation] #Vengono collezionati i documenti che hanno contribuito alla risposta;

    return answer, doc_source
    
