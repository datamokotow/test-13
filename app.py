import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import tempfile
import tiktoken

os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']
pinecone.init(api_key=st.secrets['PINECONE_API_KEY'], environment=st.secrets['PINECONE_ENVIRONMENT'])
pinecone.init(      
	=st.secrets['PINECONE_API_KEY'],      
	environment=st.secrets['PINECONE_ENVIRONMENT']      
)      
index = pinecone.Index('test-01')


# Funciones de procesamiento de documentos
def cargar_documento(archivo):
    if not os.path.exists("docs"):
        os.makedirs("docs")

    ruta_archivo = os.path.join("docs", archivo.name)
    with open(ruta_archivo, 'wb') as f:
        f.write(archivo.read())

    nombre, extension = os.path.splitext(archivo.name)
    if extension == '.pdf':
        st.write(f'Cargando {archivo.name}...')
        loader = PyPDFLoader(ruta_archivo)
    elif extension == '.docx':
        st.write(f'Cargando {archivo.name}...')
        loader = Docx2txtLoader(ruta_archivo)
    else:
        st.write('El formato del documento no está soportado!')
        return None

    data = loader.load()
    return data

def fragmentar(data, chunk_size=3530):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    fragmentos = text_splitter.split_documents(data)
    return fragmentos

def costo_embedding(texts):
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    st.write(f'Total Tokens: {total_tokens}')
    st.write(f'Embedding Cost in USD: {total_tokens / 1000 * 0.0001:.5f}')

#@st.cache(allow_output_mutation=True)
@st.cache_data()
def borrar_indices(index_name='alter'):
    if index_name == 'alter':
        indexes = pinecone.list_indexes()
        st.write('Borrando todos los índices ... ')
        for index in indexes:
            pinecone.delete_index(index)
        st.write('Listo!')
    else:
        st.write(f'Borrando el índice: {index_name} ...', end='')
        pinecone.delete_index(index_name)
        st.write('Listo')

@st.cache(allow_output_mutation=True)
def creando_vectores(index_name, fragmentos):
    embeddings = OpenAIEmbeddings()

    if index_name in pinecone.list_indexes():
        st.write(f'El índice {index_name} ya existe. Cargando los embeddings ... ')
        vectores = Pinecone.from_existing_index(index_name, embeddings)
    else:
        st.write(f'Creando el índice {index_name} y los embeddings ...')
        pinecone.create_index(index_name, dimension=1536, metric='cosine')
        vectores = Pinecone.from_documents(fragmentos, embeddings, index_name=index_name)

    st.write('Ok')
    return vectores
def consulta_con_memoria(vectores, pregunta, memoria=[]):
    from langchain.chains import ConversationalRetrievalChain
    from langchain.chat_models import ChatOpenAI
    llm = ChatOpenAI(temperature=0, model_name="gpt-4")
    retriever = vectores.as_retriever(search_type='similarity', search_kwargs={'k': 5})
    crc = ConversationalRetrievalChain.from_llm(llm, retriever)
    respuesta = crc({'question': pregunta, 'chat_history': memoria})
    memoria.append((pregunta, respuesta['answer']))

    return respuesta, memoria


def main():
    # Montar Google Drive
    # Cargar archivo desde el input
    st.title("Chatbot de Preguntas y Respuestas")
    #CSS al input 

    archivo = st.file_uploader("Selecciona un archivo PDF o DOCX", type=["pdf", "docx"])
    if archivo:
        contenido = cargar_documento(archivo)
        if contenido:
            fragmentos = fragmentar(contenido)
            st.write(f"El Número de fragmentos es de: {len(fragmentos)} fragmentos")

            # Calcular costo de embedding
            costo_embedding(fragmentos)

            # Crear índice y vectores
            borrar_indices("alter")
            index_name = 'alter'
            vectores = creando_vectores(index_name, fragmentos)

            # Consultas
            preguntas(vectores)
            # pregunta = st.text_input("Escribe tu pregunta:")
            # if pregunta:
            #     respuesta, memoria = consulta_con_memoria(vectores, pregunta, memoria=[])
            #     st.write("Respuesta:", respuesta['answer'])

def preguntas(vectores):
    st.subheader("Consulta de Preguntas")
    pregunta = st.text_input("Escribe tu pregunta:")
    if pregunta:
        respuesta, memoria = consulta_con_memoria(vectores, pregunta, memoria=[])
        st.write("Respuesta:", respuesta['answer'])

if __name__ == "__main__":
    main()
