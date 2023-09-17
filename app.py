import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.vectorstores import Pinecone
import tempfile
import tiktoken

# Initialize environment variables for API keys
os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']
pinecone.init(api_key=st.secrets['PINECONE_API_KEY'], environment=st.secrets['PINECONE_ENVIRONMENT'])

# Initialize Pinecone index
index = pinecone.Index('test-01')

# Document Processing Functions

# Load Document
def load_document(file):
    if not os.path.exists("docs"):
        os.makedirs("docs")

    file_path = os.path.join("docs", file.name)
    with open(file_path, 'wb') as f:
        f.write(file.read())

    name, extension = os.path.splitext(file.name)
    if extension == '.pdf':
        st.write(f'Loading {file.name}...')
        loader = PyPDFLoader(file_path)
    elif extension == '.docx':
        st.write(f'Loading {file.name}...')
        loader = Docx2txtLoader(file_path)
    else:
        st.write('Document format not supported!')
        return None

    data = loader.load()
    return data

# Split Text
def split_text(data, chunk_size=3530):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    chunks = text_splitter.split_documents(data)
    return chunks

# Calculate Embedding Cost
def embedding_cost(texts):
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    st.write(f'Total Tokens: {total_tokens}')
    st.write(f'Embedding Cost in USD: {total_tokens / 1000 * 0.0001:.5f}')

# Cache Pinecone Index Deletion
@st.cache_data()
def delete_indexes(index_name=index):
    if index_name == index:
        indexes = pinecone.list_indexes()
        st.write('Deleting all indexes ... ')
        for index in indexes:
            pinecone.delete_index(index)
        st.write('Done!')
    else:
        st.write(f'Deleting the index: {index_name} ...', end='')
        pinecone.delete_index(index_name)
        st.write('Done')

# Create Vectors
@st.cache(allow_output_mutation=True)
def create_vectors(index_name, chunks):
    embeddings = OpenAIEmbeddings()

    if index_name in pinecone.list_indexes():
        st.write(f'The index {index_name} already exists. Loading embeddings ... ')
        vectors = Pinecone.from_existing_index(index_name, embeddings)
    else:
        st.write(f'Creating the index {index_name} and embeddings ...')
        pinecone.create_index(index_name, dimension=1536, metric='cosine')
        vectors = Pinecone.from_documents(chunks, embeddings, index_name=index_name)

    st.write('Ok')
    return vectors

# Memory-based Query
def query_with_memory(vectors, question, memory=[]):
    from langchain.chains import ConversationalRetrievalChain
    from langchain.chat_models import ChatOpenAI
    llm = ChatOpenAI(temperature=0, model_name="gpt-4")
    retriever = vectors.as_retriever(search_type='similarity', search_kwargs={'k': 5})
    crc = ConversationalRetrievalChain.from_llm(llm, retriever)
    answer = crc({'question': question, 'chat_history': memory})
    memory.append((question, answer['answer']))

    return answer, memory

# Main Function
def main():
    st.title("Question and Answer Chatbot")
    file = st.file_uploader("Select a PDF or DOCX file", type=["pdf", "docx"])
    if file:
        content = load_document(file)
        if content:
            chunks = split_text(content)
            st.write(f'The number of chunks is: {len(chunks)} chunks')

            # Calculate embedding cost
            embedding_cost(chunks)

            # Create index and vectors
            delete_indexes("alter")
            index_name = 'alter'
            vectors = create_vectors(index_name, chunks)

            # Queries
            questions(vectors)

def questions(vectors):
    st.subheader("Question Query")
    question = st.text_input("Write your question:")
    if question:
        answer, memory = query_with_memory(vectors, question, memory=[])
        st.write("Answer:", answer['answer'])

if __name__ == "__main__":
    main()
