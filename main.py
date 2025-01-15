import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

# Set up Streamlit page configuration
st.set_page_config(page_title="YouTube Chatbot", layout="wide")

# Title of the app
st.title("Chat with YouTube Videos")

# Step 1: Securely input Google API key using Streamlit Secrets
st.sidebar.header("API Key Configuration")
google_api_key = st.sidebar.text_input("Enter your Google API Key", type="password")
if google_api_key:
    st.session_state["google_api_key"] = google_api_key

# Step 2: Input YouTube video link
video_url = st.text_input("Enter the YouTube video URL:")

# Helper function to extract video ID from URL
def get_video_id(url):
    parsed_url = urlparse(url)
    if parsed_url.hostname == "youtu.be":
        return parsed_url.path[1:]
    if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
        if parsed_url.path == "/watch":
            return parse_qs(parsed_url.query).get("v", [None])[0]
    return None

# Step 3: Load and process video transcript
if video_url:
    video_id = get_video_id(video_url)
    if video_id:
        try:
            st.write(f"Fetching transcript for video ID: {video_id}...")
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            full_text = " ".join([entry["text"] for entry in transcript])
            
            # Split text into chunks for better processing
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            documents = text_splitter.split_text(full_text)

            # Display first few lines of the transcript
            st.write("Transcript loaded successfully!")
            st.write("First 500 characters of the transcript:")
            st.write(full_text[:500])

            # Step 4: Set up LangChain components for chat functionality
            embeddings = OpenAIEmbeddings(openai_api_key=st.session_state["google_api_key"])
            vector_store = Chroma.from_documents(documents, embeddings)
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            retriever_chain = ConversationalRetrievalChain.from_llm(
                llm=ChatOpenAI(temperature=0, openai_api_key=st.session_state["google_api_key"]),
                retriever=vector_store.as_retriever(),
                memory=memory,
            )

            # Step 5: Chat Interface
            st.subheader("Chat with the Video")
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Display chat history on app rerun
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Accept user input and generate responses
            user_input = st.chat_input("Ask something about the video...")
            if user_input:
                with st.chat_message("user"):
                    st.markdown(user_input)
                st.session_state.messages.append({"role": "user", "content": user_input})

                # Generate response from LangChain retriever chain
                response = retriever_chain.run(user_input)
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            st.error(f"Error fetching transcript: {str(e)}")
    else:
        st.error("Invalid YouTube URL. Please enter a valid link.")
else:
    st.info("Please enter a YouTube URL to begin.")
