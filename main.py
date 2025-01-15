import streamlit as st
from langchain.schema import Document
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma

# Page config
st.set_page_config(
    page_title="YouTube Video Chat",
    page_icon="🎥",
    layout="wide"
)

# Initialize session states
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'ready' not in st.session_state:
    st.session_state.ready = False

# Title
st.title("🎥 YouTube Video Chatbot")

def get_video_id(url):
    """Extract video ID from YouTube URL"""
    parsed_url = urlparse(url)
    if parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]
    if parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
        if parsed_url.path == '/watch':
            return parse_qs(parsed_url.query)['v'][0]
    return None

def load_video_transcript(video_url):
    """Load and process YouTube video transcript"""
    try:
        video_id = get_video_id(video_url)
        if not video_id:
            st.error("Could not extract video ID from URL")
            return None
            
        # Get transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Combine transcript text
        full_text = ' '.join([entry['text'] for entry in transcript])
        
        # Create document
        return Document(
            page_content=full_text,
            metadata={"source": video_url}
        )
        
    except Exception as e:
        st.error(f"Error loading transcript: {str(e)}")
        return None

def initialize_chat_engine(document):
    """Initialize the chat engine with the document"""
    try:
        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents([document])

        # Create embeddings and vectorstore
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = Chroma.from_documents(splits, embeddings)

        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-002",
            temperature=0.7,
            top_k=3,
            top_p=0.8,
            max_output_tokens=2048
        )

        # Create memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Create chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            return_source_documents=True
        )
        
        return chain

    except Exception as e:
        st.error(f"Error initializing chat engine: {str(e)}")
        return None

# Sidebar
with st.sidebar:
    st.header("📝 Video Input")
    video_url = st.text_input(
        "Enter YouTube Video URL:",
        placeholder="https://www.youtube.com/watch?v=..."
    )
    
    if video_url:
        with st.spinner("Loading video transcript..."):
            doc = load_video_transcript(video_url)
            if doc:
                with st.spinner("Initializing chat engine..."):
                    qa_chain = initialize_chat_engine(doc)
                    if qa_chain:
                        st.session_state.qa_chain = qa_chain
                        st.session_state.ready = True
                        st.success("✅ Ready to chat!")
                    else:
                        st.error("Failed to initialize chat engine")

# Main chat interface
if st.session_state.ready:
    # Chat interface
    st.header("💬 Chat with Video Content")
    
    # Display messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the video..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.qa_chain({"question": prompt})
                    st.markdown(response['answer'])
                    st.session_state.messages.append({"role": "assistant", "content": response['answer']})
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
else:
    st.info("👆 Enter a YouTube URL in the sidebar to start chatting!")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and LangChain")
