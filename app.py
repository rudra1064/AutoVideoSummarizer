import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file, get_file
import google.generativeai as genai
import time
from pathlib import Path
import tempfile
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("❌ Google API Key not found! Please check your `.env` file.")
    st.stop()

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# Custom CSS for background and styling
st.markdown("""
    <style>
        body {
            background: linear-gradient(120deg, #1e3c72, #2a5298);
            color: white;
        }
        .main {
            background-color: rgba(0, 0, 0, 0.3);
            padding: 20px;
            border-radius: 10px;
        }
        h1, h2, h3, h4 {
            color: #f9f9f9;
        }
        footer {
            visibility: hidden;
        }
        .footer-custom {
            text-align: center;
            padding: 15px;
            font-size: 14px;
            color: #ccc;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 style='text-align: center;'>🎥 Video Summary App</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Powered by Google Gemini 2.0 Flash 🚀</h3>", unsafe_allow_html=True)

# Introduction
with st.expander("ℹ️ About This App", expanded=True):
    st.write("""
        Welcome to the **Video Summary App**!  
        This tool uses Google's **Gemini 2.0 Flash** AI model to analyze and summarize videos.  
        Simply upload your video, ask a question, and let AI do the heavy lifting — including optional web research for richer context.
    """)

# Initialize AI Agent
@st.cache_resource
def initialize_agent():
    return Agent(
        name="video-summary-agent",
        model=Gemini(id="gemini-2.0-flash"),
        tools=[DuckDuckGo()],
        markdown=True
    )

multimodal_agent = initialize_agent()

# Upload Section
st.markdown("### 📤 Upload a Video")
video_file = st.file_uploader(
    "Choose a video file to summarize",
    type=["mp4", "avi", "mov"],
    help="Supported formats: mp4, avi, mov"
)

if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name

    st.video(video_path, format="video/mp4", start_time=0)

    # Query box
    user_query = st.text_area(
        "💡 What insights do you want?",
        placeholder="E.g., 'Summarize this video', 'Find key events', or 'Explain in detail'.",
        help="Be specific for better results."
    )

    # Analyze button
    if st.button("🔍 Analyze Video", key="analyze_video_button"):
        if not user_query:
            st.warning("Please enter a question or request for the video.")
        else:
            try:
                with st.spinner("⏳ Processing video and gathering insights..."):
                    processed_video = upload_file(video_path)
                    while processed_video.state.name == "PROCESSING":
                        time.sleep(1)
                        processed_video = get_file(processed_video.name)

                    analysis_prompt = f"""
                    Analyze the uploaded video for content and context.
                    Respond to the following query using video insights and supplementary web research:
                    {user_query}

                    Provide a detailed, user-friendly, and actionable response.
                    """

                    response = multimodal_agent.run(analysis_prompt, videos=[processed_video])

                st.subheader("📊 Analysis Result")
                st.markdown(response.content)

            except Exception as error:
                st.error(f"An error occurred: {error}")
            finally:
                Path(video_path).unlink(missing_ok=True)
else:
    st.info("⬆️ Please upload a video to start.")

# Footer
st.markdown("<div class='footer-custom'>Made with ❤️ using Streamlit & Google Gemini | © 2025</div>", unsafe_allow_html=True)
