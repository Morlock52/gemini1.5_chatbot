Chatbot without RAG!!!
Create a streamlit app that lets you chat with any type of media file (documents, images, videos, audio).
All this without doing any RAG!
That's because we are using Gemini 1.5 family of models, which give us a context window of 1MM tokens.  That's 1 hour of video, over 700K words, 11 hours of audio, etc.
Enjoy!  Please "Star" this repo if you find my code helpful and useful, much appreciated!

## Setup

1. Install the Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Configure the following environment variables:

   - `GOOGLE_API_KEY_NEW` – API key for Gemini models.
   - `GOOG_PROJECT` – Google Cloud project ID (required for Vertex AI usage).
   - `OPENAI_API_KEY` – OpenAI API key (optional).
   - `MEDIA_PATH` – directory where temporary media files are stored (defaults to the current directory).

## Running the app

Launch the Streamlit interface with:

```bash
streamlit run multi_service_chatbot.py
```

You can deploy the same command on Streamlit Cloud or any other platform that supports Streamlit.
