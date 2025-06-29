import os
import tempfile
from dotenv import load_dotenv
from gtts import gTTS
from faster_whisper import WhisperModel
from typing import List, Tuple, Dict

# LangChain and Pinecone imports
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

import gradio as gr

# Load environment variables
load_dotenv()

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

if not all([OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT]):
    print("Error: Missing environment variables. Please check your .env file.")
    exit(1)

index_name = "financial-literacy-chatbot"

# Initialize Embeddings
embeddings = None
try:
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    print("OpenAI Embeddings initialized.")
except Exception as e:
    print(f"Error initializing embeddings: {e}")

# Initialize Pinecone Vectorstore
vectorstore = None
if embeddings:
    try:
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )
        print(f"Pinecone Vectorstore initialized with index: {index_name}")
    except Exception as e:
        print(f"Error initializing Pinecone Vectorstore: {e}")
else:
    print("Skipping Pinecone Vectorstore initialization due to embedding error.")

# Initialize LLM
llm = None
try:
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.7)
    print("ChatOpenAI LLM initialized.")
except Exception as e:
    print(f"Error initializing ChatOpenAI LLM: {e}")

# Initialize RetrievalQA
qa_chain = None
if vectorstore and llm:
    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )
        print("RetrievalQA chain initialized.")
    except Exception as e:
        print(f"Error initializing RetrievalQA chain: {e}")
else:
    print("Skipping RetrievalQA chain initialization.")

# Initialize Whisper Model
whisper_model = None
try:
    whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    print("Faster Whisper model loaded.")
except Exception as e:
    print(f"Error loading Whisper model: {e}")

INIT_SUCCESS = bool(vectorstore and llm and whisper_model and qa_chain)
if not INIT_SUCCESS:
    print("\n--- WARNING: Chatbot components not fully initialized. ---")

# Audio processing functions
# def transcribe_audio(audio_path):
#     if not audio_path:
#         return ""
#     if not whisper_model:
#         return "Error: Whisper model not loaded."
#     try:
#         segments, info = whisper_model.transcribe(audio_path, beam_size=5)
#         text = " ".join([seg.text for seg in segments])
#         print(f"Transcribed: '{text}'")
#         return text
#     except Exception as e:
#         print(f"Transcription error: {e}")
#         return f"Error transcribing audio: {e}"
    

import numpy as np
import soundfile as sf
import tempfile

def transcribe_audio(audio_data):
    if audio_data is None:
        return ""
    if not whisper_model:
        return "Error: Whisper model not loaded. Cannot transcribe."
    try:
        # audio_data is a tuple (numpy_array, sample_rate) from gr.Microphone
        audio_array, sample_rate = audio_data
        # Save numpy array to a temp WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            sf.write(tmp_wav.name, audio_array, sample_rate)
            tmp_wav_path = tmp_wav.name

        segments, info = whisper_model.transcribe(tmp_wav_path, beam_size=5)
        transcribed_text = " ".join([segment.text for segment in segments])
        print(f"Transcribed: '{transcribed_text}' (Language: {info.language}, Probability: {info.language_probability:.2f})")
        return transcribed_text
    except Exception as e:
        print(f"Error during audio transcription: {e}")
        return f"Error transcribing audio: {e}"


def speak(text):
    if not text.strip():
        return None
    try:
        tts = gTTS(text=text, lang='en')
        temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_audio_file.name)
        temp_audio_file.close()
        print(f"TTS generated: {temp_audio_file.name}")
        return temp_audio_file.name
    except Exception as e:
        print(f"TTS error: {e}")
        return None

def chatbot_response(message: str | None, audio_input_path: str | None, history: List[List[str | None]]) -> Tuple[str, List[List[str | None]], str | None]:
    if not INIT_SUCCESS:
        err = "Chatbot not initialized properly."
        history.append([(message or "Audio input"), err])
        return "", history, speak(err)
    user_text = ""
    if audio_input_path:
        transcribed = transcribe_audio(audio_input_path)
        if transcribed.startswith("Error"):
            history.append([f"Audio Error: {transcribed}", "Could not process audio. Try typing your question."])
            return "", history, None
        user_text = transcribed
    elif message:
        user_text = message.strip()
    if not user_text:
        return "", history, None
    try:
        response_text = qa_chain.run(user_text) if qa_chain else "Knowledge base not available."
        audio_path = speak(response_text)
        history.append([user_text, response_text])
        return "", history, audio_path
    except Exception as e:
        err_msg = f"Unexpected error: {e}"
        history.append([user_text, err_msg])
        return "", history, speak(err_msg)

# Sample questions data
SAMPLE_QUESTIONS = {
    "Budgeting": [
        "What is the 5/30/20 rule for budgeting?",
        "Can you recommend a simple budgeting template?",
        "How do I track my expenses effectively?",
        "What are common mistakes to avoid when budgeting?"
    ],
    "Savings": [
        "What is compound interest and how does it affect savings?",
        "What are some strategies to save money quickly?",
        "How much should I save for an emergency fund?",
        "What's the difference between saving and investing?"
    ],
    "Credit Score": [
        "What is a credit score and why is it important?",
        "How can I improve my credit score?",
        "What factors affect my credit score?",
        "How often should I check my credit report?"
    ],
    "Investing": [
        "What are stocks and bonds?",
        "What is diversification in investing?",
        "How do I start investing as a beginner?",
        "Explain the concept of risk tolerance in investments."
    ],
    "Retirement Planning": [
        "What is a 401(k) and how does it work?",
        "What's the difference between a traditional IRA and a Roth IRA?",
        "How much should I save for retirement?",
        "When can I start withdrawing from my retirement accounts without penalty?"
    ]
}

# --- CSS for styling ---
css = """
#user-input-row {
    display: flex;
    gap: 8px;
    margin-bottom: 10px;
}
#user-question-textbox textarea {
    flex-grow: 1;
    font-size: 1.1rem;
    padding: 10px 15px;
    border-radius: 10px;
    border: 2px solid #4a90e2;
    transition: border-color 0.3s ease;
}
#user-question-textbox textarea:focus {
    border-color: #0078d7;
    outline: none;
    box-shadow: 0 0 5px #0078d7;
}
#send-text-btn {
    background-color: #4a90e2;
    color: white;
    border-radius: 50%;
    width: 48px;
    height: 48px;
    border: none;
    cursor: pointer;
    transition: background-color 0.3s ease;
    display: flex;
    align-items: center;
    justify-content: center;
}
#send-text-btn:hover {
    background-color: #0078d7;
}
#mic-btn {
    background-color: #28a745;
    color: white;
    border-radius: 50%;
    width: 48px;
    height: 48px;
    border: none;
    cursor: pointer;
    transition: background-color 0.3s ease;
    display: flex;
    align-items: center;
    justify-content: center;
}
#mic-btn:hover {
    background-color: #1e7e34;
}
.gr-chatbot .chatbot-message.user {
    background-color: #e1f0ff;
}
.gr-chatbot .chatbot-message.bot {
    background-color: #f0f4f8;
}
"""

mic_svg_icon = """
<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="white" viewBox="0 0 24 24">
  <path d="M12 14a3 3 0 003-3V6a3 3 0 00-6 0v5a3 3 0 003 3zm5-3a5 5 0 01-10 0H5a7 7 0 0014 0h-2z"/>
  <path d="M19 11v2a7 7 0 01-14 0v-2H3v2a9 9 0 0018 0v-2h-2z"/>
</svg>
"""

# --- Gradio Interface ---
with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ Financial Literacy Chatbot 💰")
    gr.Markdown("Ask me anything about personal finance or select a category for sample questions.")
    gr.Markdown("---")

    # Category Buttons Row
    with gr.Row():
        budgeting_btn = gr.Button("Budgeting", scale=1, size="lg")
        savings_btn = gr.Button("Savings", scale=1, size="lg")
        credit_score_btn = gr.Button("Credit Score", scale=1, size="lg")
        investing_btn = gr.Button("Investing", scale=1, size="lg")
        retirement_btn = gr.Button("Retirement Planning", scale=1, size="lg")

    gr.Markdown("---")

    # Sample Questions Buttons + Hidden Texts
    with gr.Row() as sample_questions_row:
        sample_question_texts = [gr.Textbox(value="", visible=False) for _ in range(5)]
        sample_question_btns = [gr.Button("", visible=False) for _ in range(5)]

    gr.Markdown("---")

    # Chatbot display area
    chatbot_display = gr.Chatbot(
        height=400,
        render_markdown=True,
        bubble_full_width=False,
        value=[[None, "Hello! I'm your Financial Literacy Chatbot. Choose a category, speak, or type your question!"]]
    )

    # Combined user input: text box + mic + send button side by side
    with gr.Row(elem_id="user-input-row"):
        user_question_textbox = gr.Textbox(
            placeholder="Type your question here...",
            elem_id="user-question-textbox",
            show_label=False,
            interactive=True
        )
        audio_input_mic = gr.Microphone(label="Record your question")  # Hidden mic input for recording

        mic_btn = gr.Button(value=mic_svg_icon, elem_id="mic-btn")
        send_text_btn = gr.Button("➡️", elem_id="send-text-btn")

    audio_output_player = gr.Audio(label="Chatbot's Spoken Response", autoplay=True, type="filepath")

    # --- Functions for sample question button updates ---
    def update_sample_questions_display(category_name: str) -> list:
        questions = SAMPLE_QUESTIONS.get(category_name, [])
        btn_updates = []
        text_updates = []
        for i in range(5):
            if i < len(questions):
                btn_updates.append(gr.update(value=questions[i], visible=True))
                text_updates.append(gr.update(value=questions[i]))
            else:
                btn_updates.append(gr.update(visible=False))
                text_updates.append(gr.update(value=""))
        return btn_updates + text_updates

    def on_sample_question_click(question_text, chat_history):
        return chatbot_response(question_text, None, chat_history)

    # --- Event bindings ---

    # Category buttons update sample question buttons
    budgeting_btn.click(fn=lambda: update_sample_questions_display("Budgeting"), inputs=[], outputs=sample_question_btns + sample_question_texts)
    savings_btn.click(fn=lambda: update_sample_questions_display("Savings"), inputs=[], outputs=sample_question_btns + sample_question_texts)
    credit_score_btn.click(fn=lambda: update_sample_questions_display("Credit Score"), inputs=[], outputs=sample_question_btns + sample_question_texts)
    investing_btn.click(fn=lambda: update_sample_questions_display("Investing"), inputs=[], outputs=sample_question_btns + sample_question_texts)
    retirement_btn.click(fn=lambda: update_sample_questions_display("Retirement Planning"), inputs=[], outputs=sample_question_btns + sample_question_texts)

    # Sample question buttons click: send question text to chatbot
    for btn, txt in zip(sample_question_btns, sample_question_texts):
        btn.click(fn=on_sample_question_click, inputs=[txt, chatbot_display], outputs=[user_question_textbox, chatbot_display, audio_output_player])

    # Mic button click: toggle recording (show hidden audio_input_mic)
    def toggle_recording():
        # This is a simple toggle that triggers the hidden gr.Audio component.
        # Because Gradio doesn't support direct toggle via Button, we'll simulate by focusing on audio input.
        return gr.update(visible=True)

    mic_btn.click(fn=toggle_recording, inputs=[], outputs=[audio_input_mic])

    # When audio is recorded (audio_input_mic changes), transcribe and send to chatbot
    def handle_audio_transcription(audio_path, chat_history):
        return chatbot_response(None, audio_path, chat_history)

    audio_input_mic.change(fn=handle_audio_transcription, inputs=[audio_input_mic, chatbot_display], outputs=[user_question_textbox, chatbot_display, audio_output_player])

    # Send text question on send button click or textbox submit
    send_text_btn.click(fn=lambda msg, hist: chatbot_response(msg, None, hist), inputs=[user_question_textbox, chatbot_display], outputs=[user_question_textbox, chatbot_display, audio_output_player])
    user_question_textbox.submit(fn=lambda msg, hist: chatbot_response(msg, None, hist), inputs=[user_question_textbox, chatbot_display], outputs=[user_question_textbox, chatbot_display, audio_output_player])

if __name__ == "__main__":
    if not INIT_SUCCESS:
        print("\n# ERROR: Chatbot not fully initialized. Please check setup.")
    else:
        print("\nLaunching Gradio Interface...")
        demo.launch(share=True) # share=True generates a public link (expires after 72 hours)
