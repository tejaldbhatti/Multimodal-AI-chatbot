import os
import tempfile
import logging
import re
import asyncio
from typing import List, Tuple, Dict, Optional, Generator

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# --- Gradio import ---
import gradio as gr

# --- Speech-to-Text (Faster Whisper) and Text-to-Speech (gTTS) ---
from faster_whisper import WhisperModel
from gtts import gTTS

# --- Import from your backend logic ---
from chatbot_backend import chatbot_respond # Import the core function

# --- Configure logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration (from .env) - ONLY FOR CLIENT-SIDE DEPENDENCIES IF ANY ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Speech-to-Text (Faster Whisper) ---
try:
    whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    logging.info("Faster Whisper model loaded.")
except Exception as e:
    logging.error(f"Error loading Faster Whisper model: {e}. Speech-to-text functionality will be limited.")
    whisper_model = None

async def transcribe_audio(audio_path: str) -> str:
    if not whisper_model:
        return "Error: Speech-to-text model not loaded."
    try:
        segments, info = await asyncio.to_thread(whisper_model.transcribe, audio_path, beam_size=5)
        transcription = " ".join([segment.text for segment in segments])
        logging.info(f"Audio Transcribed: '{transcription}'")
        return transcription
    except Exception as e:
        logging.error(f"Error during audio transcription: {e}")
        return f"Error: Could not transcribe audio. {e}"

async def speak(text: str) -> Optional[str]:
    try:
        if not text:
            return None
        tts = gTTS(text=text, lang='en')
        audio_file_path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
        await asyncio.to_thread(tts.save, audio_file_path)
        logging.info(f"Text-to-speech audio saved to: {audio_file_path}")
        return audio_file_path
    except Exception as e:
        logging.error(f"Error during text-to-speech: {e}")
        return None

# --- Gradio Chatbot Response Function ---
async def chatbot_response(
    message: str | None,
    audio_input_path: str | None,
    history: List[List[str | None]],
    uploaded_salary_slip_content: Optional[str] = None
) -> Generator[Tuple[str, List[List[str | None]], Optional[str]], None, None]:

    if history is None:
        history = []

    user_input_text = ""
    if audio_input_path:
        logging.info(f"Processing audio input from: {audio_input_path}")
        transcribed = await transcribe_audio(audio_input_path)
        if transcribed.startswith("Error"):
            response_text = f"I could not process your audio: {transcribed}. Try typing your question or re-recording."
            history.append([(message or "Audio input error"), response_text])
            audio_output_path = await speak(response_text)
            yield "", history, audio_output_path
            return
        user_input_text = transcribed
    elif message:
        user_input_text = message.strip()

    display_user_input = user_input_text if user_input_text else "File Uploaded"

    current_chat_entry = [display_user_input, ""]
    history.append(current_chat_entry)
    yield "", history, None

    full_response_text = ""
    audio_output_path = None
    try:
        if uploaded_salary_slip_content:
            logging.info("Calling backend for salary slip analysis via chatbot_respond.")
            response_from_backend = await chatbot_respond(user_input=user_input_text, uploaded_salary_slip_content=uploaded_salary_slip_content)
            full_response_text = response_from_backend
            current_chat_entry[1] = full_response_text # Update the chat history entry with the full response

            logging.info(f"Chatbot Full Response for Salary Slip: {full_response_text}")

            # Generate audio for the full response and yield the final state in one go
            audio_output_path = await speak(full_response_text)
            yield "", history, audio_output_path # <-- Adjusted yield for salary slip path

        elif user_input_text:
            logging.info(f"Calling backend with user input: '{user_input_text}'")
            async for chunk in chatbot_respond(user_input=user_input_text, uploaded_salary_slip_content=None):
                full_response_text += chunk
                current_chat_entry[1] = full_response_text
                yield "", history, None
            
            logging.info(f"Chatbot Full Response: {full_response_text}")

            audio_output_path = await speak(full_response_text)
            yield "", history, audio_output_path # <-- Final yield for streaming path

        else:
            yield "", history, None
            return

    except Exception as e:
        error_message = f"An unexpected error occurred during agent execution: {e}"
        logging.error(error_message)
        current_chat_entry[1] = error_message
        audio_output_path = await speak(error_message)
        yield "", history, audio_output_path


# --- Sample Questions Data ---
sample_questions_data = {
    "Budgeting": [
        "What is the 50/30/20 rule for budgeting?",
        "How do I create a zero-based budget?",
        "Can you explain the envelope system for cash budgeting?",
        "What are common expenses I should track in a budget?",
        "How can I stick to my budget effectively?"
    ],
    "Savings": [
        "How much should I save from my income each month?",
        "What is an emergency fund and why is it important?",
        "Tips for saving money on daily expenses?",
        "What is compound interest and how does it help savings?",
        "How can I automate my savings?"
    ],
    "Investment": [
        "What are ETFs and how do they work?",
        "What is diversification in investing?",
        "What are some low-risk investment options?",
        "How does inflation affect my investments?",
        "What is a Roth IRA and who is it for?"
    ],
    "Credit Score": [
        "What factors affect my credit score?",
        "How long does negative information stay on my credit report?",
        "What is a good credit score range?",
        "Should I close old credit card accounts?",
        "How can I get a free credit report?"
    ],
    "Retirement Planning": [
        "When should I start saving for retirement?",
        "What is a 401(k) and how does it work?",
        "What is the average retirement age in the US?",
        "How much money do I need to retire comfortably?",
        "Can you explain Social Security benefits?"
    ]
}


# --- Gradio Wrapper for Salary Slip Processing ---
async def process_salary_slip_wrapper(file_obj, history) -> Generator[Tuple[str, List[List[str | None]], Optional[str]], None, None]:
    if file_obj is None:
        yield "", history, None
        return

    file_path = file_obj.name
    user_message_for_history = f"Attempting to process salary slip: {os.path.basename(file_path)}"

    try:
        if file_path.lower().endswith('.pdf'):
            response_for_history = "PDF processing is not fully implemented in this demo. Please convert your PDF to text and paste it."
            history.append([user_message_for_history, response_for_history])
            yield "", history, await speak(response_for_history)
            return
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            async for current_textbox, current_history, current_audio_path in chatbot_response(
                None, 
                None, 
                history, 
                uploaded_salary_slip_content=content 
            ):
                yield current_textbox, current_history, current_audio_path


    except Exception as e:
        logging.error(f"Error in process_salary_slip_wrapper: {e}")
        response_for_history = f"Error processing file: {e}"
        history.append([user_message_for_history, response_for_history])
        yield "", history, await speak(response_for_history)


# --- Gradio Interface Setup ---
with gr.Blocks(theme=gr.themes.Soft(), title="Financial Literacy Chatbot") as demo:
    gr.Markdown(
        """
        # 📈 Financial Literacy Chatbot
        Your personal AI assistant for financial advice, budgeting, and knowledge.
        Select a category below to see sample questions, or ask your own!
        """
    )

    gr.Markdown("## Select a Category:")
    with gr.Row():
        category_budget_btn = gr.Button("Budgeting")
        category_savings_btn = gr.Button("Savings")
        category_investment_btn = gr.Button("Investment")
        category_credit_btn = gr.Button("Credit Score")
        category_retirement_btn = gr.Button("Retirement Planning")

    gr.Markdown("## Sample Questions for Selected Category:")
    sample_questions_row = gr.Row(visible=False)
    with gr.Row():
        sq_btn1 = gr.Button("")
        sq_btn2 = gr.Button("")
        sq_btn3 = gr.Button("")
        sq_btn4 = gr.Button("")
        sq_btn5 = gr.Button("")
        sample_question_btns = [sq_btn1, sq_btn2, sq_btn3, sq_btn4, sq_btn5]

        sample_question_text_inputs = [
            gr.Textbox(visible=False, label=f"Hidden Sample Question {i+1}")
            for i in range(5)
        ]

    chatbot_display = gr.Chatbot(
        height=400,
        label="Conversation History",
        render_markdown=True,
        type='tuples',
        value=[[None, "Hello! I'm your Financial Literacy Chatbot. Choose a category above, speak your question, or type it below!"]]
    )

    gr.Markdown("## Ask Your Own Question:")
    with gr.Row():
        with gr.Column(scale=4):
            user_question_textbox = gr.Textbox(
                placeholder="Type your question here...",
                label="Your Question",
                autofocus=True
            )
            with gr.Row():
                send_text_btn = gr.Button("Send Text 🚀")
                audio_input_mic = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="Speak Your Question 🎤",
                )
        with gr.Column(scale=1):
            transcribe_btn = gr.Button("Transcribe Audio (to text box) 📝")
            record_send_btn = gr.Button("Record & Send Voice 🗣️")
            audio_output_player = gr.Audio(
                label="Chatbot Audio Response 🔊",
                autoplay=True
            )
            salary_slip_upload = gr.File(
                label="Upload Salary Slip (Text/PDF)",
                file_types=["text", ".pdf"],
                file_count="single"
            )
            process_slip_btn = gr.Button("Process Salary Slip 📊")

    # --- Gradio Event Handlers ---

    def update_sample_questions(category: str) -> Tuple[
        gr.update, gr.update, gr.update, gr.update, gr.update,
        gr.update, gr.update, gr.update, gr.update, gr.update,
        gr.update,
        List[List[str | None]],
        None
    ]:
        questions = sample_questions_data.get(category, ["No sample questions available."]*5)
        questions_padded = (questions + [""] * 5)[:5]

        button_label_updates = [gr.update(value=q) for q in questions_padded]
        hidden_textbox_value_updates = [gr.update(value=q) for q in questions_padded]

        return (
            *button_label_updates,
            *hidden_textbox_value_updates,
            gr.update(visible=True),
            [],
            None
        )

    category_budget_btn.click(
        fn=update_sample_questions,
        inputs=[gr.State("Budgeting")],
        outputs=sample_question_btns + sample_question_text_inputs + [sample_questions_row, chatbot_display, audio_output_player]
    )
    category_savings_btn.click(
        fn=update_sample_questions,
        inputs=[gr.State("Savings")],
        outputs=sample_question_btns + sample_question_text_inputs + [sample_questions_row, chatbot_display, audio_output_player]
    )
    category_investment_btn.click(
        fn=update_sample_questions,
        inputs=[gr.State("Investment")],
        outputs=sample_question_btns + sample_question_text_inputs + [sample_questions_row, chatbot_display, audio_output_player]
    )
    category_credit_btn.click(
        fn=update_sample_questions,
        inputs=[gr.State("Credit Score")],
        outputs=sample_question_btns + sample_question_text_inputs + [sample_questions_row, chatbot_display, audio_output_player]
    )
    category_retirement_btn.click(
        fn=update_sample_questions,
        inputs=[gr.State("Retirement Planning")],
        outputs=sample_question_btns + sample_question_text_inputs + [sample_questions_row, chatbot_display, audio_output_player]
    )

    async def on_sample_question_click(question_text: str, chat_history: List[List[str | None]]) -> Generator[Tuple[str, List[List[str | None]], Optional[str]], None, None]:
        async for clear_textbox, updated_history, audio_path in chatbot_response(
            message=question_text,
            audio_input_path=None,
            history=chat_history,
            uploaded_salary_slip_content=None
        ):
            yield clear_textbox, updated_history, audio_path

    for btn, text_input in zip(sample_question_btns, sample_question_text_inputs):
        btn.click(
            fn=on_sample_question_click,
            inputs=[text_input, chatbot_display],
            outputs=[user_question_textbox, chatbot_display, audio_output_player]
        )

    transcribe_btn.click(
        fn=transcribe_audio,
        inputs=[audio_input_mic],
        outputs=[user_question_textbox]
    )

    user_question_textbox.submit(
        fn=chatbot_response,
        inputs=[user_question_textbox, audio_input_mic, chatbot_display, gr.State(None)],
        outputs=[user_question_textbox, chatbot_display, audio_output_player]
    )
    send_text_btn.click(
        fn=chatbot_response,
        inputs=[user_question_textbox, audio_input_mic, chatbot_display, gr.State(None)],
        outputs=[user_question_textbox, chatbot_display, audio_output_player]
    )

    record_send_btn.click(
        fn=chatbot_response,
        inputs=[gr.State(None), audio_input_mic, chatbot_display, gr.State(None)],
        outputs=[user_question_textbox, chatbot_display, audio_output_player]
    )

    process_slip_btn.click(
        fn=process_salary_slip_wrapper,
        inputs=[salary_slip_upload, chatbot_display],
        outputs=[user_question_textbox, chatbot_display, audio_output_player]
    )

demo.queue().launch(share=True)