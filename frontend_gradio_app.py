"""
This module implements a Gradio web interface for a Financial Literacy Chatbot.
It integrates speech-to-text (Faster Whisper), text-to-speech (gTTS),
PDF processing (pypdf), and a custom chatbot backend.
"""

import asyncio
import logging
import os
import tempfile
from typing import Any, AsyncGenerator, List, Optional, Tuple

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# --- Third-party imports ---
import gradio as gr
from faster_whisper import WhisperModel
from gtts import gTTS

# --- Local application imports ---
# IMPORTANT: Ensure 'chatbot_backend.py' exists in the same directory and
# its dependencies are installed. This import will fail if it's not present,
# or if 'chatbot_respond' is not correctly defined within it.
try:
    from chatbot_backend import chatbot_respond # Import the core function
except ImportError as e:
    logging.error("Failed to import chatbot_backend: %s. "
                  "Ensure chatbot_backend.py is in the same directory "
                  "and 'chatbot_respond' is defined.", e)
    # Define a placeholder function to avoid NameError if import fails
    async def chatbot_respond( # pylint: disable=unused-argument
        user_input: str,
        uploaded_salary_slip_content: Optional[str]
    ):
        """
        Placeholder for chatbot_respond if the backend cannot be imported.
        """
        yield "Error: Chatbot backend not loaded. Please check server logs."


# --- PDF Processing Library ---
# You might need to install this: pip install pypdf
try:
    import pypdf
    logging.info("PyPDF2 (pypdf) library loaded for PDF processing.")
except ImportError:
    logging.error("PyPDF2 (pypdf) not found. Please install it with "
                  "'pip install pypdf' to enable PDF processing.")
    pypdf = None


# --- Configure logging ---
# Changed level to WARNING for production to reduce log overhead.
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration (from .env) - ONLY FOR CLIENT-SIDE DEPENDENCIES IF ANY ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Speech-to-Text (Faster Whisper) ---
try:
    # Consider using device="cuda" if you have a compatible GPU for faster transcription
    # pylint: disable=invalid-name
    whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    logging.info("Faster Whisper model loaded.")
except Exception as e: # pylint: disable=broad-except
    logging.error("Error loading Faster Whisper model: %s. "
                  "Speech-to-text functionality will be limited.", e)
    whisper_model = None

async def transcribe_audio(audio_path: str) -> str:
    """
    Transcribes audio from a given file path using Faster Whisper.

    Args:
        audio_path (str): The path to the audio file.

    Returns:
        str: The transcribed text or an error message.
    """
    if not whisper_model:
        return "Error: Speech-to-text model not loaded."
    try:
        segments, _ = await asyncio.to_thread( # pylint: disable=W0212
            whisper_model.transcribe, audio_path, beam_size=5
        )
        transcription = " ".join([segment.text for segment in segments])
        logging.info("Audio Transcribed: '%s'", transcription)
        return transcription
    except Exception as e: # pylint: disable=broad-except
        logging.error("Error during audio transcription: %s", e)
        return f"Error: Could not transcribe audio. {e}"

async def speak(text: str) -> Optional[str]:
    """
    Converts text to speech using gTTS and saves it to a temporary MP3 file.

    Args:
        text (str): The text to convert to speech.

    Returns:
        Optional[str]: The path to the generated audio file, or None if an
                       error occurred or text is empty.
    """
    try:
        # Ensure there's actual text to speak before calling gTTS
        if not text or not text.strip():
            return None
        audio_file_path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
        tts = gTTS(text=text, lang='en')
        # pylint: disable=W0212,consider-using-with
        await asyncio.to_thread(tts.save, audio_file_path)
        logging.info("Text-to-speech audio saved to: %s", audio_file_path)
        return audio_file_path
    except Exception as e: # pylint: disable=broad-except
        logging.error("Error during text-to-speech: %s", e)
        return None

# --- Gradio Chatbot Response Function ---
async def chatbot_response( # pylint: disable=too-many-locals
    message: str | None,
    audio_input_path: str | None,
    history: List[List[str | None]],
    uploaded_salary_slip_content: Optional[str] = None
) -> AsyncGenerator[Tuple[str, List[List[str | None]], Optional[str]], Any]:
    """
    Processes user input (text or audio) and generates chatbot responses,
    handling streaming and optional salary slip content.

    Args:
        message (str | None): Text message from the user.
        audio_input_path (str | None): Path to an audio file for transcription.
        history (List[List[str | None]]): The chat history for Gradio.
        uploaded_salary_slip_content (Optional[str]): Extracted text content
                                                      from a salary slip file.

    Yields:
        Tuple[str, List[List[str | None]], Optional[str]]:
            - An empty string (to clear the user input textbox).
            - Updated chat history.
            - Path to an audio response file, or None.
    """
    if history is None:
        history = []

    user_input_text = ""
    if audio_input_path:
        logging.info("Processing audio input from: %s", audio_input_path)
        transcribed = await transcribe_audio(audio_input_path)
        if transcribed.startswith("Error"):
            response_text = (f"I could not process your audio: {transcribed}. "
                             "Try typing your question or re-recording.")
            history.append([message if message is not None else
                            "Audio input error", response_text])
            audio_output_path = await speak(response_text)
            yield "", history, audio_output_path
            return # Ensure to return from the async generator
        user_input_text = transcribed
    elif message:
        user_input_text = message.strip()

    display_user_input = user_input_text if user_input_text else "File Uploaded"

    # Add user input to history immediately
    current_chat_entry = [display_user_input, ""]
    history.append(current_chat_entry)
    yield "", history, None # Yield to update UI with user's message

    full_response_text_for_tts = "" # Accumulate full response for TTS at the end
    audio_output_path = None
    try:
        if uploaded_salary_slip_content:
            logging.info("Calling backend for salary slip analysis via chatbot_respond.")
            # For salary slip, the backend returns the full response at once,
            # not streaming chunks.
            response_generator = chatbot_respond(
                user_input=user_input_text,
                uploaded_salary_slip_content=uploaded_salary_slip_content
            )
            # Use anext() for Python 3.10+
            response_from_backend = await anext(response_generator)

            full_response_text_for_tts = response_from_backend
            # Update the chat history entry with the full response
            current_chat_entry[1] = full_response_text_for_tts
            yield "", history, None # Update UI with full text

            logging.info("Chatbot Full Response for Salary Slip: %s",
                         full_response_text_for_tts)

            audio_output_path = await speak(full_response_text_for_tts)
            yield "", history, audio_output_path # Final yield with audio

        elif user_input_text:
            logging.info("Calling backend with user input: '%s' (streaming expected)",
                         user_input_text)
            # Iterate over streaming chunks from the backend
            async for chunk in chatbot_respond(user_input=user_input_text,
                                               uploaded_salary_slip_content=None):
                current_chat_entry[1] += chunk # Append new chunk to the current response
                full_response_text_for_tts += chunk # Accumulate for TTS
                yield "", history, None # Yield to update Gradio UI with the new chunk

            logging.info("Chatbot Full Response (for TTS): %s",
                         full_response_text_for_tts)

            # Once all chunks are received, generate audio for the complete response
            audio_output_path = await speak(full_response_text_for_tts)
            yield "", history, audio_output_path # Final yield with audio

        else:
            # If no message and no audio, and no slip, do nothing significant
            yield "", history, None
            return

    except Exception as e: # pylint: disable=broad-except
        error_message = f"An unexpected error occurred during agent execution: {e}"
        logging.error("An unexpected error occurred during agent execution: %s",
                      e, exc_info=True)
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
    ],
    "Debt Repayment": [
        "What are the different ways to pay off debt?",
        "Explain the debt snowball method.",
        "What is the debt avalanche method and how does it work?",
        "Can you tell me about debt consolidation?",
        "What's a balance transfer and when should I consider it?"
    ]
}


# --- Gradio Wrapper for Salary Slip Processing ---
async def process_salary_slip_wrapper(
    file_obj: Any,
    history: List[List[str | None]]
) -> AsyncGenerator[Tuple[str, List[List[str | None]], Optional[str]], Any]:
    """
    Processes an uploaded salary slip file (PDF or text), extracts its content,
    and then passes it to the chatbot backend for analysis.

    Args:
        file_obj (Any): The file object provided by Gradio (e.g., gr.File).
        history (List[List[str | None]]): The chat history for Gradio.

    Yields:
        Tuple[str, List[List[str | None]], Optional[str]]:
            - An empty string (to clear the user input textbox).
            - Updated chat history.
            - Path to an audio response file, or None.
    """
    if file_obj is None:
        yield "", history, None
        return

    file_path = file_obj.name
    user_message_for_history = (f"Attempting to process salary slip: "
                                f"{os.path.basename(file_path)}")

    try:
        content = ""
        if file_path.lower().endswith('.pdf'):
            if pypdf is None:
                response_for_history = ("PDF processing library (pypdf) is not installed. "
                                        "Please install it with 'pip install pypdf' to "
                                        "enable PDF processing.")
                history.append([user_message_for_history, response_for_history])
                yield "", history, await speak(response_for_history)
                return

            # Extract text from PDF
            reader = pypdf.PdfReader(file_path)
            for page in reader.pages:
                content += page.extract_text() + "\n"
            logging.info("Extracted text from PDF: %s", file_path)
        else: # Assume it's a text file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            logging.info("Read text from file: %s", file_path)

        # Call chatbot_response with the extracted content
        async for current_textbox, current_history, current_audio_path in chatbot_response(
            None, # No direct text message, content comes from file
            None, # No audio input
            history,
            uploaded_salary_slip_content=content
        ):
            yield current_textbox, current_history, current_audio_path


    except Exception as e: # pylint: disable=broad-except
        logging.error("Error in process_salary_slip_wrapper: %s", e, exc_info=True)
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
        category_debt_btn = gr.Button("Debt Repayment")

    gr.Markdown("## Sample Questions for Selected Category:")
    sample_questions_row = gr.Row(visible=False)
    with gr.Row():
        sq_btn1 = gr.Button("") # pylint: disable=invalid-name
        sq_btn2 = gr.Button("") # pylint: disable=invalid-name
        sq_btn3 = gr.Button("") # pylint: disable=invalid-name
        sq_btn4 = gr.Button("") # pylint: disable=invalid-name
        sq_btn5 = gr.Button("") # pylint: disable=invalid-name
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
        value=[[None, "Hello! I'm your Financial Literacy Chatbot. Choose a category "
                "above, speak your question, or type it below!"]]
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
        Any, Any, Any, Any, Any, # 5 for sq_btn updates (gr.update objects)
        Any, Any, Any, Any, Any, # 5 for text_input updates (gr.update objects)
        Any, # 1 for sample_questions_row update (gr.update object)
        List[List[str | None]], # 1 for chatbot_display update (clearing history)
        Optional[str] # 1 for audio_output_player update (path or None)
    ]:
        """
        Updates the sample questions displayed based on the selected category.
        Clears the chat history and audio output when a new category is selected.

        Args:
            category (str): The selected category (e.g., "Budgeting").

        Returns:
            Tuple: A tuple of Gradio updates to reset buttons, text inputs,
                   visibility, chat history, and audio player.
        """
        questions = sample_questions_data.get(category, ["No sample questions available."]*5)
        # Ensure 5 questions, padding with empty if less
        questions_padded = (questions + [""] * 5)[:5]

        button_label_updates = [gr.update(value=q) for q in questions_padded]
        hidden_textbox_value_updates = [gr.update(value=q) for q in questions_padded]

        # The order of returned values must match the outputs defined in the .click() events
        return (
            *button_label_updates,
            *hidden_textbox_value_updates,
            gr.update(visible=True), # Make the sample questions row visible
            [], # Clear chat history when new category is selected
            None # Clear any playing audio when new category is selected
        )

    category_budget_btn.click( # pylint: disable=no-member
        fn=update_sample_questions,
        inputs=[gr.State("Budgeting")],
        outputs=sample_question_btns + sample_question_text_inputs + \
                [sample_questions_row, chatbot_display, audio_output_player]
    )
    category_savings_btn.click( # pylint: disable=no-member
        fn=update_sample_questions,
        inputs=[gr.State("Savings")],
        outputs=sample_question_btns + sample_question_text_inputs + \
                [sample_questions_row, chatbot_display, audio_output_player]
    )
    category_investment_btn.click( # pylint: disable=no-member
        fn=update_sample_questions,
        inputs=[gr.State("Investment")],
        outputs=sample_question_btns + sample_question_text_inputs + \
                [sample_questions_row, chatbot_display, audio_output_player]
    )
    category_credit_btn.click( # pylint: disable=no-member
        fn=update_sample_questions,
        inputs=[gr.State("Credit Score")],
        outputs=sample_question_btns + sample_question_text_inputs + \
                [sample_questions_row, chatbot_display, audio_output_player]
    )
    category_retirement_btn.click( # pylint: disable=no-member
        fn=update_sample_questions,
        inputs=[gr.State("Retirement Planning")],
        outputs=sample_question_btns + sample_question_text_inputs + \
                [sample_questions_row, chatbot_display, audio_output_player]
    )
    category_debt_btn.click( # pylint: disable=no-member
        fn=update_sample_questions,
        inputs=[gr.State("Debt Repayment")],
        outputs=sample_question_btns + sample_question_text_inputs + \
                [sample_questions_row, chatbot_display, audio_output_player]
    )

    async def on_sample_question_click(
        question_text: str,
        chat_history: List[List[str | None]]
    ) -> AsyncGenerator[Tuple[str, List[List[str | None]], Optional[str]], Any]:
        """
        Handles the event when a sample question button is clicked.
        Feeds the sample question into the chatbot response.

        Args:
            question_text (str): The text of the sample question.
            chat_history (List[List[str | None]]): The current chat history.

        Yields:
            Tuple: The chatbot response (textbox update, updated history, audio path).
        """
        async for clear_textbox, updated_history, audio_path in chatbot_response(
            message=question_text,
            audio_input_path=None,
            history=chat_history,
            uploaded_salary_slip_content=None
        ):
            yield clear_textbox, updated_history, audio_path

    for btn, text_input in zip(sample_question_btns, sample_question_text_inputs):
        btn.click( # pylint: disable=no-member
            fn=on_sample_question_click,
            inputs=[text_input, chatbot_display],
            outputs=[user_question_textbox, chatbot_display, audio_output_player]
        )

    transcribe_btn.click( # pylint: disable=no-member
        fn=transcribe_audio,
        inputs=[audio_input_mic],
        outputs=[user_question_textbox]
    )

    # Main interaction points:
    # 1. User types in textbox and presses Enter
    user_question_textbox.submit( # pylint: disable=no-member
        fn=chatbot_response,
        inputs=[user_question_textbox, gr.State(None), chatbot_display, gr.State(None)],
        outputs=[user_question_textbox, chatbot_display, audio_output_player]
    )
    # 2. User types in textbox and clicks "Send Text" button
    send_text_btn.click( # pylint: disable=no-member
        fn=chatbot_response,
        inputs=[user_question_textbox, gr.State(None), chatbot_display, gr.State(None)],
        outputs=[user_question_textbox, chatbot_display, audio_output_player]
    )

    # 3. User records audio and clicks "Record & Send Voice" button
    record_send_btn.click( # pylint: disable=no-member
        fn=chatbot_response,
        inputs=[gr.State(None), audio_input_mic, chatbot_display, gr.State(None)],
        outputs=[user_question_textbox, chatbot_display, audio_output_player]
    )

    # 4. User uploads salary slip and clicks "Process Salary Slip" button
    process_slip_btn.click( # pylint: disable=no-member
        fn=process_salary_slip_wrapper,
        inputs=[salary_slip_upload, chatbot_display],
        outputs=[user_question_textbox, chatbot_display, audio_output_player]
    )

demo.queue().launch(share=True)