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

from dotenv import load_dotenv

load_dotenv()

import gradio as gr
from faster_whisper import WhisperModel
from gtts import gTTS

try:
    from backend.chatbot_backend import chatbot_respond
except ImportError as e:
    logging.error(
        "Failed to import backend.chatbot_backend: %s. Ensure the backend package is available.",
        e,
    )

    async def chatbot_respond(  # pylint: disable=unused-argument
        user_input: str,
        uploaded_salary_slip_content: Optional[str],
    ):
        yield "Error: Chatbot backend not loaded. Please check server logs."

try:
    import pypdf
    logging.info("PyPDF library loaded for PDF processing.")
except ImportError:
    logging.error("PyPDF not found. Install it with 'pip install pypdf'.")
    pypdf = None

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

try:
    whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
except Exception as e:  # pylint: disable=broad-except
    logging.error("Error loading Faster Whisper model: %s", e)
    whisper_model = None


async def transcribe_audio(audio_path: str) -> str:
    if not whisper_model:
        return "Error: Speech-to-text model not loaded."
    try:
        segments, _ = await asyncio.to_thread(whisper_model.transcribe, audio_path, beam_size=5)
        return " ".join([segment.text for segment in segments])
    except Exception as e:  # pylint: disable=broad-except
        logging.error("Error during audio transcription: %s", e)
        return f"Error: Could not transcribe audio. {e}"


async def speak(text: str) -> Optional[str]:
    try:
        if not text or not text.strip():
            return None
        audio_file_path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
        tts = gTTS(text=text, lang="en")
        await asyncio.to_thread(tts.save, audio_file_path)
        return audio_file_path
    except Exception as e:  # pylint: disable=broad-except
        logging.error("Error during text-to-speech: %s", e)
        return None


async def chatbot_response(
    message: str | None,
    audio_input_path: str | None,
    history: List[List[str | None]],
    uploaded_salary_slip_content: Optional[str] = None,
) -> AsyncGenerator[Tuple[str, List[List[str | None]], Optional[str]], Any]:
    if history is None:
        history = []

    user_input_text = ""
    if audio_input_path:
        transcribed = await transcribe_audio(audio_input_path)
        if transcribed.startswith("Error"):
            response_text = f"I could not process your audio: {transcribed}. Try typing your question or re-recording."
            history.append([message if message is not None else "Audio input error", response_text])
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

    full_response_text_for_tts = ""
    audio_output_path = None
    try:
        if uploaded_salary_slip_content:
            response_generator = chatbot_respond(
                user_input=user_input_text,
                uploaded_salary_slip_content=uploaded_salary_slip_content,
            )
            response_from_backend = await anext(response_generator)
            full_response_text_for_tts = response_from_backend
            current_chat_entry[1] = full_response_text_for_tts
            yield "", history, None
            audio_output_path = await speak(full_response_text_for_tts)
            yield "", history, audio_output_path
        elif user_input_text:
            async for chunk in chatbot_respond(user_input=user_input_text, uploaded_salary_slip_content=None):
                current_chat_entry[1] += chunk
                full_response_text_for_tts += chunk
                yield "", history, None
            audio_output_path = await speak(full_response_text_for_tts)
            yield "", history, audio_output_path
        else:
            yield "", history, None
            return
    except Exception as e:  # pylint: disable=broad-except
        error_message = f"An unexpected error occurred during agent execution: {e}"
        logging.error(error_message, exc_info=True)
        current_chat_entry[1] = error_message
        audio_output_path = await speak(error_message)
        yield "", history, audio_output_path


sample_questions_data = {
    "Budgeting": [
        "What is the 50/30/20 rule for budgeting?",
        "How do I create a zero-based budget?",
        "Can you explain the envelope system for cash budgeting?",
    ],
}

