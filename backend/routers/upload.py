"""
Upload router — handles file uploads.
Salary slips and bank statements.
"""
from fastapi import APIRouter, UploadFile, File, Form
from typing import Optional
import PyPDF2
import io
from backend.tools.document_analyzer import (
    analyze_salary_slip,
    analyze_bank_statement
)

router = APIRouter()


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extracts text from a PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        return f"Error extracting PDF text: {str(e)}"


@router.post("/salary-slip")
async def upload_salary_slip(
    file: UploadFile = File(...),
    country: str = Form(default="germany")
):
    """
    Upload a salary slip PDF and get analysis.
    Accepts PDF or text files.
    """
    file_bytes = await file.read()

    # Extract text based on file type
    if file.filename.endswith(".pdf"):
        text = extract_text_from_pdf(file_bytes)
    else:
        text = file_bytes.decode("utf-8")

    if not text or len(text) < 10:
        return {"error": "Could not extract text from file. Please try a text file."}

    result = analyze_salary_slip.invoke({
        "salary_slip_text": text,
        "country": country
    })

    return {
        "filename": file.filename,
        "country": country,
        "analysis": result
    }


@router.post("/bank-statement")
async def upload_bank_statement(
    file: UploadFile = File(...),
    country: str = Form(default="germany")
):
    """
    Upload a bank statement PDF and get spending analysis.
    """
    file_bytes = await file.read()

    if file.filename.endswith(".pdf"):
        text = extract_text_from_pdf(file_bytes)
    else:
        text = file_bytes.decode("utf-8")

    if not text or len(text) < 10:
        return {"error": "Could not extract text from file."}

    result = analyze_bank_statement.invoke({
        "statement_text": text,
        "country": country
    })

    return {
        "filename": file.filename,
        "country": country,
        "analysis": result
    }


@router.get("/health")
def upload_health():
    return {"status": "upload router ok"}