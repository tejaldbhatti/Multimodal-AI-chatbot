"""
FastAPI application entry point.
All routes are registered here.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.config import APP_TITLE, APP_VERSION

# Initialize FastAPI app
app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description="Personalised financial advisor and tax preparation assistant"
)

# Allow Gradio frontend to talk to FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)





# ── Import routers ───────────────────────────────────────
from backend.routers import chat, market, tax, upload

app.include_router(chat.router,   prefix="/chat",   tags=["Chat"])
app.include_router(market.router, prefix="/market", tags=["Market"])
app.include_router(tax.router,    prefix="/tax",    tags=["Tax"])
app.include_router(upload.router, prefix="/upload", tags=["Upload"])


# ── Health check ─────────────────────────────────────────
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "app": APP_TITLE,
        "version": APP_VERSION
    }


# ── Root ─────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Financial Chatbot API is running!",
        "docs": "/docs",
        "health": "/health"
    }