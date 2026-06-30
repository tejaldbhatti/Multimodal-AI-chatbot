# 🤖 Financial Literacy & Tax Advisor Chatbot

A production-grade, personalised financial advisor and tax preparation assistant built with LangGraph, FastAPI, and OpenAI. Supports Germany, USA, India, and Australia with real-time market data and country-specific tax calculations.

> ⚠️ **Legal Disclaimer:** This chatbot provides financial education and tax preparation assistance only. It does not replace a licensed Steuerberater (§2 StBerG), CPA, CA, or registered tax agent.

---

## 🎯 What makes this different from ChatGPT?

| Feature | ChatGPT | This Chatbot |
|---|---|---|
| Country-specific tax rules | Generic | DE, USA, IN, AU official rules |
| Real-time market data | ❌ Static | ✅ Live via Alpha Vantage |
| Persistent user profile | ❌ Forgets | ✅ Remembered via Supabase |
| Freelancer tax calculator | ❌ | ✅ VAT, Gewerbesteuer, GST |
| Ehegattensplitting calculator | ❌ | ✅ III/V vs IV/IV comparison |
| Salary slip analysis | Basic | ✅ Deep analysis via GPT-4o |
| Source citations | ❌ | ✅ Official government sources |
| Legal disclaimer on tax | ❌ | ✅ Mandatory, automatic |
| Goal tracking | ❌ | ✅ Across sessions |

---

## 🏗️ Architecture

```
User (Voice or Text)
        ↓
┌─────────────────────────────────────────────┐
│              FastAPI Backend                 │
│  POST /chat  POST /tax/*  POST /market/*    │
│  POST /upload/*  GET /health                │
└─────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────┐
│           LangGraph Agent Flow               │
│                                             │
│  detect_country → load_user_profile         │
│       → classify_intent                     │
│       → generate_response                   │
└─────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────┐
│              Tools & Data                    │
│                                             │
│  📊 Market: Alpha Vantage (live data)       │
│  📰 News: NewsAPI (latest headlines)        │
│  💰 Tax: 7 tax calculator tools             │
│  📄 Docs: GPT-4o document analyzer         │
│  🗄️  DB: Supabase (user profiles)          │
│  🔍 RAG: Pinecone (knowledge base)         │
└─────────────────────────────────────────────┘
        ↓
Personalised Response (text + optional voice)
```

---

## 🌍 Supported Countries & Features

### 🇩🇪 Germany
- Progressive income tax (14% – 45%)
- All Steuerklassen (I – VI)
- Ehegattensplitting calculator (III/V vs IV/IV)
- Freelancer: Einkommensteuer, Umsatzsteuer (19%/7%), Gewerbesteuer
- Kleinunternehmer threshold (€22,000)
- Single parent Steuerklasse II + Entlastungsbetrag
- Investment: ETF, Riester Rente, bAV, Freistellungsauftrag
- Steuerberater preparation checklist

### 🇺🇸 USA
- Federal tax brackets (10% – 37%)
- Married filing jointly vs separately comparison
- Self-employment tax (15.3%) + QBI deduction
- Retirement: 401(k), Roth IRA, SEP-IRA limits
- Head of Household filing for single parents

### 🇮🇳 India
- Old vs New tax regime comparison
- Section 80C deductions (₹1.5L limit)
- Presumptive taxation 44ADA for freelancers
- GST registration threshold
- Investment: PPF, NPS, ELSS, FD

### 🇦🇺 Australia
- Progressive tax + Medicare levy (2%)
- Sole trader tax with GST threshold
- Superannuation (11% employer contribution)
- Capital gains discount (50% for assets > 12 months)
- Low income tax offset (LITO)

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| LLM (Chat) | OpenAI GPT-4o-mini | Fast, cheap general responses |
| LLM (Docs) | OpenAI GPT-4o | Long document analysis |
| Orchestration | LangGraph | Multi-step agent flow |
| Framework | LangChain | Tools, memory, RAG |
| API | FastAPI | Production REST backend |
| Vector DB | Pinecone | RAG knowledge base |
| Database | Supabase (PostgreSQL) | User profiles, goals |
| Market Data | Alpha Vantage | Live stocks, ETF, forex |
| News | NewsAPI | Latest financial headlines |
| UI | Gradio | Voice + text interface |
| Evaluation | LangSmith + RAGAs | LLM output quality |
| Deployment | Docker + Google Cloud Run | Production deployment |
| CI/CD | GitHub Actions | Auto deploy on push |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker Desktop
- API keys (see Environment Variables)

### Local Setup

```bash
# Clone the repository
git clone https://github.com/tejaldbhatti/financial-chatbot
cd financial-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Run the API
uvicorn backend.main:app --reload

# Visit http://localhost:8000/docs
```

### Docker Setup

```bash
# Build and run with Docker
docker build -t financial-chatbot .
docker run -p 8000:8000 --env-file .env financial-chatbot

# Or use docker-compose
docker compose up
```

---

## 🔑 Environment Variables

```bash
# Required
OPENAI_API_KEY=           # OpenAI API key
PINECONE_API_KEY=         # Pinecone vector DB
PINECONE_ENVIRONMENT=     # Pinecone region
LANGCHAIN_API_KEY=        # LangSmith tracing

# Phase 2 — Market Data
ALPHA_VANTAGE_API_KEY=    # Live stock/ETF/forex
NEWS_API_KEY=             # Financial news headlines

# Phase 3 — Database
SUPABASE_URL=             # Supabase project URL
SUPABASE_KEY=             # Supabase anon key
```

---

## 📡 API Endpoints

### Chat
```
POST /chat/              → main chat endpoint
```

### Market Data (Live)
```
POST /market/stock       → live stock price
POST /market/forex       → live exchange rate
POST /market/etf         → live ETF price
POST /market/news        → latest financial news
```

### Tax Calculators
```
POST /tax/employee       → employee income tax
POST /tax/freelancer     → freelancer tax (VAT, GST, SE tax)
POST /tax/couple         → married couple tax (Ehegattensplitting)
POST /tax/single-parent  → single parent tax (Steuerklasse II)
POST /tax/steuerklasse   → recommend best Steuerklasse
POST /tax/checklist      → Steuerberater prep checklist
```

### Document Upload
```
POST /upload/salary-slip     → analyze salary slip
POST /upload/bank-statement  → analyze bank statement
```

---

## 💬 Example Conversations

**Tax calculation:**
```
User: I am a freelancer in Germany earning €80,000. How much tax?
Bot:  Gross: €80,000 | Net profit (after expenses): €70,000
      Income tax: €14,058 | VAT: €15,200 (must pay to Finanzamt)
      Take home: €55,942/yr | €4,662/mo
```

**Couple tax:**
```
User: My husband earns €80,000 and I earn €30,000. Which Steuerklasse?
Bot:  III/V saves €2,383/yr vs IV/IV.
      Partner 1 → Steuerklasse III
      Partner 2 → Steuerklasse V
```

**Live market data:**
```
User: What is Apple stock price?
Bot:  AAPL: $293.08 📈 +$1.22 (+0.41%)
      High: $299.70 | Low: $292.94 | Volume: 53M
```

---

## 📊 Evaluation Results

| Category | Score |
|---|---|
| Overall Accuracy | 97.5% |
| Germany Tax | 100% |
| Germany Couple Tax (Ehegattensplitting) | 75% |
| Germany Single Parent | 100% |
| USA Tax | 100% |
| India Education | 100% |
| Australia Education | 100% |
| Live Market Data | 100% |
| Forex Rates | 100% |
| India Investment | 100% |
| Germany Investment | 100% |
| **Total Tests Passed** | **10/10** |

---

## 🗂️ Project Structure

```
financial-chatbot/
├── backend/
│   ├── agents/
│   │   ├── state.py         # LangGraph state schema
│   │   ├── nodes.py         # Agent nodes
│   │   └── graph.py         # LangGraph flow
│   ├── data/
│   │   └── country_rules.py # Tax rules for 4 countries
│   ├── tools/
│   │   ├── market_data.py   # Live market tools
│   │   ├── tax_employee.py  # Employee tax calculator
│   │   ├── tax_freelancer.py# Freelancer tax calculator
│   │   ├── tax_couple.py    # Ehegattensplitting
│   │   ├── tax_single_parent.py # Single parent tax
│   │   ├── tax_class_advisor.py # Steuerklasse advisor
│   │   ├── steuerberater_prep.py# Checklist generator
│   │   └── document_analyzer.py # Salary slip + bank statement
│   ├── routers/
│   │   ├── chat.py          # /chat endpoints
│   │   ├── market.py        # /market endpoints
│   │   ├── tax.py           # /tax endpoints
│   │   └── upload.py        # /upload endpoints
│   ├── db/
│   │   └── supabase_client.py # Database operations
│   ├── config.py            # All environment variables
│   └── main.py              # FastAPI entry point
├── frontend/
│   └── gradio_app.py        # Voice + text UI
├── evaluation/
│   ├── create_dataset.py    # LangSmith dataset
│   └── run_evaluation.py    # RAGAs evaluation
├── docs/
│   └── phase1_phase2_documentation.md
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 🎓 AI Engineering Concepts Demonstrated

- ✅ RAG (Retrieval Augmented Generation) with Pinecone
- ✅ Agentic AI with LangGraph multi-node orchestration
- ✅ Tool calling with OpenAI function calling
- ✅ Multi-model routing (GPT-4o-mini vs GPT-4o)
- ✅ Real-time data integration via MCP pattern
- ✅ FastAPI production REST backend
- ✅ Streaming LLM responses
- ✅ Vector database (Pinecone)
- ✅ LLM evaluation with LangSmith + RAGAs
- ✅ Prompt engineering with system prompts
- ✅ Data persistence with Supabase
- ✅ Docker containerisation
- ✅ Cloud deployment (Google Cloud Run)
- ✅ CI/CD with GitHub Actions
- ✅ Voice input (Whisper) + output (gTTS)
- ✅ Document analysis with GPT-4o

---

## 👩‍💻 Author

**Tejal Bhatti** — AI Engineer  
📍 Nußloch, Germany  
🔗 [LinkedIn](https://www.linkedin.com/in/tejal-bhatti-dataanalyst/)  
🐙 [GitHub](https://github.com/tejaldbhatti)  
📧 tejaldbhatti@gmail.com

---

## ⚖️ Legal Notice

This application provides financial education and tax preparation assistance only. All tax calculations are estimates for educational purposes. This tool does not constitute professional financial or tax advice and cannot replace:
- A licensed **Steuerberater** in Germany (§2 StBerG)
- A licensed **CPA** in the USA
- A licensed **CA** in India
- A registered **Tax Agent** in Australia

Always consult a qualified professional for your specific situation.
