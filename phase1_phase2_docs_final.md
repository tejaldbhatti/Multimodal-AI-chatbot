# Financial Chatbot — Phase 1 & Phase 2 Documentation
**Author:** Tejal Bhatti  
**Project:** Financial Literacy & Tax Advisor Chatbot  
**Last updated:** June 2026

---

## What is this project?

A personalised financial advisor and tax preparation assistant chatbot that:
- Detects which country the user is from
- Gives country-specific tax advice (Germany, USA, India, Australia)
- Fetches live stock prices, ETF data, and forex rates
- Calculates income tax based on real tax brackets
- Adds a legal disclaimer on all tax advice automatically

---

## How the project is structured

```
financial-chatbot/
├── backend/
│   ├── agents/
│   │   ├── state.py         ← what the agent remembers
│   │   ├── nodes.py         ← what each step does
│   │   └── graph.py         ← how steps connect together
│   ├── data/
│   │   └── country_rules.py ← tax and investment rules for 4 countries
│   ├── tools/
│   │   ├── country_advisor.py ← tool to fetch country rules
│   │   └── market_data.py     ← tool to fetch live market data
│   ├── routers/
│   │   ├── chat.py          ← /chat API endpoint
│   │   └── market.py        ← /market API endpoints
│   ├── config.py            ← all environment variables
│   └── main.py              ← FastAPI app entry point
├── frontend/
│   └── gradio_app.py        ← UI (Phase 4)
├── .env                     ← API keys (never commit this)
└── requirements.txt
```

---

## Technologies used

| Technology | What it does | Why we chose it |
|---|---|---|
| LangChain | Connects LLMs to tools and memory | Industry standard for AI apps |
| LangGraph | Orchestrates multi-step agent flow | Controls decision making |
| OpenAI GPT-4o-mini | Generates all responses | Fast, cheap, accurate |
| OpenAI GPT-4o | Analyses long documents | Better for PDFs (Phase 3) |
| Pinecone | Stores and searches transcripts | Vector search for RAG |
| FastAPI | REST API backend | Production grade, fast |
| Alpha Vantage | Live stock/ETF/forex data | Free tier, reliable |
| NewsAPI | Latest financial headlines | Free tier, 100 calls/day |
| Supabase | User profile database | Free PostgreSQL (Phase 3) |

---

# PHASE 1 — LangGraph Agent + Country Detection + Tax Rules

## What Phase 1 builds

Phase 1 replaces a simple linear chatbot with a smart multi-step agent that:
1. Detects which country the user is from
2. Loads the user's profile
3. Classifies what the user wants
4. Generates a personalised response with correct tax rules

---

## File 1 — `backend/agents/state.py`

### What is it?
This file defines **what the agent remembers** throughout the conversation. Think of it as the agent's memory or notebook.

### Why do we need it?
Without state, the agent forgets everything between steps. State lets every node read and write shared information.

### How it works

```python
class UserProfile(TypedDict):
    country: Optional[str]           # which country the user is from
    employment_type: Optional[str]   # employed / freelancer / both
    family_situation: Optional[str]  # single / married / single parent
    age: Optional[int]
    monthly_income: Optional[float]
    risk_tolerance: Optional[str]    # low / medium / high


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # full conversation history
    user_profile: UserProfile                # who the user is
    country: Optional[str]                   # detected country
    intent: Optional[str]                    # what user wants
    tax_disclaimer_needed: bool              # add legal disclaimer?
    context: Optional[str]                   # extra tool output
```

### Simple explanation
Imagine you are filling a form while talking to someone:
- `messages` = the full conversation transcript
- `user_profile` = the form about the user (name, age, income etc.)
- `country` = which country box you ticked
- `intent` = what they came for (tax help / investment / education)
- `tax_disclaimer_needed` = should we print a legal warning?

Every node in the graph reads this state and can update it.

---

## File 2 — `backend/agents/nodes.py`

### What is it?
This file defines **each step** the agent takes. Each step is called a "node."

### Why do we need it?
Different tasks need different logic. Country detection is different from tax calculation. Nodes keep each task separate and clean.

### The 4 nodes

---

#### Node 1 — `detect_country`

**What it does:** Reads the user's message and figures out which country they are from.

**How it works:**
```
Step 1: Look for country keywords in the message
        e.g. "steuer" → germany, "irs" → usa, "80c" → india

Step 2: If keywords found → set country

Step 3: If no keywords → ask GPT-4o-mini to detect country
        (returns only one word: germany / usa / india / australia / unknown)

Step 4: Save country to state
```

**Example:**
```
User message: "I live in Germany and want to invest"
Keywords found: "germany" → country = "germany"
```

---

#### Node 2 — `load_user_profile`

**What it does:** Loads the user's profile from state. If country was just detected, adds it to the profile.

**How it works:**
```
Step 1: Get current profile from state (may be empty for new users)
Step 2: If country detected but not in profile → add it
Step 3: Save updated profile to state
```

**Why this matters:** In Phase 3 we will load the profile from Supabase database so the chatbot remembers users across sessions.

---

#### Node 3 — `classify_intent`

**What it does:** Figures out what the user actually wants.

**How it works:**
```
Check message against keyword lists:

market     → "stock price", "exchange rate", "apple stock", "eur to usd"
tax        → "tax", "steuer", "freelancer", "how much tax", "80c"
investment → "should i invest", "best investment", "etf", "ppf"
education  → "what is", "explain", "how does", "tell me about"
calculation→ "calculate", "compound interest", "how much will i have"
goal       → "save for", "retirement", "emergency fund", "in 5 years"
```

**Example:**
```
User: "What is the current Apple stock price?"
Matches: "stock price", "apple stock" → intent = "market"
tax_disclaimer_needed = False (not a tax question)
```

```
User: "How much tax do I pay as a freelancer?"
Matches: "tax", "freelancer" → intent = "tax"
tax_disclaimer_needed = True (tax question → add disclaimer)
```

---

#### Node 4 — `generate_response`

**What it does:** Generates the final answer using all the information collected by previous nodes.

**How it works:**
```
Step 1: Load country rules (tax brackets, investment options, freelancer rules)

Step 2: Build a system prompt with:
        - Country context (tax brackets, currency, investment options)
        - User profile (age, income, family situation)
        - Intent (what they want)
        - Instructions for the LLM

Step 3: If tax question → add disclaimer instruction to prompt

Step 4: Call LLM with tools available
        (tools: get_stock_price, get_forex_rate, get_etf_price, get_financial_news)

Step 5: If LLM called a tool → execute the tool → get live data

Step 6: If tool was used → ask LLM to summarise tool result into a clear answer

Step 7: Return final answer + updated messages
```

**Why tools are important:**
```
Without tools:
User: "What is Apple stock price?"
LLM: "I don't have real-time data..." ❌

With tools:
User: "What is Apple stock price?"
LLM calls get_stock_price("AAPL") → gets $293.08
LLM: "Apple is currently trading at $293.08, down 0.41% today" ✅
```

**Key lesson:** `bind_tools()` gives LLM access to tools. But you must also tell the LLM in the system prompt to use them — otherwise it ignores them.

---

## File 3 — `backend/agents/graph.py`

### What is it?
This file **connects all nodes** into a flow. Think of it as the map that shows which step comes after which.

### How it works

```python
graph = StateGraph(AgentState)

# Add all nodes
graph.add_node("detect_country",    detect_country)
graph.add_node("load_user_profile", load_user_profile)
graph.add_node("classify_intent",   classify_intent)
graph.add_node("generate_response", generate_response)

# Connect them in order
graph.set_entry_point("detect_country")
graph.add_edge("detect_country",    "load_user_profile")
graph.add_edge("load_user_profile", "classify_intent")

# After intent → route to response
graph.add_conditional_edges("classify_intent", route_intent, {...})
graph.add_edge("generate_response", END)
```

### Visual flow

```
User sends message
        ↓
[detect_country]      → figures out: germany / usa / india / australia
        ↓
[load_user_profile]   → loads or initialises user profile
        ↓
[classify_intent]     → figures out: tax / market / investment / education / goal
        ↓
[generate_response]   → builds personalised answer using country rules + tools
        ↓
Response sent to user
```

### What is `conditional_edges`?
Normal edges always go to the same next node. Conditional edges choose the next node based on logic. 

In Phase 3 we will use this to route:
- Document upload → document analyzer node
- Tax question → tax calculator node
- Market question → market data node

---

## File 4 — `backend/data/country_rules.py`

### What is it?
A structured Python dictionary with tax brackets, investment options, freelancer rules, and retirement info for all 4 countries.

### Why structured data and not PDFs?
- Faster to query than searching a PDF
- Easy to update when tax rules change
- No PDF loading overhead
- Official PDFs are added to Pinecone in Phase 3 for deeper RAG answers

### What it contains

**Germany:**
```python
"germany": {
    "tax_brackets": [
        {"min": 0,      "max": 11604,  "rate": 0.00},  # tax free
        {"min": 11604,  "max": 17005,  "rate": 0.14},  # 14%
        {"min": 17005,  "max": 66760,  "rate": 0.24},  # 24%
        {"min": 66760,  "max": 277826, "rate": 0.42},  # 42%
        {"min": 277826, "max": None,   "rate": 0.45},  # 45%
    ],
    "freelancer": {
        "kleinunternehmer_threshold": 22000,  # no VAT below this
        "vat_standard": 0.19,                 # 19% VAT
        "gewerbesteuer_free": 24500,          # no trade tax below this
    },
    "investment_vehicles": ["ETF", "Riester Rente", "bAV", ...],
    "retirement": {"age": 67, "vehicles": ["Riester", "Rürup", ...]},
}
```

Same structure exists for USA, India, and Australia.

---

## File 5 — `backend/tools/country_advisor.py`

### What is it?
LangChain tools that read from `country_rules.py` and format the output for the LLM.

### Two tools

**Tool 1 — `get_country_financial_rules`**
```
Input:  country name (e.g. "germany")
Output: formatted text with tax brackets, investment options, links
```

**Tool 2 — `get_tax_bracket`**
```
Input:  country + annual income
Output: step by step tax calculation with breakdown

Example:
  Input:  country="germany", annual_income=50000
  Output: EUR 50,000 → tax EUR 8,674.94 → effective rate 17.3%
          take home EUR 41,325.06
```

---

## File 6 — `backend/config.py`

### What is it?
Central place where all environment variables and constants are loaded. Every other file imports from here — never reads `.env` directly.

### Why centralise config?
- One place to change if a key changes
- Validation at startup — warns if keys are missing
- Model routing logic — decides which OpenAI model to use

```python
OPENAI_MODEL_CHAT = "gpt-4o-mini"   # fast + cheap for chat
OPENAI_MODEL_DOCS = "gpt-4o"        # powerful for documents

def get_model_for_task(task: str) -> str:
    if task in ["document", "salary_slip", "bank_statement"]:
        return OPENAI_MODEL_DOCS    # use gpt-4o for long documents
    return OPENAI_MODEL_CHAT        # use gpt-4o-mini for everything else
```

---

## Phase 1 Test Results

```
User: "I live in Germany, I am a freelancer earning 60000 euros, how much tax?"
Country detected: germany
Intent detected:  tax

Response:
  Tax bracket 14% on EUR 11,604 – 17,005 = EUR 756.14
  Tax bracket 24% on EUR 17,005 – 60,000 = EUR 10,319.28
  Total tax: EUR 11,075.42
  Take home: EUR 48,924.58
  ⚠️ Disclaimer added automatically
```

---

# PHASE 2 — FastAPI Backend + MCP Real-time Market Data

## What Phase 2 builds

Phase 2 adds:
1. A production REST API (FastAPI) as the front door for all requests
2. Live stock prices, ETF data, and forex rates via Alpha Vantage
3. Latest financial news via NewsAPI
4. Market tools connected directly into the LangGraph agent

---

## File 7 — `backend/main.py`

### What is it?
The entry point of the FastAPI application. All routers are registered here.

### How it works

```python
app = FastAPI(title="Financial Chatbot API")

# Allow frontend to talk to backend
app.add_middleware(CORSMiddleware, allow_origins=["*"])

# Register all routers
app.include_router(chat.router,   prefix="/chat")
app.include_router(market.router, prefix="/market")

# Health check
@app.get("/health")
def health_check():
    return {"status": "ok"}
```

### How to run

```bash
uvicorn backend.main:app --reload
```

Then visit:
- `http://localhost:8000` — API running confirmation
- `http://localhost:8000/docs` — Interactive Swagger UI to test all endpoints
- `http://localhost:8000/health` — Health check

---

## File 8 — `backend/routers/chat.py`

### What is it?
Handles the `/chat` endpoint. Receives user messages and passes them through LangGraph.

### How it works

```
POST /chat/
    ↓
Receive: { message, country, user_profile }
    ↓
Pass to LangGraph financial_graph.invoke()
    ↓
LangGraph runs all 4 nodes
    ↓
Return: { response, country, intent, tax_disclaimer }
```

### Request format
```json
{
  "message": "I live in Germany, how much tax as a freelancer earning 60000?",
  "country": null,
  "user_profile": {}
}
```

### Response format
```json
{
  "response": "Your estimated tax is EUR 11,075.42...",
  "country": "germany",
  "intent": "tax",
  "tax_disclaimer": true
}
```

---

## File 9 — `backend/tools/market_data.py`

### What is it?
Four LangChain tools that fetch live financial data from Alpha Vantage and NewsAPI.

### The 4 tools

**Tool 1 — `get_stock_price`**
```
Input:  symbol (e.g. "AAPL", "SAP", "MSFT")
Output: current price, change, volume, high, low
Source: Alpha Vantage GLOBAL_QUOTE endpoint
```

**Tool 2 — `get_forex_rate`**
```
Input:  from_currency, to_currency (e.g. "EUR", "USD")
Output: live exchange rate + last update time
Source: Alpha Vantage CURRENCY_EXCHANGE_RATE endpoint
```

**Tool 3 — `get_etf_price`**
```
Input:  symbol (e.g. "SPY", "QQQ", "VWRL")
Output: current ETF price, change, high, low
Source: Alpha Vantage GLOBAL_QUOTE endpoint
```

**Tool 4 — `get_financial_news`**
```
Input:  country (e.g. "germany", "usa", "india", "australia")
Output: 5 latest financial news headlines with sources and links
Source: NewsAPI everything endpoint
```

### How tools connect to the agent

```python
# In nodes.py
llm_with_tools = llm.bind_tools([
    get_stock_price,
    get_forex_rate,
    get_etf_price,
    get_financial_news
])
```

`bind_tools()` tells the LLM what tools are available. The LLM then decides when to call them based on the user's question and the system prompt instructions.

### Important lesson learned

Having tools bound is not enough. You must tell the LLM to use them:

```python
# In system prompt — CRITICAL
"For stock prices, ETF prices, and forex rates — ALWAYS use the available tools"
"Never say you do not have real-time data — you have tools for live market data"
```

Without these instructions the LLM says "I don't have real-time data" even though tools are available.

---

## File 10 — `backend/routers/market.py`

### What is it?
Handles all `/market` endpoints. Exposes market tools as REST API endpoints.

### Endpoints

```
POST /market/stock   → get live stock price
POST /market/forex   → get live exchange rate
POST /market/etf     → get live ETF price
POST /market/news    → get latest financial news
GET  /market/health  → health check
```

### Example — POST /market/forex
```json
Request:
{
  "from_currency": "EUR",
  "to_currency": "USD"
}

Response:
{
  "data": "Exchange Rate: EUR → USD\n1 EUR = 1.1357 USD\nLast updated: 2026-06-25"
}
```

---

## How Phase 2 changed the architecture

**Before Phase 2:**
```
User → LangGraph → OpenAI → Response
```

**After Phase 2:**
```
User → FastAPI → LangGraph → OpenAI + Alpha Vantage + NewsAPI → Response
```

FastAPI is now the front door. Every request goes through it. This makes the system:
- Testable via Swagger UI
- Callable from any frontend (Gradio, React, mobile app)
- Ready for deployment to cloud

---

## Phase 2 Test Results

```
Test 1 — Germany freelancer tax (60,000 EUR)
  Country: germany | Intent: tax
  ✅ Tax: EUR 11,075.42 | Take home: EUR 48,924.58 | Disclaimer added

Test 2 — Apple stock price
  Country: unknown | Intent: market
  ✅ Tool called: get_stock_price("AAPL")
  ✅ Live price: $293.08 | Change: -$1.22 (-0.41%) | Volume: 53M

Test 3 — EUR to USD exchange rate
  Country: usa | Intent: market
  ✅ Tool called: get_forex_rate("EUR", "USD")
  ✅ Live rate: 1 EUR = 1.1357 USD

Test 4 — India ETF vs PPF investment advice
  Country: india | Intent: investment
  ✅ Detailed comparison with tax implications (Section 80C, LTCG)

Test 5 — USA tax bracket ($80,000)
  Country: usa | Intent: tax
  ✅ Correct brackets: 10% / 12% / 22% | Disclaimer added

Test 6 — Australia superannuation
  Country: australia | Intent: investment
  ✅ Full explanation with contribution rates, tax advantages, access rules
```

---

## Complete API endpoints available after Phase 2

```
GET  /              → API running confirmation
GET  /health        → health check
GET  /docs          → Swagger interactive UI

POST /chat/         → main chat endpoint
GET  /chat/health   → chat router health

POST /market/stock  → live stock price
POST /market/forex  → live exchange rate
POST /market/etf    → live ETF price
POST /market/news   → latest financial news
GET  /market/health → market router health
```

---

## Environment variables used in Phase 1 and 2

```
# Required — Phase 1
OPENAI_API_KEY         → for GPT-4o-mini responses
PINECONE_API_KEY       → for vector search (RAG)
PINECONE_ENVIRONMENT   → Pinecone region
LANGCHAIN_API_KEY      → for LangSmith tracing

# Required — Phase 2
ALPHA_VANTAGE_API_KEY  → for live stock/ETF/forex data
NEWS_API_KEY           → for latest financial news headlines

# Optional — Phase 3
SUPABASE_URL           → for user profile database
SUPABASE_KEY           → for Supabase authentication
```

---

## Key concepts learned in Phase 1 and 2

### 1. LangGraph State
Every node reads from and writes to a shared state object. This is how nodes communicate without calling each other directly.

### 2. Intent Classification
Before generating a response, classify what the user wants. This lets you apply different logic for different types of questions — tax vs market vs education.

### 3. Tool Calling
LLMs can call external functions (tools) to fetch live data. Two things are required:
- `bind_tools()` — gives LLM access to tools
- System prompt instruction — tells LLM WHEN to use them

### 4. Conditional Edges
LangGraph lets you route to different nodes based on logic. This is how we will add tax calculators and document analysis in Phase 3.

### 5. FastAPI as the gateway
FastAPI sits in front of everything. The frontend talks to FastAPI. FastAPI talks to LangGraph. LangGraph talks to OpenAI and external APIs. This separation makes the system testable, scalable, and deployable.

### 6. Model routing
Use the right model for the right task:
- `gpt-4o-mini` → fast and cheap for chat (Phase 1 and 2)
- `gpt-4o` → powerful for long document analysis (Phase 3)

---

## What comes next — Phase 3

Phase 3 adds:
- Tax calculators: Ehegattensplitting, freelancer, employee, single parent
- Salary slip upload and analysis using GPT-4o
- Bank statement upload and spending breakdown
- Steuerberater preparation checklist
- Supabase user profiles — remember users across sessions
- Savings goal tracking
- Official tax document PDFs loaded into Pinecone

