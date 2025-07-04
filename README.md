# 💬 Financial Literacy Chatbot

The **Financial Literacy Chatbot** transforms overwhelming financial YouTube content into smart, personalized, and conversational insights.

Designed to enhance **financial literacy**, it offers **goal-based, region-aware guidance** in:

- 💰 Budgeting  
- 📊 Credit Score  
- 🧓 Retirement Planning  
- 📈 Investments  
- 💵 Saving  

Users can interact through voice, upload salary slips, and receive tailored advice — all via a **Gradio web interface**.

---

## 🧠 Data Ingestion & Knowledge Base

Financial content is sourced from YouTube videos and podcasts, then cleaned and chunked for processing.

- **Embeddings**: `text-embedding-ada-002`  
- **Vector Database**: [Pinecone](https://www.pinecone.io/)  
  - Dimensions: `1536`  
  - Similarity: `cosine`  
  - Batch size: `100`  
- Enables **semantic search** for accurate, context-aware answers

---

## 🧩 Backend: LangChain Agent + Tools

Built using **LangChain** with `gpt-4o-mini` for reasoning and tool orchestration.

### 🔧 Integrated Tools

| Tool | Function |
|------|----------|
| `KnowledgeBaseQuery` | Retrieves answers from Pinecone |
| `SavingsRecommendation` | Custom saving plans by region & profile |
| `BudgetingTemplates` | Budgeting methods (50/30/20, zero-based) |
| `CreditScoreAdvice` | Personalized credit improvement |
| `InvestmentAdvice` | Region-specific investment advice |
| `RetirementPlanning` | Retirement goals & strategies |
| `CompoundInterestCalculator` | Interest growth over time |
| `SalarySlipAnalysis` | Extracts financial data from salary slips |
| `DebtRepaymentStrategies` | Explains Snowball, Avalanche, etc. |

---

## 🖥 Deployment: Gradio Web Interface

The chatbot is accessible via a **Gradio interface** with:

- ✍️ Text and 🎙️ Voice interaction (STT + TTS)  
- 📄 File upload support (PDF/Text salary slips)  
- 💬 Chat history and contextual memory  

---

## 📊 Evaluation: LangSmith QA Metrics

Evaluated using **LangSmith** with a custom QA dataset:

| Metric | Description |
|--------|-------------|
| ✅ Answer Accuracy | Match between response and reference |
| ❌ Hallucination Rate | Detects unsupported/generated content |
| 📄 Document Relevance | Measures context relevance |

- **Dataset**: `My_Financial_Literacy_QA_Dataset`  
- **Script**: `evalution_trial.py`
## 📁 Project Structure

```
├── chatbot_backend.py         # Core logic, agent, tool registration
├── frontend_gradio_app.py     # Gradio frontend (chat + voice + file upload)
├── prepare_chunks.py          # Clean & split YouTube transcripts
├── pinecone_loader.py         # Generate embeddings & upload to Pinecone
├── evalution_trial.py         # LangSmith QA evaluation
├── .env                       # API keys and config (not committed)
├── requirements.txt           # Python dependencies
```
## ⚙️ Setup & Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd financial-literacy-chatbot

2. Create Virtual Environment

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

3. Install Dependencies

pip install -r requirements.txt

4. Add Environment Variables

Create a .env file in the root directory:

OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_env
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=financial-literacy-chatbot

5. Prepare Knowledge Base

    💡 Ensure the Pinecone index financial-literacy-chatbot exists before running the backend.

🧪 Run Evaluation

To test chatbot quality using LangSmith:

python evalution_trial.py


