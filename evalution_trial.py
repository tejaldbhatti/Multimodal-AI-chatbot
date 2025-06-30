import logging
from typing import Dict, Any, List
from langchain import hub
from langchain_openai import ChatOpenAI
from langsmith.evaluation import evaluate
from datetime import datetime
from langchain_core.documents import Document # Import Document type for type hinting
from dotenv import load_dotenv
import os

# --- Load environment variables from .env file ---
load_dotenv()

# --- Configure logging for this evaluation script ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- IMPORTANT: Import your agent_executor and retriever from app.py ---
# This assumes 'app.py' is in the same directory as this script.
# If app.py is in a different location or part of a package, adjust the import path.
try:
    from app import agent_executor, retriever 
    logging.info("Successfully imported agent_executor and retriever from app.py")
except ImportError as e:
    logging.error(f"Failed to import components from app.py: {e}")
    logging.error("Please ensure 'app.py' is in the correct directory and executable.")
    exit("Exiting: Could not load core chatbot components. Check app.py and its dependencies.")


# --- Define get_relevant_documents using the imported retriever ---
# This function is crucial for your predict_agent_run_for_eval to provide contexts
def get_relevant_documents(query: str) -> List[Document]:
    """
    Retrieves relevant documents using the initialized retriever from app.py.
    """
    if retriever is None:
        raise ValueError("Retriever is None. Ensure app.py initializes it correctly.")
    return retriever.get_relevant_documents(query)


# --- Define Prompts for Hallucination and Document Relevance ---
# These prompts are pulled from LangChain Hub and used by the LLM-as-a-judge evaluators.
grade_prompt_hallucinations = hub.pull("langchain-ai/rag-answer-hallucination")
grade_prompt_doc_relevance = hub.pull("langchain-ai/rag-document-relevance")
grade_prompt_answer = hub.pull("langchain-ai/rag-answer-vs-reference")


# --- Custom Evaluator Function for Answer Accuracy (Type 1) ---
def answer_evaluator(run, example) -> dict:
    """
    Evaluates the accuracy of the generated answer against a reference answer.
    """
    input_question = example.inputs.get("question", "")
    correct_answer = example.outputs.get("answer", "") # Assumes 'answer' key in dataset outputs for reference
    prediction = run.outputs.get("answer", "") # Assumes 'answer' key in prediction function output

    llm = ChatOpenAI(model="gpt-4o", temperature=0) # LLM used as the grader
    answer_grader = grade_prompt_answer | llm

    try:
        score_response = answer_grader.invoke({
            "question": input_question,
            "correct_answer": correct_answer,
            "student_answer": prediction
        })
        
        # Robustly parse the score from the LLM's response
        raw_score = score_response.get("Score")
        score = 0.0 # Default score if parsing fails

        if isinstance(raw_score, (int, float)):
            score = float(raw_score)
        elif isinstance(raw_score, str):
            score_text = raw_score.strip().lower()
            if score_text in ["yes", "y", "true", "t"]:
                score = 1.0
            elif score_text in ["no", "n", "false", "f"]:
                score = 0.0
            else:
                try:
                    score = float(score_text)
                except ValueError:
                    logging.warning(f"Could not parse answer accuracy score '{raw_score}' as float or boolean for question '{input_question}'.")
                    score = 0.0
        else:
            logging.warning(f"Unexpected answer accuracy score type: {type(raw_score)} with value {raw_score} for question '{input_question}'.")
            score = 0.0

    except Exception as e:
        logging.error(f"Error grading answer accuracy with LLM for question '{input_question}': {e}", exc_info=True)
        score = 0.0

    print(f'question="{input_question}", score={score}')
    return {"key": "answer_accuracy", "score": score}


# --- Custom Answer Hallucination Evaluator Function (Type 2) ---
def answer_hallucination_evaluator(run, example) -> dict:
    """
    Evaluates if the generated answer contains information not supported by the retrieved documents (hallucinations).
    """
    input_question = example.inputs.get("question", "")
    contexts = run.outputs.get("contexts", []) # Assumes 'contexts' key in prediction function output
    prediction = run.outputs.get("answer", "") # Assumes 'answer' key in prediction function output

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    answer_grader = grade_prompt_hallucinations | llm

    try:
        score_response = answer_grader.invoke({
            "documents": contexts,
            "student_answer": prediction
        })
        
        raw_score = score_response.get("Score")
        score = 0.0

        if isinstance(raw_score, (int, float)):
            score = float(raw_score)
        elif isinstance(raw_score, str):
            score_text = raw_score.strip().lower()
            if score_text in ["yes", "y", "true", "t"]:
                score = 1.0
            elif score_text in ["no", "n", "false", "f"]:
                score = 0.0
            else:
                try:
                    score = float(score_text)
                except ValueError:
                    logging.warning(f"Could not parse hallucination score '{raw_score}' as float or boolean for question '{input_question}'.")
                    score = 0.0
        else:
            logging.warning(f"Unexpected hallucination score type: {type(raw_score)} with value {raw_score} for question '{input_question}'.")
            score = 0.0

    except Exception as e:
        logging.error(f"Error grading hallucination with LLM for question '{input_question}': {e}", exc_info=True)
        score = 0.0

    print(f'question="{input_question}", hallucination_score={score}')
    return {"key": "answer_hallucination", "score": score}

# --- Custom Document Relevance Evaluator Function (Type 3) ---
def docs_relevance_evaluator(run, example) -> dict:
    """
    Evaluates if the retrieved documents are relevant to the input question.
    """
    input_question = example.inputs.get("question", "")
    contexts = run.outputs.get("contexts", []) # Assumes 'contexts' key in prediction function output

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    answer_grader = grade_prompt_doc_relevance | llm

    try:
        score_response = answer_grader.invoke({
            "question": input_question,
            "documents": contexts
        })
        
        raw_score = score_response.get("Score")
        score = 0.0

        if isinstance(raw_score, (int, float)):
            score = float(raw_score)
        elif isinstance(raw_score, str):
            score_text = raw_score.strip().lower()
            if score_text in ["yes", "y", "true", "t"]:
                score = 1.0
            elif score_text in ["no", "n", "false", "f"]:
                score = 0.0
            else:
                try:
                    score = float(score_text)
                except ValueError:
                    logging.warning(f"Could not parse document relevance score '{raw_score}' as float or boolean for question '{input_question}'.")
                    score = 0.0
        else:
            logging.warning(f"Unexpected document relevance score type: {type(raw_score)} with value {raw_score} for question '{input_question}'.")
            score = 0.0

    except Exception as e:
        logging.error(f"Error grading document relevance with LLM for question '{input_question}': {e}", exc_info=True)
        score = 0.0

    print(f'question="{input_question}", doc_relevance_score={score}')
    return {"key": "document_relevance", "score": score}


# --- Prediction Function for LangSmith Evaluation ---
# This function wraps your agent_executor to produce output in the format
# expected by the custom evaluators (i.e., 'answer' and 'contexts').
def predict_agent_run_for_eval(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Uses the agent_executor to answer a question for evaluation purposes.
    Returns the answer under the 'answer' key and retrieved documents under 'contexts' key.
    """
    question = example.get("question")
    if not question:
        logging.error("Cannot evaluate: 'question' key missing from example inputs.")
        return {"answer": "Error: Question missing.", "contexts": []}

    logging.info(f"Evaluating question with agent_executor: {question}")
    
    retrieved_docs_strings = [] # Initialize as empty list
    try:
        # Use the get_relevant_documents function defined above, which uses the imported retriever
        retrieved_docs_langchain = get_relevant_documents(question)
        # Convert LangChain Document objects to strings (or whatever format your evaluators expect)
        # The evaluators expect a list of strings for 'documents'
        retrieved_docs_strings = [doc.page_content for doc in retrieved_docs_langchain]
        print(f"DEBUG: Retrieved documents (first 500 chars): {str(retrieved_docs_strings)[:500]}")
    except Exception as e:
        logging.warning(f"Failed to explicitly retrieve documents for evaluation context for question '{question}': {e}")
        # retrieved_docs_strings remains empty list if there's an error

    try:
        # Invoke the agent executor to get the answer
        response = agent_executor.invoke({"input": question})
        final_answer = response.get("output", "No answer generated by agent.")

        # Return both 'answer' and 'contexts' as expected by the evaluators
        return {"answer": final_answer, "contexts": retrieved_docs_strings}

    except Exception as e:
        logging.error(f"Error during agent execution for evaluation for question '{question}': {e}", exc_info=True)
        # Ensure contexts is always returned even on error
        return {"answer": f"Error during agent execution: {e}", "contexts": retrieved_docs_strings}


# --- Run the Evaluation with ALL Evaluators ---
# IMPORTANT: Ensure "My_Financial_Literacy_QA_Dataset" matches the exact name of your dataset in LangSmith
dataset_name = "My_Financial_Literacy_QA_Dataset" 

print(f"\n--- Starting LangSmith Evaluation for dataset: {dataset_name} ---")

experiment_results = evaluate(
    predict_agent_run_for_eval, # The function that runs your agent and prepares output for evaluators
    data=dataset_name,          # The name of your dataset in LangSmith
    evaluators=[
        answer_evaluator,           # Evaluates accuracy against reference answer
        answer_hallucination_evaluator, # Evaluates if answer is grounded in retrieved docs
        docs_relevance_evaluator    # Evaluates if retrieved docs are relevant to the question
    ],
    # Create a unique experiment name for better tracking in LangSmith
    experiment_prefix=f"Financial_Literacy_Chatbot_Full_RAG_Eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    metadata={
        "variant": "Full RAG Evaluation with all 3 types",
    },
)

print("\n--- Evaluation complete! ---")
print("View results on LangSmith:")
# Attempt to print the direct URL to the experiment results
try:
    print(experiment_results.url)
except AttributeError:
    # Fallback if .url attribute is not available (e.g., older langsmith versions)
    print(f"Experiment name: {experiment_results.experiment_name}")
    print("Note: If the URL is not printed, you may need to upgrade your langsmith library (`pip install --upgrade langsmith`) or construct the URL manually using the experiment name and your LangSmith organization/project details.")