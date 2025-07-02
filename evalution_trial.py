import logging
from typing import Dict, Any, List
from langchain import hub
from langchain_openai import ChatOpenAI
# Using aevaluate for asynchronous evaluation
from langsmith.evaluation import aevaluate 
from datetime import datetime
from langchain_core.documents import Document # Import Document type for type hinting
from dotenv import load_dotenv
import os
import asyncio # Import asyncio for running async functions

# --- Load environment variables from .env file ---
load_dotenv()

# --- Configure logging for this evaluation script ---
# Set logging level to INFO for debugging purposes to see more detailed messages.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Import your agent_executor and retriever from chatbot_backend.py
# --- CORRECTED THIS LINE: Changed filename to chatbot_backend ---
try:
    from chatbot_backend import AGENT_EXECUTOR, RETRIEVER 
    logging.info("Successfully imported AGENT_EXECUTOR and RETRIEVER from chatbot_backend.py")
except ImportError as e:
    logging.error(f"Failed to import components from chatbot_backend.py: {e}")
    logging.error("Please ensure 'chatbot_backend.py' is in the correct directory and executable.")
    exit("Exiting: Could not load core chatbot components. Check chatbot_backend.py and its dependencies.")

# --- Define get_relevant_documents using the imported retriever ---
# This function is crucial for your predict_agent_run_for_eval to provide contexts
async def get_relevant_documents(query: str) -> List[Document]: # Made async
    """
    Retrieves relevant documents using the initialized retriever from chatbot_backend.py.
    """
    if RETRIEVER is None: # Corrected to uppercase RETRIEVER
        raise ValueError("Retriever is None. Ensure chatbot_backend.py initializes it correctly.")
    return await RETRIEVER.ainvoke(query) # Corrected to uppercase RETRIEVER and ainvoke


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
        # Note: Evaluators themselves are synchronous, they receive the already-run 'run' object
        score_response = answer_grader.invoke({
            "question": input_question,
            "correct_answer": correct_answer,
            "student_answer": prediction
        })
        
        # Robustly parse the score from the LLM's response
        # This version expects the score directly from a dictionary output
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
        
        # This version expects the score directly from a dictionary output
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
        
        # This version expects the score directly from a dictionary output
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


# --- Prediction Function for LangSmith Evaluation (Now Async) ---
async def predict_agent_run_for_eval(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Uses the AGENT_EXECUTOR to answer a question for evaluation purposes.
    Returns the answer under the 'answer' key and retrieved documents under 'contexts' key.
    """
    question = example.get("question")
    if not question:
        logging.error("Cannot evaluate: 'question' key missing from example inputs.")
        return {"answer": "Error: Question missing.", "contexts": []}

    logging.info(f"Evaluating question with AGENT_EXECUTOR: {question}")
    
    retrieved_docs_strings = [] # Initialize as empty list
    try:
        # Use the get_relevant_documents function defined above, which uses the imported retriever
        retrieved_docs_langchain = await get_relevant_documents(question) # Await the async function
        # Convert LangChain Document objects to strings (or whatever format your evaluators expect)
        # The evaluators expect a list of strings for 'documents'
        retrieved_docs_strings = [doc.page_content for doc in retrieved_docs_langchain]
        print(f"DEBUG: Retrieved documents (first 500 chars): {str(retrieved_docs_strings)[:500]}")
    except Exception as e:
        logging.warning(f"Failed to explicitly retrieve documents for evaluation context for question '{question}': {e}")
        # retrieved_docs_strings remains empty list if there's an error

    try:
        # Invoke the agent executor to get the answer (now using ainvoke for async)
        response = await AGENT_EXECUTOR.ainvoke({"input": question}) # Corrected to uppercase AGENT_EXECUTOR and ainvoke
        final_answer = response.get("output", "No answer generated by agent.")

        # Return both 'answer' and 'contexts' as expected by the evaluators
        return {"answer": final_answer, "contexts": retrieved_docs_strings}

    except Exception as e:
        logging.error(f"Error during agent execution for evaluation for question '{question}': {e}", exc_info=True)
        # Ensure contexts is always returned even on error
        return {"answer": f"Error during agent execution: {e}", "contexts": retrieved_docs_strings}


# --- Run the Evaluation with ALL Evaluators ---
# IMPORTANT: Replace "YOUR_GENERATED_DATASET_NAME_HERE" with the actual name
# from your create_dataset.py output (e.g., "My_Financial_Literacy_QA_Dataset_20250702_203634_c6ddb0")
dataset_name = "My_Financial_Literacy_QA_Dataset" 

print(f"\n--- Starting LangSmith Evaluation for dataset: {dataset_name} ---")

async def main():
    experiment_results = await aevaluate( # Use aevaluate for async prediction function
        predict_agent_run_for_eval, # The function that runs your agent and prepares output for evaluators
        data=dataset_name,          # The name of your dataset in Langsmith
        evaluators=[
            answer_evaluator,           # Evaluates accuracy against reference answer
            answer_hallucination_evaluator, # Evaluates if answer is grounded in retrieved docs
            docs_relevance_evaluator    # Evaluates if retrieved docs are relevant to the question
        ],
        # Create a unique experiment name for better tracking in Langsmith
        experiment_prefix=f"Financial_Literacy_Chatbot_Specific_QA_Eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        metadata={
            "variant": "Evaluation with 5 specific questions from Langsmith dataset",
            "source_dataset_name": dataset_name # Log the dataset name for reference
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
        print("Note: If the URL is not printed, you may need to upgrade your langsmith library (`pip install --upgrade langsmith`) or construct the URL manually using the experiment name and your Langsmith organization/project details.)")

if __name__ == "__main__":
    asyncio.run(main())
