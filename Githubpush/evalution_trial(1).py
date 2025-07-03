import logging
import asyncio # Standard library import
from datetime import datetime # Standard library import
from typing import Dict, Any, List # Standard library import

# Third-party imports
from dotenv import load_dotenv
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document # Import Document type for type hinting
from langsmith.evaluation import aevaluate # Using aevaluate for asynchronous evaluation

# --- Module Docstring ---
"""
This script provides an evaluation framework for a financial literacy chatbot
using LangSmith. It defines custom evaluators for answer accuracy, hallucination,
and document relevance, and integrates with a chatbot backend to run predictions
against a predefined dataset.
"""


# --- Load environment variables from .env file ---
load_dotenv()

# --- Configure logging for this evaluation script ---
# Set logging level to INFO for debugging purposes to see more detailed messages.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Import your agent_executor and retriever from chatbot_backend.py
try:
    # pylint: disable=E0401 # import-error: Assuming chatbot_backend is in a reachable path.
    from chatbot_backend import AGENT_EXECUTOR, RETRIEVER
    logging.info("Successfully imported AGENT_EXECUTOR and RETRIEVER from chatbot_backend.py")
except ImportError as err:
    logging.error(
        "Failed to import components from chatbot_backend.py: %s", err, exc_info=True
    )
    logging.error(
        "Please ensure 'chatbot_backend.py' is in the correct directory and "
        "executable."
    )
    # pylint: disable=R1722 # Pylint prefers sys.exit for exiting script
    exit("Exiting: Could not load core chatbot components. Check chatbot_backend.py "
         "and its dependencies.")


# --- Define get_relevant_documents using the imported retriever ---
# This function is crucial for your predict_agent_run_for_eval to provide contexts
async def get_relevant_documents(query: str) -> List[Document]:
    """
    Retrieves relevant documents using the initialized retriever from
    chatbot_backend.py.

    Args:
        query (str): The search query.

    Returns:
        List[Document]: A list of relevant LangChain Document objects.

    Raises:
        ValueError: If the RETRIEVER is not initialized.
    """
    if RETRIEVER is None:
        raise ValueError(
            "Retriever is None. Ensure chatbot_backend.py initializes it correctly."
        )
    logging.info("Retrieving documents for query: '%s'", query)
    return await RETRIEVER.ainvoke(query)


# --- Define Prompts for Hallucination and Document Relevance ---
# These prompts are pulled from LangChain Hub and used by the LLM-as-a-judge evaluators.
grade_prompt_hallucinations = hub.pull("langchain-ai/rag-answer-hallucination")
grade_prompt_doc_relevance = hub.pull("langchain-ai/rag-document-relevance")
grade_prompt_answer = hub.pull("langchain-ai/rag-answer-vs-reference")


# --- Custom Evaluator Function for Answer Accuracy (Type 1) ---
def answer_evaluator(run: Dict[str, Any], example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates the accuracy of the generated answer against a reference answer.

    Args:
        run (Dict[str, Any]): The LangSmith run object containing predictions.
        example (Dict[str, Any]): The LangSmith example object containing reference data.

    Returns:
        Dict[str, Any]: A dictionary with 'key' as "answer_accuracy" and 'score'.
    """
    input_question = example.inputs.get("question", "")
    correct_answer = example.outputs.get("answer", "") # Assumes 'answer' key in dataset outputs
    prediction = run.outputs.get("answer", "") # Assumes 'answer' key in prediction function output

    llm = ChatOpenAI(model="gpt-4o", temperature=0) # LLM used as the grader
    answer_grader = grade_prompt_answer | llm

    score = 0.0 # Default score if parsing fails
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
                    logging.warning(
                        "Could not parse answer accuracy score '%s' as float or "
                        "boolean for question '%s'.",
                        raw_score, input_question
                    )
                    score = 0.0
        else:
            logging.warning(
                "Unexpected answer accuracy score type: %s with value %s for "
                "question '%s'.",
                type(raw_score), raw_score, input_question
            )
            score = 0.0

    except Exception as err: # pylint: disable=W0718 # Broad exception for external LLM call
        logging.error(
            "Error grading answer accuracy with LLM for question '%s': %s",
            input_question, err, exc_info=True
        )
        score = 0.0

    logging.info(
        "Answer Accuracy Evaluation: question='%s', score=%s", input_question, score
    )
    return {"key": "answer_accuracy", "score": score}


# --- Custom Answer Hallucination Evaluator Function (Type 2) ---
def answer_hallucination_evaluator(
    run: Dict[str, Any], example: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluates if the generated answer contains information not supported by the
    retrieved documents (hallucinations).

    Args:
        run (Dict[str, Any]): The LangSmith run object containing predictions and contexts.
        example (Dict[str, Any]): The LangSmith example object containing reference data.

    Returns:
        Dict[str, Any]: A dictionary with 'key' as "answer_hallucination" and 'score'.
    """
    input_question = example.inputs.get("question", "")
    contexts = run.outputs.get("contexts", []) # Assumes 'contexts' key in prediction function output
    prediction = run.outputs.get("answer", "") # Assumes 'answer' key in prediction function output

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    answer_grader = grade_prompt_hallucinations | llm

    score = 0.0
    try:
        score_response = answer_grader.invoke({
            "documents": contexts,
            "student_answer": prediction
        })

        # This version expects the score directly from a dictionary output
        raw_score = score_response.get("Score")

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
                    logging.warning(
                        "Could not parse hallucination score '%s' as float or "
                        "boolean for question '%s'.",
                        raw_score, input_question
                    )
                    score = 0.0
        else:
            logging.warning(
                "Unexpected hallucination score type: %s with value %s for "
                "question '%s'.",
                type(raw_score), raw_score, input_question
            )
            score = 0.0

    except Exception as err: # pylint: disable=W0718 # Broad exception for external LLM call
        logging.error(
            "Error grading hallucination with LLM for question '%s': %s",
            input_question, err, exc_info=True
        )
        score = 0.0

    logging.info(
        "Hallucination Evaluation: question='%s', hallucination_score=%s",
        input_question, score
    )
    return {"key": "answer_hallucination", "score": score}

# --- Custom Document Relevance Evaluator Function (Type 3) ---
def docs_relevance_evaluator(run: Dict[str, Any], example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates if the retrieved documents are relevant to the input question.

    Args:
        run (Dict[str, Any]): The LangSmith run object containing retrieved documents.
        example (Dict[str, Any]): The LangSmith example object containing reference data.

    Returns:
        Dict[str, Any]: A dictionary with 'key' as "document_relevance" and 'score'.
    """
    input_question = example.inputs.get("question", "")
    contexts = run.outputs.get("contexts", []) # Assumes 'contexts' key in prediction function output

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    answer_grader = grade_prompt_doc_relevance | llm

    score = 0.0
    try:
        score_response = answer_grader.invoke({
            "question": input_question,
            "documents": contexts
        })

        # This version expects the score directly from a dictionary output
        raw_score = score_response.get("Score")

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
                    logging.warning(
                        "Could not parse document relevance score '%s' as float or "
                        "boolean for question '%s'.",
                        raw_score, input_question
                    )
                    score = 0.0
        else:
            logging.warning(
                "Unexpected document relevance score type: %s with value %s for "
                "question '%s'.",
                type(raw_score), raw_score, input_question
            )
            score = 0.0

    except Exception as err: # pylint: disable=W0718 # Broad exception for external LLM call
        logging.error(
            "Error grading document relevance with LLM for question '%s': %s",
            input_question, err, exc_info=True
        )
        score = 0.0

    logging.info(
        "Document Relevance Evaluation: question='%s', doc_relevance_score=%s",
        input_question, score
    )
    return {"key": "document_relevance", "score": score}


# --- Prediction Function for LangSmith Evaluation (Now Async) ---
async def predict_agent_run_for_eval(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Uses the AGENT_EXECUTOR to answer a question for evaluation purposes.
    Returns the answer under the 'answer' key and retrieved documents under 'contexts' key.

    Args:
        example (Dict[str, Any]): An example from the dataset, expected to have a 'question' key.

    Returns:
        Dict[str, Any]: A dictionary containing the 'answer' generated by the agent
                       and a list of 'contexts' (retrieved documents).
    """
    question = example.get("question")
    if not question:
        logging.error("Cannot evaluate: 'question' key missing from example inputs.")
        return {"answer": "Error: Question missing.", "contexts": []}

    logging.info("Evaluating question with AGENT_EXECUTOR: '%s'", question)

    retrieved_docs_strings: List[str] = [] # Initialize as empty list with explicit type
    try:
        # Use the get_relevant_documents function defined above, which uses the imported retriever
        retrieved_docs_langchain = await get_relevant_documents(question)
        # Convert LangChain Document objects to strings (or whatever format your evaluators expect)
        # The evaluators expect a list of strings for 'documents'
        retrieved_docs_strings = [
            doc.page_content for doc in retrieved_docs_langchain
        ]
        logging.debug(
            "Retrieved documents (first 500 chars): %s",
            str(retrieved_docs_strings)[:500]
        ) # Use debug for verbose output
    except ValueError as err:
        logging.warning(
            "Retriever initialization error for question '%s': %s", question, err
        )
    except Exception as err: # pylint: disable=W0718 # Broad exception for external retriever calls
        logging.warning(
            "Failed to explicitly retrieve documents for evaluation context for "
            "question '%s': %s",
            question, err, exc_info=True
        )
        # retrieved_docs_strings remains empty list if there's an error

    try:
        # Invoke the agent executor to get the answer
        response = await AGENT_EXECUTOR.ainvoke({"input": question})
        final_answer = response.get("output", "No answer generated by agent.")

        # Return both 'answer' and 'contexts' as expected by the evaluators
        return {"answer": final_answer, "contexts": retrieved_docs_strings}

    except Exception as err: # pylint: disable=W0718 # Broad exception for general agent execution errors
        logging.error(
            "Error during agent execution for evaluation for question '%s': %s",
            question, err, exc_info=True
        )
        # Ensure contexts is always returned even on error
        return {"answer": f"Error during agent execution: {err}", "contexts": retrieved_docs_strings}


# --- Run the Evaluation with ALL Evaluators ---
async def main():
    """
    Main asynchronous function to orchestrate the LangSmith evaluation run.
    """
    # IMPORTANT: Replace "YOUR_GENERATED_DATASET_NAME_HERE" with the actual name
    # from your create_dataset.py output (e.g., "My_Financial_Literacy_QA_Dataset_20250702_203634_c6ddb0")
    dataset_name = "My_Financial_Literacy_QA_Dataset" # Example name, user should update this

    logging.info("\n--- Starting LangSmith Evaluation for dataset: '%s' ---", dataset_name)

    experiment_results = await aevaluate( # Use aevaluate for async prediction function
        predict_agent_run_for_eval, # The function that runs your agent and prepares output for evaluators
        data=dataset_name,           # The name of your dataset in Langsmith
        evaluators=[
            answer_evaluator,               # Evaluates accuracy against reference answer
            answer_hallucination_evaluator, # Evaluates if answer is grounded in retrieved docs
            docs_relevance_evaluator        # Evaluates if retrieved docs are relevant to the question
        ],
        # Create a unique experiment name for better tracking in Langsmith
        experiment_prefix=(
            f"Financial_Literacy_Chatbot_Specific_QA_Eval_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ),
        metadata={
            "variant": "Evaluation with specific questions from Langsmith dataset",
            "source_dataset_name": dataset_name # Log the dataset name for reference
        },
    )

    logging.info("\n--- Evaluation complete! ---")
    logging.info("View results on LangSmith:")
    # Attempt to print the direct URL to the experiment results
    try:
        logging.info(experiment_results.url)
    except AttributeError:
        # Fallback if .url attribute is not available (e.g., older langsmith versions)
        logging.info("Experiment name: %s", experiment_results.experiment_name)
        logging.warning(
            "Note: If the URL is not printed, you may need to upgrade your "
            "langsmith library (`pip install --upgrade langsmith`) or construct the URL "
            "manually using the experiment name and your Langsmith organization/project "
            "details.)"
        )

if __name__ == "__main__":
    asyncio.run(main())