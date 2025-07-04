# --- Module Docstring ---
"""
This script creates a financial literacy question-answer dataset in LangSmith.
It defines a set of specific Q&A pairs and uploads them to a new or existing
LangSmith dataset for use in evaluating chatbot performance.
"""

import logging # Standard library import
import os # Standard library import
from typing import List, Dict # Standard library import (Removed 'Any')

# Third-party imports
from dotenv import load_dotenv
from langsmith import Client
# Removed 'from langsmith.schemas import Example' as it's unused

# --- Load environment variables ---
load_dotenv() # Load .env file for API keys

# --- Configure logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Langsmith Client
client = Client()

logging.info("LANGCHAIN_API_KEY loaded: %s", bool(os.getenv('LANGCHAIN_API_KEY')))
logging.info("LANGCHAIN_TRACING_V2 loaded: %s", bool(os.getenv('LANGCHAIN_TRACING_V2')))

# Define the dataset name. Using a fixed name for simple dataset re-creation.
# For unique datasets, you could add a timestamp:
# DATASET_NAME = f"My_Financial_Literacy_QA_Dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
DATASET_NAME = "My_Financial_Literacy_QA_Dataset" # Renamed to UPPER_CASE

# Define your 5 specific question-answer pairs
qa_pairs: List[Dict[str, Dict[str, str]]] = [
    {
        "inputs": {
            "question": (
                "How does the speaker define 'debt,' and what is the key distinction "
                "made between 'good debt' and 'bad debt'?"
            )
        },
        "outputs": {
            "answer": (
                "Debt is defined as the amount of money that you owe, typically in "
                "the form of loans or credit card balances.\n\n"
                "The key distinction between 'good debt' and 'bad debt' is as "
                "follows:\n"
                "* **Good Debt**: This occurs when you borrow money to make an "
                "investment that is expected to generate more money than the amount "
                "borrowed, including covering the interest costs. Examples include "
                "borrowing to buy a house (which can reduce future rent expenses "
                "and may appreciate in value) or taking out a loan for a vehicle "
                "that can enhance employment opportunities.\n"
                "* **Bad Debt**: This happens when you borrow money for non-essential "
                "items, which essentially takes money away from your future self. "
                "Bad debt often arises from spending on wants rather than needs "
                "or investments.\n\nSource: MFCdA2vGVh4.txt"
            )
        }
    },
    {
        "inputs": {
            "question": (
                "What should be considered when deciding to use a credit card "
                "versus paying with cash?"
            )
        },
        "outputs": {
            "answer": (
                "When deciding to use a credit card versus paying with cash, several "
                "factors should be considered:\n\n"
                "1.  **Convenience and Acceptance**: Credit cards offer convenience, "
                "eliminating the need to carry physical cash, and are accepted in "
                "more places.\n"
                "2.  **Security**: Credit cards generally offer more security against "
                "theft or fraud compared to carrying large amounts of cash.\n"
                "3.  **Tracking Transactions**: Credit cards provide an easy way to "
                "track your spending through statements and online banking, which can "
                "be beneficial for budgeting.\n"
                "4.  **Rewards and Benefits**: Many credit cards offer rewards programs, "
                "cashback, or travel points that you can earn on purchases.\n"
                "5.  **Interest Rates**: Credit cards often come with high Annual "
                "Percentage Rates (APRs), meaning you'll incur significant interest "
                "if you don't pay your balance in full each month. Cash transactions "
                "do not have interest charges.\n"
                "6.  **Debt Accumulation**: Using credit cards can lead to accumulating "
                "debt if not managed responsibly, especially if you carry a balance. "
                "Cash prevents debt accumulation.\n"
                "7.  **Investment Potential (with caution)**: While credit can be used "
                "for investments, it's crucial to distinguish between using it for "
                "assets that generate returns (like a house) versus non-essential "
                "consumption, which can lead to \"bad debt.\"\n\n"
                "Ultimately, the choice depends on your financial discipline, "
                "spending habits, and whether you can consistently pay off your "
                "credit card balance to avoid interest charges.\n\n"
                "Source: MqqXTrEEZ7Y.txt, szUhmDH98oQ.txt"
            )
        }
    },
    {
        "inputs": {
            "question": (
                "What are the two broad categories of loans discussed, and what is "
                "the fundamental difference in their structure?"
            )
        },
        "outputs": {
            "answer": (
                "The two broad categories of loans discussed are **secured loans** "
                "and **unsecured loans**.\n\n"
                "The fundamental difference in their structure lies in the presence "
                "of collateral:\n"
                "* **Secured Loans**: These loans are backed by collateral, meaning "
                "the borrower offers an asset (like a house or car) as security for "
                "the loan. If the borrower fails to repay, the lender can claim "
                "the collateral. Because of this security, secured loans typically "
                "have lower interest rates.\n"
                "* **Unsecured Loans**: These loans are not backed by any collateral. "
                "The lender relies solely on the borrower's creditworthiness and "
                "promise to repay. Because they are riskier for lenders, unsecured "
                "loans often come with higher interest rates.\n\n"
                "Source: 8DF1f2jB5cY.txt"
            )
        }
    },
    {
        "inputs": {
            "question": (
                "What is credit, and what are some of its potential benefits or "
                "convenient uses mentioned in the transcript?"
            )
        },
        "outputs": {
            "answer": (
                "Credit is defined as the ability to borrow money or access goods "
                "and services with the understanding that payment will be made in "
                "the future.\n\n"
                "Some potential benefits or convenient uses of credit mentioned "
                "include:\n"
                "1.  **Building Credit History**: Using credit responsibly can help "
                "establish a positive credit history, which is essential for "
                "obtaining loans in the future.\n"
                "2.  **Emergency Expenses**: Credit provides a financial safety net "
                "for unexpected expenses, allowing individuals to manage unforeseen "
                "financial challenges without immediate cash.\n"
                "3.  **Convenience**: Credit cards offer a convenient way to make "
                "purchases without the need to carry cash.\n"
                "4.  **Rewards and Benefits**: Many credit cards offer rewards programs, "
                "cashback, or travel benefits, providing additional value "
                "for purchases made.\n"
                "5.  **Larger Purchases**: Credit allows individuals to make larger "
                "purchases and pay for them over time, making it easier to afford "
                "essential items or services.\n\n"
                "Source: Y7X4D7v7U1X.txt"
            )
        }
    },
    {
        "inputs": {
            "question": (
                "Explain what APR (Annual Percentage Rate) is in the context of "
                "credit, how it's typically calculated for credit cards, and why "
                "it's important for consumers."
            )
        },
        "outputs": {
            "answer": (
                "APR (Annual Percentage Rate) is the annualized interest rate that "
                "lenders charge borrowers for the use of credit. It reflects the cost "
                "of borrowing and is expressed as a percentage of the total loan "
                "amount over a year.\n\n"
                "**Calculation for Credit Cards**:\n"
                "APR is typically calculated based on the interest charged on the "
                "outstanding balance of the credit card. The APR is divided by the "
                "number of days in a year (usually 365) to get a daily periodic rate. "
                "This daily rate is then applied to the outstanding balance each day, "
                "accumulating interest. For example, if a credit card has an APR of "
                "18%, the daily rate would be approximately 0.04932% (18% ÷ 365).\n\n"
                "**Importance for Consumers**:\n"
                "1.  **Cost Awareness**: Understanding the APR helps consumers gauge "
                "how much borrowing will cost over time, enabling informed financial "
                "decisions.\n"
                "2.  **Comparison Shopping**: Consumers can compare different credit "
                "products and choose those with lower APRs, potentially saving money "
                "on interest payments.\n"
                "3.  **Managing Debt**: Knowing the APR is crucial for managing credit "
                "card debt effectively, as it impacts how quickly debt can accumulate "
                "if payments are not made in full.\n\n"
                "Source: bxRrjQj8sM1D.txt"
            )
        }
    }
]

try:
    dataset = client.read_dataset(dataset_name=DATASET_NAME)
    logging.info("Dataset '%s' already exists with ID: %s", DATASET_NAME, dataset.id)
except Exception as err: # pylint: disable=W0718 # Broad exception for Langsmith API call
    logging.info("Dataset '%s' not found. Creating a new one... (Error: %s)", DATASET_NAME, err)

    # Create the dataset
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="A dataset for evaluating a financial literacy chatbot with "
                    "questions and ideal answers.",
    )
    logging.info("Dataset '%s' created with ID: %s", DATASET_NAME, dataset.id)

    # Add examples to the dataset
    for qa_pair in qa_pairs: # Renamed 'qa' to 'qa_pair' for clarity
        client.create_example(
            inputs=qa_pair["inputs"],
            outputs=qa_pair["outputs"],
            dataset_id=dataset.id
        )
    logging.info("Added %d examples to dataset '%s'.", len(qa_pairs), DATASET_NAME)