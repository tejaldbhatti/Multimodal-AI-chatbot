# --- Langsmith Dataset Creation ---
from dotenv import load_dotenv
load_dotenv() # Load .env file for API keys

import os
import uuid # Add this import at the top of your file

from langsmith import Client
from langsmith.schemas import Example # Import Example schema for cleaner structure

# Initialize Langsmith Client
client = Client()

print(f"LANGCHAIN_API_KEY loaded: {os.getenv('LANGCHAIN_API_KEY')}")
print(f"LANGCHAIN_TRACING_V2 loaded: {os.getenv('LANGCHAIN_TRACING_V2')}")

# Define the dataset name (THIS WAS MISSING!)
dataset_name = "My_Financial_Literacy_QA_Dataset"

# Define your 17 question-answer pairs
# Replace this with your actual 17 question-answer pairs
# Each dictionary represents an 'example' for Langsmith.
# 'inputs' will contain the question given to the chatbot.
# 'outputs' will contain the ideal/reference answer for evaluation.
qa_pairs = [
    # {
    #     "inputs": {"question": "What is Torren's financial dilemma regarding saving for a new house, and what is his ideal plan for his currently owned property?"},
    #     "outputs": {"answer": "Torren plans to buy a new house in about three years, but this timeline might be flexible. His dilemma is whether to save the money for the down payment in a more conservative investment, like a high-yield savings account, or take a more aggressive approach by investing in the stock market. His ideal plan for his current property is to keep it and rent it out, rather than selling it."}
    # },
    # {
    #     "inputs": {"question": "What is the primary guiding question the financial planners advise Torren to consider when deciding how to invest his money for a future house down payment?"},
    #     "outputs": {"answer": "The primary guiding question advised is: \"What happens if you need it in three years and the market's down 20%?\" This forces Torren to consider the worst-case scenario and assess if he can absorb such a loss without fundamentally altering his plans, such as renting for longer, changing job plans, or being forced to sell his current house."}
    # },
    # {
    #     "inputs": {"question": "Explain the 'disappointment avoidance' approach suggested by the financial planners for Torren's investment decision, contrasting it with focusing on 'optimal' outcomes."},
    #     "outputs": {"answer": "The financial planners advocate for a 'disappointment avoidance' approach rather than solely seeking the 'optimal' investment outcome. This means focusing on avoiding the worst-case scenario that would cause the most significant negative impact or force undesired changes to one's plans. For instance, if a 20% market downturn would prevent Torren from buying a house or force him to sell his current property, then that scenario must be avoided by choosing a more conservative investment. This prioritizes mitigating unacceptable risks over maximizing potential returns."}
    # },
    # {
    #     "inputs": {"question": "Based on the advice given, what specific investment strategy is recommended if Torren *absolutely needs* the money by the three-year mark versus if his timeline is genuinely flexible and he can absorb potential market downturns?"},
    #     "outputs": {"answer": "If Torren absolutely needs the money by the three-year mark and cannot absorb a significant market downturn (e.g., down 20%), the recommended strategy is to put the money into **cash or something like cash**, such as a high-yield savings account or a short-term CD. However, if his timeline is truly flexible, and he is okay with renting for an extra year or two if the market is down, or if he can absorb a reduced down payment, then the financial planners are \"much more open to investing it\" in the stock market (a more aggressive approach). The decision hinges entirely on his ability to absorb the worst-case scenario."}
    # },
    # {
    #     "inputs": {"question": "Why do the financial planners advise against moving from high-yield savings to bonds for a 5-6 year horizon in this scenario, especially concerning the yield curve and potential return?"},
    #     "outputs": {"answer": "The financial planners advise against moving from a high-yield savings account to bonds for a 5-6 year horizon because the **yield curve** does not significantly benefit longer-term lending compared to shorter-term. They state that the difference in interest rates between a two-year CD and a five-year CD is not as substantial as, for example, a six-month to two-year CD. Specifically, a six-year bond might pay a similar rate to a two-year CD, meaning you lock up your money for much longer without gaining a significant increase in return to justify the added risk of a fixed-income product compared to a guaranteed product like a CD or high-yield savings account. The incremental return might not be \"life-altering.\""}
    # },
    # {
    #     "inputs": {"question": "What are the two broad categories of loans discussed, and what is the fundamental difference in their structure?"},
    #     "outputs": {"answer": "The two broad categories of loans are **installment loans** (or installment credit) and **revolving loans** (or revolving credit). The fundamental difference is that installment loans involve borrowing a single, usually large, amount of money that is then paid back in fixed installments over a set period. In contrast, revolving loans provide a credit limit that you can borrow up to, pay down, and then borrow again repeatedly as needed, essentially 'revolving' the credit."}
    # },
    # {
    #     "inputs": {"question": "Provide common examples for both installment loans and revolving credit, and briefly describe their typical repayment processes."},
    #     "outputs": {"answer": "Common examples of **installment loans** include car loans, student debt, and mortgages. For these, a large sum is borrowed upfront, and repayment occurs through regular, typically fixed, payments over a set duration (e.g., 10, 15, or 30 years), covering both the principal and interest. The most common example of **revolving credit** is a credit card, but personal lines of credit from banks also apply. With revolving credit, you are given a credit limit; you can use funds up to this limit, pay off the balance, and then reuse the available credit. This cycle of borrowing and repaying is continuous, or 'revolving.'"}
    # },
    # {
    #     "inputs": {"question": "Why are credit cards classified as a type of revolving credit, and how does their usage exemplify the characteristics of this loan category?"},
    #     "outputs": {"answer": "Credit cards are classified as revolving credit because they exemplify the continuous cycle of borrowing and repaying up to a set limit. A credit card issuer grants a specific credit limit (e.g., $1,000). The cardholder can use funds up to this limit, and as they pay down the outstanding balance, the available credit 'revolves' or becomes available again for future use. This means you can continually use some of the available credit, pay it back, and then use it again, without needing to apply for a new loan each time you need funds."}
    # },
    # {
    #     "inputs": {"question": "What is credit, and what are some of its potential benefits or convenient uses mentioned in the transcript?"},
    #     "outputs": {"answer": "Credit is defined as the ability to borrow money from someone else. Its potential benefits or convenient uses include: \n* **Convenience:** With credit cards, you don't have to carry cash.\n* **Acceptance:** Some places only accept credit or credit cards for payments.\n* **Facilitating good investments:** You can borrow money to make investments that are expected to yield a higher return than the cost of borrowing."}
    # },
    {
        "inputs": {"question": "When should individuals be particularly cautious about using credit, and what are the warning signs or negative implications discussed?"},
        "outputs": {"answer": "Individuals should be very cautious about using credit, especially when it's for **consumption** rather than investment, particularly if they don't actually have the money to pay for the purchase. This is a significant warning sign that one is spending more money than they are earning, using credit to mask this unsustainable spending. The negative implications include unsustainability and incurring significant costs due to interest and fees charged on the borrowed amount."}
    },
    {
        "inputs": {"question": "Explain what APR (Annual Percentage Rate) is in the context of credit, how it's typically calculated for credit cards, and why it's important for consumers."},
        "outputs": {"answer": "APR stands for **Annual Percentage Rate**. It represents the annual cost of borrowing money, including both interest and certain fees, expressed as a percentage. For credit cards, the APR is calculated by looking at your average daily balance and applying a certain amount of interest and fees to it, then standardizing that to an annual rate (multiplying by 365). Although it slightly understates the true cost due to not compounding daily, it serves as a crucial standardized measure for consumers to understand the approximate cost of a loan or credit, making it easier to compare different credit products."}
    },
    {
        "inputs": {"question": "How do the APRs for credit cards typically compare to those for mortgages, and what reason is given for this difference?"},
        "outputs": {"answer": "APRs for credit cards are typically much higher, often in the high teens, twenties, or even thirties, compared to mortgages. Mortgages will have significantly lower APRs. The reason given for this difference is that lenders view mortgages as a safer bet from their point of view. This lower perceived risk associated with mortgages translates into lower borrowing costs for the consumer."}
},
    # {
    #     "inputs": {"question": "How does the speaker define 'debt,' and what is the key distinction made between 'good debt' and 'bad debt'?"},
    #     "outputs": {"answer": "Debt is defined as the amount of money that one owes, typically in the form of loans or a credit card balance. The key distinction made is that **good debt** is incurred when borrowing money to make an investment that is expected to produce more money than the amount borrowed, enough to offset the interest payments. Conversely, **bad debt** is when money is borrowed for things that are not needed, essentially taking money away from one's future self."}
    # },
    # {
    #     "inputs": {"question": "What examples of 'good debt' does the speaker provide, and what crucial advice or warnings are given for each of these examples?"},
    #     "outputs": {"answer": "The speaker provides three main examples of 'good debt':\n1.  **Buying a house:** This can reduce future rent expenses and potentially appreciate in value. *Caveat:* It's not always the right decision; one must weigh expected rent savings and potential appreciation against the risk of depreciation and ensure ability to repay, even if unforeseen events occur (e.g., job loss, emergency).\n2.  **Buying a car/motorcycle for transportation to a job:** This can help get to a current job or a better job, or allow for more work hours. *Caveat:* Be careful not to buy a car that is \"fancy\" for showing off, as its fanciness won't necessarily improve job access or income. Debt for wants, not needs, becomes bad debt.\n3.  **Investing in oneself through education (course/degree):** This can increase job prospects and income. *Caveat:* It's crucial to talk to people who have taken on debt for similar programs/careers to ensure it truly paid off for them."}
    # },
#     {
#         "inputs": {"question": "According to the speaker, what defines 'bad debt,' and what is the fundamental negative consequence of incurring this type of debt?"},
#         "outputs": {"answer": "'Bad debt' is defined as borrowing money for things that are not truly needed. The fundamental negative consequence of incurring bad debt is that it essentially involves \"taking money away from your future self,\" as you are committing future earnings to pay for unnecessary past consumption, potentially hindering future financial well-being."}
#     },
#     {
#     "inputs": {"question": "What should be considered when deciding to use a credit card versus paying with cash?"},
#     "outputs": {"answer": "Factors include whether the merchant accepts credit, whether you're earning rewards or points, and if you can pay off the balance in full. If you might carry a balance and incur high interest, cash or debit may be better to avoid debt."}
#    },
#    {
#     "inputs": {"question": "How can breaking a large SMART goal into smaller goals help in financial planning?"},
#     "outputs": {"answer": "It makes the larger goal more manageable and trackable. For instance, saving $100,000 in 5 years becomes a goal of saving $20,000 per year, which can further be divided into monthly savings targets. This increases clarity and motivation."}
#    }
]

try:
    dataset = client.read_dataset(dataset_name=dataset_name)
    print(f"Dataset '{dataset_name}' already exists with ID: {dataset.id}")
except Exception:
    print(f"Dataset '{dataset_name}' not found. Creating a new one...")

    # Create the dataset
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="A dataset for evaluating a financial literacy chatbot with questions and ideal answers.",
    )
    print(f"Dataset '{dataset_name}' created with ID: {dataset.id}")

    # Add examples to the dataset
    for qa in qa_pairs:
        client.create_example(
            inputs=qa["inputs"],
            outputs=qa["outputs"],
            dataset_id=dataset.id
        )
    print(f"Added {len(qa_pairs)} examples to dataset '{dataset_name}'.")
