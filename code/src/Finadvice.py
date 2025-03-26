import os
import pandas as pd
from dotenv import load_dotenv
from phi.agent import Agent,RunResponse
from phi.model.groq import Groq
from phi.playground import Playground, serve_playground_app
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo

# Load environment variables
load_dotenv()
phi_api_key = os.getenv("PHI_API_KEY")
if not phi_api_key:
    raise ValueError("PHI_API_KEY is missing. Check your .env file.")

# Load customer comments CSV
def load_customer_comments(csv_path="customer_advice.csv"):
    try:
        return pd.read_csv(csv_path,encoding="utf-8")
    except Exception as e:
        return f"Error loading CSV file: {str(e)}"

# Get customer comment by ID
def get_customer_comment(customer_id, csv_path="customer_advice.csv"):
    df = load_customer_comments(csv_path)
    if isinstance(df, str):  # If an error occurred, return it
        return df
    
    customer_row = df[df["customer_id"] == customer_id]
    if customer_row.empty:
        return f"No comment found for customer_id {customer_id}"
    
    return customer_row["customer_comment"].values[0]

# Web Search Agent
websearch_agent = Agent(
    name="Web Search Agent",
    description="You are a financial expert and you search the latest information from the web.",
    role="Search the web for financial information.",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=["Always include the source in your response."],
    show_tool_calls=True
)

# Finance Agent
finance_agent = Agent(
    name="Finance Agent",
    description="You are a financial planning expert. You provide financial plans based on customer input.",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True, company_news=True)],
    instructions=[
        "Greet the user with 'Welcome to Wells Fargo Financial Planner'.",
        "Extract important financial details from the user's comment.",
        "Provide personalized financial advice based on the user's needs.",
        "Use the Web Search Agent to check for the latest financial information.",
        "Summarize the advice concisely in a tabular format: name, age, salary, advice."
    ],
    expected_output="""\
    ## Your Personalized Financial Advice

    ### Personal Details
    {Extract name, age, salary, liabilities, etc.}
    {Tabulate the details}

    ### Financial Advice
    {Analyze the user's income, liabilities, assets, etc.}
    {Provide financial planning advice with numbers and calculations}

    ### Conclusion
    {Summarize the key takeaways and provide actionable financial advice.}

    - Published on {date} in dd/mm/yyyy
    """,    
    show_tool_calls=False,
    markdown=True,
)

# Query function for financial advice based on customer_id
def get_financial_advice(customer_id, csv_path="customer_advice.csv"):
    comment = get_customer_comment(customer_id, csv_path)
    if "No comment found" in comment or "Error loading" in comment:
        return comment

    # Get financial advice based on the customer comment
    response = finance_agent.run(comment)
    return response.content if isinstance(response, RunResponse) else response

print(get_financial_advice("CUST5871"))
"""
# Define the playground app
app = Playground(agents=[finance_agent, websearch_agent]).get_app()

# Serve the app
if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)
"""