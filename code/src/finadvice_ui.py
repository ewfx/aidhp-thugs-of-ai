import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from phi.agent import Agent, RunResponse
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo

# Load environment variables
load_dotenv()
phi_api_key = os.getenv("PHI_API_KEY")
if not phi_api_key:
    raise ValueError("PHI_API_KEY is missing. Check your .env file.")

# Load customer comments CSV
@st.cache_data
def load_customer_comments(csv_path="customer_advice.csv"):
    try:
        return pd.read_csv(csv_path, encoding="ISO-8859-1")
    except Exception as e:
        st.error(f"Error loading CSV file: {str(e)}")
        return None

# Get customer comment by ID
def get_customer_comment(customer_id, csv_path="customer_advice.csv"):
    df = load_customer_comments(csv_path)
    if df is None:
        return "Error loading customer data."
    
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
def get_financial_advice_from_id(customer_id, csv_path="customer_advice.csv"):
    comment = get_customer_comment(customer_id, csv_path)
    if "No comment found" in comment or "Error loading" in comment:
        return comment

    # Get financial advice based on the customer comment
    response = finance_agent.run(comment)
    return response.content if isinstance(response, RunResponse) else response

# Query function for financial advice from free-text user input
def get_financial_advice_from_text(user_text):
    if not user_text.strip():
        return "⚠️ Please enter a valid financial query."
    
    # Get financial advice from user text input
    response = finance_agent.run(user_text)
    return response.content if isinstance(response, RunResponse) else response

# Streamlit UI
st.title("💰 Financial Advice Generator")
st.write("Choose how you want to receive financial advice.")

# Select option: Customer ID-based OR Free-text query
option = st.radio("Select an Option:", ["📌 Get Advice by Customer ID", "✍️ Ask a Financial Question"])

if option == "📌 Get Advice by Customer ID":
    st.subheader("🔹 Get Financial Advice Based on Customer ID")
    customer_id = st.text_input("Enter Customer ID:")
    if st.button("Get Financial Advice"):
        if customer_id:
            with st.spinner("Fetching advice... ⏳"):
                advice = get_financial_advice_from_id(customer_id)
            st.subheader("📌 Financial Advice:")
            st.markdown(advice)
        else:
            st.warning("⚠️ Please enter a valid Customer ID.")

elif option == "✍️ Ask a Financial Question":
    st.subheader("✍️ Free-Text Financial Advice")
    user_query = st.text_area("Describe your financial situation or ask a question:")
    if st.button("Get Advice"):
        if user_query:
            with st.spinner("Generating advice... ⏳"):
                advice = get_financial_advice_from_text(user_query)
            st.subheader("📌 Financial Advice:")
            st.markdown(advice)
        else:
            st.warning("⚠️ Please enter a valid query.")