#!/usr/bin/env python
import sys
import warnings
import pandas as pd

from datetime import datetime

from hyperpersonalization.crew import Hyperpersonalization

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# Load the datasets
CUSTOMER_PROFILES = "/newsynthetic/Updated_Customer_Profile.csv"
TRANSACTION_DATA = "/newsynthetic/Updated_Transaction_Data.csv"
SENTIMENT_DATA = "/newsynthetic/Updated_Sentiment_Data.csv"
ORG_PROFILES = "/newsynthetic/Updated_Organization_Profile.csv"
CREDIT_DEBIT_CARDS = "/newsynthetic/Credit_Debit_Cards.csv"
FINANCIAL_PRODUCTS = "/newsynthetic/Financial_Products_Master.csv"


def load_data():
    """Load all datasets into Pandas DataFrames."""
    return {
        "customers": pd.read_csv(CUSTOMER_PROFILES),
        "transactions": pd.read_csv(TRANSACTION_DATA),
        "sentiments": pd.read_csv(SENTIMENT_DATA),
        "organizations": pd.read_csv(ORG_PROFILES),
        "cards": pd.read_csv(CREDIT_DEBIT_CARDS),
        "products": pd.read_csv(FINANCIAL_PRODUCTS)
    }


def run():
    """
    Run the hyper-personalization engine for a given Customer ID.
    """
    #print(f"🔍 Fetching top 5 recommendations for Customer ID: {customer_id}...")

    #inputs = {"customer_id": str(customer_id)}

    try:
        results = Hyperpersonalization().crew().kickoff()
        print("\n**Top 5 Recommended Products:**")
        for idx, product in enumerate(results, 1):
            print(f"{idx}. {product}")
    except Exception as e:
        print(f"Error: {e}")


def train(n_iterations=10):
    """
    Train the hyper-personalization engine to improve recommendations.
    """
    print(f"🚀 Training Hyper-Personalization Engine for {n_iterations} iterations...")

    try:
        Hyperpersonalization().crew().train(n_iterations=n_iterations, filename="trained_model.json")
        print("✅Training completed successfully!")
    except Exception as e:
        print(f"Error during training: {e}")


def replay(task_id):
    """
    Replay a specific task execution in the crew pipeline.
    """
    print(f"🔄 Replaying task {task_id}...")

    try:
        Hyperpersonalization().crew().replay(task_id=task_id)
        print("✅ Task replayed successfully!")
    except Exception as e:
        print(f"Error during replay: {e}")


def test(n_iterations=5, model_name="deepseek-r1"):
    """
    Test the hyper-personalization engine with a given model.
    """
    print(f"🧪 Testing the system for {n_iterations} iterations using model: {model_name}...")

    try:
        Hyperpersonalization().crew().test(n_iterations=n_iterations, openai_model_name=model_name)
        print("✅ Testing completed successfully!")
    except Exception as e:
        print(f"❌ Error during testing: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python main.py <Customer_ID>         → Get recommendations")
        print("  python main.py train <iterations>    → Train the engine")
        print("  python main.py replay <task_id>      → Replay a task")
        print("  python main.py test <iterations> <model_name> → Test with a model")
    else:
        command = sys.argv[1]
        
        if command == "train":
            iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            train(n_iterations=iterations)

        elif command == "replay":
            if len(sys.argv) < 3:
                print("❌ Error: Task ID is required for replay.")
            else:
                replay(sys.argv[2])

        elif command == "test":
            iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            model_name = sys.argv[3] if len(sys.argv) > 3 else "deepseek-r1"
            test(n_iterations=iterations, model_name=model_name)

        else:
            run(command)  # Treat as Customer ID
