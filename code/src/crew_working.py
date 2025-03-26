from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
import pandas as pd
import json

@CrewBase
class Hyperpersonalization:
    """CrewAI for Personalized Financial Recommendations"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    # 🔹 1️⃣ Agent: Aggregates & Merges Customer Data
    @agent
    def customer_profile_aggregator(self) -> Agent:
        return Agent(
            config=self.agents_config['customer_profile_aggregator'],
            verbose=True
        )

    # 🔹 2️⃣ Agent: Extracts Sentiment & Behavioral Insights
    @agent
    def sentiment_behavior_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['sentiment_behavior_analyst'],
            verbose=True
        )

    # 🔹 3️⃣ Agent: Uses LLM to Generate Personalized Recommendations
    @agent
    def recommendation_engine(self) -> Agent:
        return Agent(
            config=self.agents_config['recommendation_engine'],
            verbose=True
        )

    # 🔹 1️⃣ Task: Merge all customer data (including sentiment)
    @task
    def data_collection_task(self) -> Task:
        return Task(
            description="Merge customer transactions, profiles, and sentiment data into a single file.",
            expected_output="merged_customer_data.json",
            agent=self.customer_profile_aggregator(),
            function=self.merge_customer_data
        )

    # 🔹 2️⃣ Task: Analyze Customer Behavior & Spending Patterns
    @task
    def sentiment_analysis_task(self) -> Task:
        return Task(
            description="Analyze customer behavior based on transaction and sentiment data.",
            expected_output="enriched_customer_data.json",
            agent=self.sentiment_behavior_analyst(),
            function=self.analyze_customer_behavior
        )

    # 🔹 3️⃣ Task: Generate Personalized Recommendations using LLM
    @task
    def recommendation_task(self) -> Task:
        return Task(
            description="Generate personalized financial product recommendations using LLM.",
            expected_output="recommendations.json",
            agent=self.recommendation_engine(),
            function=self.generate_recommendations
        )

    # 🔹 Function 1: Merge Customer Data (Now Includes Sentiment!)
    def merge_customer_data(self):
        print("📥 Merging customer transaction, profile & sentiment data...")

        # Load all CSVs
        transaction_df = pd.read_csv("/newsynthetic/Updated_Transaction_Data.csv")
        customer_df = pd.read_csv("/newsynthetic/Updated_Customer_Profile.csv")
        org_profile_df = pd.read_csv("/newsynthetic/Updated_Organization_Profile.csv")
        sentiment_df = pd.read_csv("/newsynthetic/Updated_Sentiment_Data.csv")

        # Merge datasets
        merged_df = (
            transaction_df
            .merge(customer_df, on="Customer_Id", how="left")
            .merge(org_profile_df, on="Customer_Id", how="left")
            .merge(sentiment_df, on="Customer_Id", how="left")  # 🔹 Sentiment added!
        )

        # Save merged data as JSON
        merged_df.to_json("/newsynthetic/merged_customer_data.json", orient="records", indent=4)

        print("✅ Merged data saved as merged_customer_data.json")

    # 🔹 Function 2: Analyze Sentiment & Behavior
    def analyze_customer_behavior(self):
        print("🔍 Analyzing customer behavior & spending patterns...")

        # Load merged data
        with open("/newsynthetic/merged_customer_data.json", "r") as f:
            customer_data = json.load(f)

        enriched_data = []
        for customer in customer_data:
            enriched_customer = {
                **customer,
                "Behavior_Analysis": f"Spending pattern: {customer.get('Category', 'Unknown')}, "
                                     f"Sentiment Score: {customer.get('Sentiment Score', 'Neutral')}"
            }
            enriched_data.append(enriched_customer)

        # Save enriched customer data
        with open("/newsynthetic/enriched_customer_data.json", "w") as f:
            json.dump(enriched_data, f, indent=4)

        print("✅ Enriched data saved as enriched_customer_data.json")

    # 🔹 Function 3: Generate LLM-Based Recommendations
    def generate_recommendations(self):
        print("🎯 Generating personalized recommendations using CrewAI...")

        # Load enriched customer data
        with open("enriched_customer_data.json", "r") as f:
            enriched_data = json.load(f)

        recommendations = {}
        for customer in enriched_data:
            customer_id = str(customer["Customer_Id"])

            # LLM agent generates recommendations
            recommendations[customer_id] = self.recommendation_engine().run(
                context=customer
            )

        # Save recommendations
        with open("/newsynthetic/recommendations.json", "w") as f:
            json.dump(recommendations, f, indent=4)

        print("✅ Recommendations saved in recommendations.json")

    # 🔹 The Crew: Executes tasks in sequence
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=[
                self.data_collection_task(),
                self.sentiment_analysis_task(),
                self.recommendation_task(),
            ],
            process=Process.sequential,
            verbose=True,
        )
