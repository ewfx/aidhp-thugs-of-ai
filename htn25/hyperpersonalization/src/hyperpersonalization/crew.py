from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import FileReadTool
import pandas as pd
import json
import os

# Define input and output directories
DATA_DIR = "newsynthetic"
csv_tool = FileReadTool(file_path=os.path.join(DATA_DIR,'*.csv'))
json1_tool = FileReadTool(file_path=os.path.join(DATA_DIR,'*.json'))
os.makedirs(DATA_DIR, exist_ok=True)

@CrewBase
class Hyperpersonalization:
    """CrewAI for Personalized Financial Recommendations"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def customer_profile_aggregator(self) -> Agent:
        return Agent(
            config=self.agents_config['customer_profile_aggregator'],
            tools=[csv_tool],
            verbose=True
        )

    @agent
    def sentiment_behavior_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['sentiment_behavior_analyst'],
            tools = [json1_tool],
            verbose=True
        )

    @agent
    def recommendation_engine(self) -> Agent:
        return Agent(
            config=self.agents_config['recommendation_engine'],
            tools=[json1_tool],
            verbose=True
        )

    # @task
    # def data_collection_task(self) -> Task:
    #     return Task(
    #         config=self.tasks_config['data_collection_task'],
    #         run=self.merge_customer_data
    #     )

    # @task
    # def sentiment_analysis_task(self) -> Task:
    #     return Task(
    #         config=self.tasks_config['sentiment_analysis_task'],
    #         run=self.analyze_customer_sentiment
    #     )

    # @task
    # def recommendation_task(self) -> Task:
    #     return Task(
    #         config=self.tasks_config['recommendation_task'],
    #         run=self.generate_recommendations_with_llm
    #     )

    @task
    def data_collection_task(self) -> Task:
        return Task(
            config=self.tasks_config['data_collection_task'],
            run=self.merge_customer_data,
            #expected_output=(file=os.path.join(DATA_DIR, "customer_data.json"))  # ✅ Corrected!
        )

    @task
    def sentiment_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config['sentiment_analysis_task'],
            run=self.analyze_customer_sentiment,
            #expected_output=TaskOutput(file=os.path.join(DATA_DIR, "enriched_customer_data.json"))  # ✅ Corrected!
        )

    @task
    def recommendation_task(self) -> Task:
        return Task(
            config=self.tasks_config['recommendation_task'],
            run=self.generate_recommendations_with_llm,
            #expected_output=TaskOutput(file=os.path.join(DATA_DIR, "recommendations.json"))  # ✅ Corrected!
        )


    def merge_customer_data(self):
        """🔹 Task 1: Merge customer data and save as customer_data.json"""
        print("📥 Merging customer data from multiple CSV files...")

        files = [
            "Updated_Transaction_Data.csv",
            "Updated_Sentiment_Data.csv",
            "Updated_Organization_Profile.csv",
            "Updated_Customer_Profile.csv"
        ]

        dfs = []
        for file in files:
            file_path = os.path.join(DATA_DIR, file)
            if os.path.exists(file_path):
                dfs.append(pd.read_csv(file_path))
            else:
                print(f"❌ Warning: {file} not found, skipping...")

        if dfs:
            merged_df = pd.concat(dfs, axis=1)  # Merge along columns
            output_file = os.path.join(DATA_DIR, "customer_data.json")
            merged_df.to_json(output_file, orient="records", indent=4)
            print(f"✅ Merged customer data saved at {output_file}")
        else:
            print("❌ No valid data files found!")

    def analyze_customer_sentiment(self):
        """🔹 Task 2: Perform sentiment analysis and save enriched_customer_data.json"""
        print("🧠 Analyzing customer sentiment...")

        input_file = os.path.join(DATA_DIR, "customer_data.json")
        if not os.path.exists(input_file):
            print("❌ Error: customer_data.json not found!")
            return

        with open(input_file, "r") as f:
            customer_data = json.load(f)

        enriched_data = []
        for customer in customer_data:
            sentiment_score = "Positive" if "high spending" in str(customer) else "Neutral"
            customer["Behavior_Analysis"] = f"Customer exhibits {sentiment_score} spending behavior."
            enriched_data.append(customer)

        output_file = os.path.join(DATA_DIR, "enriched_customer_data.json")
        with open(output_file, "w") as f:
            json.dump(enriched_data, f, indent=4)

        print(f"✅ Sentiment analysis saved at {output_file}")

    def generate_recommendations_with_llm(self):
        """🔹 Task 3: Generate recommendations using LLM and save as recommendations.json"""
        print("🤖 Using LLM to generate personalized recommendations...")

        input_file = os.path.join(DATA_DIR, "enriched_customer_data.json")
        if not os.path.exists(input_file):
            print("❌ Error: enriched_customer_data.json not found!")
            return

        with open(input_file, "r") as f:
            enriched_data = json.load(f)

        # Load product datasets
        credit_cards_file = os.path.join(DATA_DIR, "Credit_Debit_Cards.csv")
        financial_products_file = os.path.join(DATA_DIR, "Financial_Products_Master.csv")

        if not os.path.exists(credit_cards_file) or not os.path.exists(financial_products_file):
            print("❌ Error: Product files not found!")
            return

        credit_cards_df = pd.read_csv(credit_cards_file)
        financial_products_df = pd.read_csv(financial_products_file)

        credit_cards = credit_cards_df.to_dict(orient="records")
        financial_products = financial_products_df.to_dict(orient="records")

        recommendations = {}

        for customer in enriched_data[:10]:  # Process 10 customers
            customer_id = str(customer.get("Customer_Id", "Unknown"))
            customer_behavior = customer.get("Behavior_Analysis", "Neutral behavior")

            llm_prompt = f"""
            Given the following customer behavior data:
            {customer_behavior}

            Recommend the best 5 financial products from:
            - Credit/Debit Cards: {credit_cards}
            - Financial Products: {financial_products}

            Provide a JSON response:
            {{
                "customer_id": "{customer_id}",
                "recommendations": [
                    {{
                        "product_name": "<product_name>",
                        "reason": "<why this product is suitable>"
                    }},
                    ...
                ]
            }}
            """

            recommended_response = self.recommendation_engine().run(prompt=llm_prompt)

            try:
                recommendations[customer_id] = json.loads(recommended_response)["recommendations"]
            except Exception as e:
                print(f"❌ Error parsing response for Customer {customer_id}: {e}")
                recommendations[customer_id] = []

        output_file = os.path.join(DATA_DIR, "recommendations.json")
        with open(output_file, "w") as f:
            json.dump(recommendations, f, indent=4)

        print(f"✅ Recommendations saved at {output_file}")

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
