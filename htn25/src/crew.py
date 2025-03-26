from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
import pandas as pd

@CrewBase
class Hackathon:
    """CrewAI for Personalized Financial Recommendations"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def customer_profile_aggregator(self) -> Agent:
        return Agent(
            config=self.agents_config['customer_profile_aggregator'],
            verbose=True
        )

    @agent
    def sentiment_behavior_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['sentiment_behavior_analyst'],
            verbose=True
        )

    @agent
    def recommendation_engine(self) -> Agent:
        return Agent(
            config=self.agents_config['recommendation_engine'],
            verbose=True
        )

    @task
    def data_collection_task(self) -> Task:
        return Task(config=self.tasks_config['data_collection_task'])

    @task
    def sentiment_analysis_task(self) -> Task:
        return Task(config=self.tasks_config['sentiment_analysis_task'])

    @task
    def recommendation_task(self) -> Task:
        return Task(config=self.tasks_config['recommendation_task'])

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
