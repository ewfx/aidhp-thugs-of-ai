# 🚀 AI-Driven Hyper-Personalization and Recommendation System

## 📌 Table of Contents
- [Introduction](#introduction)
- [Demo](#demo)
- [Inspiration](#inspiration)
- [What It Does](#what-it-does)
- [How We Built It](#how-we-built-it)
- [Challenges We Faced](#challenges-we-faced)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)
- [Team](#team)

---

## 🎯 Introduction
### Project Overview:
Our project aims at developing an **AI based Hyper-Personalized and Recommendation System**  that will change the way businesses deal with customers. The system uses **Generative AI** to gather and analyze data from multiple sources like customer profiles, their sentiments, social media behavior, and even purchase history to provide them recommendations in the form of services, products, or content to be created for them in real-time.

### Problem Statement:
We are trying to solve the issue of modern customers wanting personal treatment. This is a complex problem that requires developing a system that goes beyond simply providing offerings, but also helps businesses identify ways to better engage with customers, forecast behaviors, and implements strategic decisions all while using AI ethically and safeguarding data privacy.



## 🎥 Demo

📹 [Financial Advice generator (Recorded Demo)](https://www.youtube.com/watch?v=KVCC3gBYXZA)

📹 [Customer credit card generator (Recorded Demo)](https://www.youtube.com/watch?v=ZoGcexydSGk)

🖼️ Screenshots:
- <img width="600" alt="image" src="https://github.com/user-attachments/assets/253ade1a-1ef5-4351-ba36-d060596ee545" />
- <img width="600" alt="image" src="https://github.com/user-attachments/assets/ccc1dc3e-2b52-4409-a995-6ffe7d770efe" />
- <img width="600" alt="image" src="https://github.com/user-attachments/assets/1d1338bd-736d-41e7-aac3-185fe0ff274a" />
- <img width="600" alt="image" src="https://github.com/user-attachments/assets/2065c3ef-2054-4384-8f3e-3de201b1219b" />
- <img width="600" alt="image" src="https://github.com/user-attachments/assets/b2a3bdef-9428-4d10-abc4-1d8089a10132" />
- <img width="600" alt="image" src="https://github.com/user-attachments/assets/79061fbf-3b8a-471a-a4b1-b6c1fe5d4cad" />
- <img width="600" alt="image" src="https://github.com/user-attachments/assets/0bce2e76-7113-4198-8df7-ca35b7727b4d" />
- <img width="600" alt="image" src="https://github.com/user-attachments/assets/63824e2d-ba22-4ef0-a6a4-7a96214ce0f4" />
- <img width="600" alt="image" src="https://github.com/user-attachments/assets/565a03ce-3ad3-4b03-ab52-ebf297323e5e" />
- <img width="1172" alt="image" src="https://github.com/user-attachments/assets/9b38af94-507c-4450-8496-ed53a4f233b0" />
- <img width="600" alt="image" src="https://github.com/user-attachments/assets/1ef8ce95-45c6-4530-b93b-23b01b915263" />
- <img width="600" alt="image" src="https://github.com/user-attachments/assets/27260b41-f932-4ddd-9268-da462b876ee3" />




## 💡 Inspiration
The inspiration behind this project comes from the growing need for **meaningful customer engagement** in an era of **information overload** and **increasing customer expectations**. Traditional recommendation systems often fall short in capturing the **emotions, preferences, and intent** behind customer actions.

We envisioned a solution that combines **Generative AI, real-time data analysis**, and **multi-channel inputs** to deliver **hyper-personalized experiences** — not just to sell, but to truly understand and connect with each individual customer. Our goal is to help businesses build **stronger relationships**, make **data-driven decisions**, and **respect privacy**, all while staying ahead in a competitive market.

## ⚙️ What It Does
### 1. Two-Mode Financial Advice System
#### Use Case 1: Existing Customers
- User selects "Get advice by Customer ID".
- System pulls user data (age, salary, liabilities, etc.) from the internal database.
- LLM analyzes the customer's financial situation and gives personalized financial advice.

#### Use Case 2: Open/Free-Form Financial Queries
- User can input financial questions via text or voice.
- System uses speech-to-text if using a mic.
- LLM generates general or custom advice based on provided inputs.

### 2. Agentic AI Architecture
- Built using the FiData framework that supports multiple intelligent agents:
- Finance Agent: Handles typical financial questions using internal knowledge.
- Web Search Agent: Activates when external or real-time information is needed (e.g., stock advice), and fetches up-to-date data via web search.

### 3. LLM-Powered Recommendation Engine
- Processes user data and natural language queries.

#### Generates:
- Budgeting strategies
- Debt repayment plans
- Investment recommendations
- Portfolio diversification tips
- Stock recommendations (if available online)

### 4. Interactive UI with Streamlit
- Simple, user-friendly web interface.
#### Options to:
- Choose between known customer and free-form input.
- Enter input via text or mic.
- Click a button to instantly get advice.

### 5. Voice Input Support
- Converts voice to text using speech recognition.
- Supports hands-free and natural interaction for financial queries.

### 6. Real-Time Information Retrieval
For dynamic queries (like stock tips), the system:
-    Delegates the query to the Web Search Agent.
-    Pulls latest analyst ratings or updates from the web.
-    Displays them to the user in a structured format.

### 7. Choose a Credit Card Design – Select from multiple card patterns.
- Upload Your Photo – Add a passport-size image for customization.
- Automatic Image Processing – The uploaded photo is resized and placed on the card.
- Instant Preview – See the customized credit card with your photo.
- Download Your Card – Save and download the final personalized credit card.
- This tool allows users to easily create a custom credit card look with their own photo in just a few click

### 8. Modular and Scalable Design
- Agentic approach allows adding new agents (e.g., tax agent, retirement planner).
- Easily extendable to other domains or customer personas.
  
### 9. Hyper-Personalized Financial Recommendations
- Uses AI agents to analyze customer profiles, transactions, and sentiment data.
- Provides top 5 personalized financial product recommendations for each customer.
  
### 10. Multi-Agent AI System (CrewAI Framework)
- Customer Profile Aggregator: Collects and processes customer data.
- Sentiment Behavior Analyst: Analyzes customer sentiment from interactions.
- Recommendation Engine: Generates financial product recommendations.
  
### 11. Automated Data Processing & Machine Learning
- Loads customer, transaction, sentiment, and financial product data.
- Runs AI models to improve recommendations based on user data.

### 12. Multiple Functionalities for AI Training & Testing
- Recommendation Generation: Get personalized financial suggestions for a customer.
- Training Mode: Improve AI recommendation accuracy through iterative training.
- Task Replay: Debug and re-run specific AI tasks for optimization.
- Testing Mode: Evaluate AI performance using different models.

### 13. Command-Line Interface for Easy Execution
- python main.py <Customer_ID> → Get recommendations for a customer.
- python main.py train <iterations> → Train the AI model.
- python main.py replay <task_id> → Replay a specific AI task.
- python main.py test <iterations> <model_name> → Test AI with different models.

### 14. Scalable and Adaptive AI Pipeline
- Uses a CrewAI workflow to execute tasks sequentially.
- Supports fine-tuning AI models for better financial insights.
- Modular structure allows easy expansion with new AI capabilities.


## 🛠️ How We Built It
Briefly outline the technologies, frameworks, and tools used in development.

### LLM (Large Language Model)
- Core engine for generating financial advice based on user input (text or voice).
- Used to understand and respond to customer queries, both structured and unstructured.

### phiData & crewAI (Agentic AI Framework)
Used to create multiple agents:
- Finance Agent: Handles finance-related queries based on LLM’s internal knowledge.
- Web Search Agent: Fetches live data (e.g., latest stock recommendations) if LLM lacks up-to-date info.

### Streamlit
- Framework used to build the interactive UI.
- Allows running the app locally with a simple command (streamlit run finise_speech.py).
- Provides input options for customer ID or free-form financial questions (text/voice).

### Speech Recognition
- Converts voice input from mic to text.
- Enables free-form verbal input for financial advice.

### Web Search Integration (via DuckDuckGo or similar)
- Used by the Web Search Agent to retrieve current information (e.g., Facebook stock recommendations).

### Database
- Stores existing customer data (ID, age, salary, liabilities, comments).
- Used in the first use case to generate personalized advice based on stored records.

## 🚧 Challenges We Faced
- Syntetic data generation
- Collaboration & Integration
-  

## 🏃 How to Run
1. Clone the repository  
   ```sh
   git clone https://github.com/ewfx/aidhp-thugs-of-ai
   ```
2. Install dependencies  
   ```sh
   pip install -r requirements.txt
   ```
3. Run the project  
   ```sh
   streamlit run <Give python files>
   ```

## 🏗️ Tech Stack
- 🔹 Frontend: Streamlit/Gradio
- 🔹 Backend: Python, FastAPI, Transformers, crewAI, phiData, opencv, groq, llama, OpenAI/vaderSentiment
- 🔹 Database: csv files

## 👥 Team
- **Vikas Chandra** - [GitHub](https://github.com/vchandraiitk) | [LinkedIn](https://www.linkedin.com/in/vikas-chandra-3112496/)
- **Madhucchand Darbha** - [GitHub](https://github.com/darbhamc) | [LinkedIn](https://www.linkedin.com/in/madhucchand-darbha-ph-d-b802033/)
- **Ramesh Nandipalli** - [GitHub](https://github.com/rnandip) | [LinkedIn](https://www.linkedin.com/in/ramesh-nandipalli-bb56a68/)
- **Krishna P. Duvvur** - [GitHub](https://github.com/krishnapranesh) | [LinkedIn](https://www.linkedin.com/in/krishna-pranesh-duvvur/)
