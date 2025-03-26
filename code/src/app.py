import gradio as gr
import os
import sys
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from sentiment import get_sentiment_score
from read_data import get_customer_details
from fetch import get_customer_sentiments_transaction

customer_row = None
def greet_and_go(customer_id):
    customer_row, customer_dict = get_customer_details(customer_id)
    if customer_row is None:
        return (
            gr.update(value=pd.DataFrame(), visible=False),   # Clear and hide DataFrame
            gr.update(visible=False),                         # Dummy output stays hidden
            gr.update(value="Customer not found.", visible=True),
            gr.update(value="", visible=False)                # Hide second message
        )
    message1 = f"Welcome back, {customer_dict['Customer_Id']}!"
    if customer_dict['Customer_Id'].startswith("CUST"):
           message2 = (
        	f"You are a {customer_dict['Occupation']} in {customer_dict['Location']} earning "
        	f"${customer_dict['Income (per year in dollars)']:,.0f} per year."
    	   )
    else:
           message2 = (
                f"You are a organization in {customer_dict['Industry']} with around {customer_dict['Employees_Mid']} employees with roughly"
                f"${customer_dict['Revenue_Mid']:,.0f} revenue"
           ) 

    return (
        gr.update(value=customer_row, visible=True),
        gr.update(visible=False),  # second DataFrame unused
        gr.update(value=message1, visible=True),
        gr.update(value=message2, visible=True)
    )

def fetch_sentiments_and_transactions(customer_id):
    sentiment_row, _, transaction_row, _ = get_customer_sentiments_transaction(customer_id)
    
    if sentiment_row is None or transaction_row is None or sentiment_row.empty or transaction_row.empty:
        return (
            gr.update(value=pd.DataFrame(), visible=False),  # sentiment
            gr.update(value=pd.DataFrame(), visible=False),  # transaction
        )
    
    return (
        gr.update(value=sentiment_row, visible=True),
        gr.update(value=transaction_row, visible=True),
    )


logo_path = os.path.join(os.path.dirname(__file__), "logo.jpg")

with gr.Blocks() as demo:
    with gr.Row():
        gr.Image(logo_path, show_label=False, height=100, elem_classes=["no-border-img"])

    gr.Markdown("### 🔍 Customer Info Lookup")

    with gr.Row():
        input_text = gr.Textbox(label="Enter Customer ID (e.g. CUST2035)")
        submit_btn = gr.Button("Submit", scale=1)
    with gr.Row():
        output_msg1 = gr.Textbox(label="", visible=False, show_label=False)

    with gr.Row():
        output_df = gr.Dataframe(label="Customer Details", visible=False)
        dummy_output = gr.Dataframe(visible=False)

    with gr.Row():
        output_msg2 = gr.Textbox(label="", visible=False, show_label=False)

    with gr.Row():
        fetch_button = gr.Button("Fetch Sentiments & Transactions", scale=1)

    with gr.Row():
        sentiment_df = gr.Dataframe(label="Sentiment Info", visible=False)

    with gr.Row():
        transaction_df = gr.Dataframe(label="Transaction Info", visible=False)
    
    submit_btn.click(
        fn=greet_and_go,
        inputs=[input_text],
        outputs=[output_df, dummy_output, output_msg1, output_msg2]
    )

    fetch_button.click(
        fn=fetch_sentiments_and_transactions,
        inputs=[input_text],
        outputs=[sentiment_df, transaction_df]
    )

demo.launch()

