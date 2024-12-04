from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os
from openai import OpenAI

load_dotenv()

OPEN_AI_API_KEY = os.getenv('OPEN_AI_API_KEY')

# Flask app
app = Flask(__name__)

# Path to the CSV file
CSV_PATH = "customer_shopping_data.csv"

# OpenAI API key
ai_client = OpenAI(api_key=OPEN_AI_API_KEY)

# Helper function to generate plot from the CSV data
def generate_plot_from_csv():
    """Generate plots from the CSV file."""
    try:
        # Load CSV file
        df = pd.read_csv(CSV_PATH)

        # Preprocess data
        df['total_purchase'] = df['price'] * df['quantity']

        # Create a plot
        fig, ax = plt.subplots(1, 2, figsize=(14, 7))

        # Plot 1: Distribution of total purchases
        df['total_purchase'].plot(kind='hist', bins=20, ax=ax[0], color='skyblue', edgecolor='black')
        ax[0].set_title('Distribution of Total Purchases')
        ax[0].set_xlabel('Total Purchase')
        ax[0].set_ylabel('Frequency')

        # Plot 2: Average purchase amount by category
        category_avg = df.groupby('category')['total_purchase'].mean().sort_values()
        category_avg.plot(kind='bar', ax=ax[1], color='coral', edgecolor='black')
        ax[1].set_title('Average Purchase Amount by Category')
        ax[1].set_xlabel('Category')
        ax[1].set_ylabel('Average Purchase')

        # Save plot to a string buffer
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plot_data = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()

        return plot_data
    except Exception as e:
        return str(e)

# Chatbot function
def ask_chat_gpt(prompt):
    """Ask OpenAI GPT model."""
    try:
        response = ai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return str(e)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/visualize")
def visualize():
    """Visualize data from CSV."""
    plot_data = generate_plot_from_csv()
    return render_template("visualize.html", plot_url=plot_data)

@app.route("/chatbot", methods=["GET", "POST"])
def chatbot():
    """Chatbot page."""
    if request.method == "POST":
        user_prompt = request.form.get("prompt")
        if user_prompt:
            response = ask_chat_gpt(user_prompt)
            return render_template("chatbot.html", user_prompt=user_prompt, bot_response=response)
    return render_template("chatbot.html")

if __name__ == "__main__":
    # Ensure CSV file exists
    if not os.path.exists(CSV_PATH):
        print(f"Error: The file {CSV_PATH} does not exist. Please provide a valid CSV file.")
    else:
        app.run(debug=True)
