import os

import pandas as pd
import plotly.express as px

from dotenv import load_dotenv
from flask import Flask, render_template, request, send_from_directory
from openai import OpenAI


load_dotenv()

OPEN_AI_API_KEY = os.getenv('OPEN_AI_API_KEY')
CSV_PATH = "customer_shopping_data.csv"


ai_client = OpenAI(api_key = OPEN_AI_API_KEY)
app = Flask(__name__)


# Helper function for favicon redering
@app.route('/favicon.svg')
def favicon():
    return send_from_directory('static', 'favicon.svg', mimetype='image/svg+xml')


# Helper function to generate plot from the CSV data
def generate_plot_from_csv():
    """Generate interactive plots from the CSV file using Plotly with transparent backgrounds."""
    try:
        # Load CSV file
        df = pd.read_csv(CSV_PATH)

        # Preprocess data
        df['total_purchase'] = df['price'] * df['quantity']

        # Plot 1: Distribution of total purchases
        fig1 = px.histogram(
            df,
            x='total_purchase',
            nbins=20,
            labels={'total_purchase': 'Total Purchase'}
        )
        fig1.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Total Purchase",
            yaxis_title="Frequency",
            paper_bgcolor="rgba(0,0,0,0)",  # Transparent background
            plot_bgcolor="rgba(0,0,0,0)"   # Transparent plot area
        )

        # Plot 2: Average purchase amount by category
        category_avg = df.groupby('category')['total_purchase'].mean().sort_values().reset_index()
        fig2 = px.bar(
            category_avg,
            x='category',
            y='total_purchase',
            labels={'category': 'Category', 'total_purchase': 'Average Purchase'}
        )
        fig2.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Category",
            yaxis_title="Average Purchase",
            paper_bgcolor="rgba(0,0,0,0)",  # Transparent background
            plot_bgcolor="rgba(0,0,0,0)"   # Transparent plot area
        )

        # Combine both plots
        return fig1.to_html(full_html=False), fig2.to_html(full_html=False)

    except Exception as e:
        return str(e), ""


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


@app.route("/research")
def research():
    """Visualize research process."""
    return render_template("research.html")


@app.route("/visualize")
def visualize():
    """Visualize data from CSV."""
    plot1, plot2 = generate_plot_from_csv()
    return render_template("visualize.html", plot1=plot1, plot2=plot2)


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
