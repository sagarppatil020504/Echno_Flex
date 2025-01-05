############################FINAL draft###################################
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import random
import numpy as np
import gold_prediction as gp
import stock_prediction as sp

app = Flask(__name__)
CORS(app)  # Enable CORS

# Simulate stock and gold data
def generate_data():
    dates = [f"2024-12-{i:02d}" for i in range(1, 29)]
    prices = np.cumsum(np.random.normal(0, 2, len(dates))) + 100
    return dates, prices

def calculate_savings(principal, rate, time):
    return principal + (principal * rate * time)

def calculate_fixed_deposit(principal, rate, time):
    return principal * (1 + rate) ** time

def calculate_recurring_deposit(monthly_deposit, rate, months):
    return sum(monthly_deposit * (1 + rate) ** i for i in range(months))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    investment_type = request.form['investment_type']
    if investment_type == "Savings Account":
        principal = float(request.form['principal'])
        rate = float(request.form['rate']) / 100
        time = int(request.form['time'])
        result = calculate_savings(principal, rate, time)
        return jsonify({"result": f"Total Balance: ${result:.2f}"})

    elif investment_type == "Fixed Deposit":
        principal = float(request.form['principal'])
        rate = float(request.form['rate']) / 100
        time = int(request.form['time'])
        result = calculate_fixed_deposit(principal, rate, time)
        return jsonify({"result": f"Total Balance: ${result:.2f}"})

    elif investment_type == "Recurring Deposit":
        monthly_deposit = float(request.form['monthly_deposit'])
        rate = float(request.form['rate']) / 100 / 12
        months = int(request.form['months'])
        result = calculate_recurring_deposit(monthly_deposit, rate, months)
        return jsonify({"result": f"Total Balance: ${result:.2f}"})


@app.route('/api/stock', methods=['GET','POST'])
def stock_data():
    dates, prices = generate_data()
    sentiment = random.uniform(0.4, 0.9)  # Example sentiment score
    predicted_growth = random.uniform(0.05, 0.15)  # Predicted growth percentage
    return jsonify({
        "dates": dates,
        "prices": list(prices),
        "predicted_value": prices[-1] * (1 + predicted_growth),
        "sentiment": sentiment,
        "features": {
            "volatility": random.uniform(0.5, 1.5),
            "growth_probability": predicted_growth
        }
    })

@app.route('/api/gold', methods=['GET','POST'])
def gold_data():
    data = gp.main()
    # data consists of actual_prices, sentiment_score ,predicted_prices ,future_predictions
    return jsonify({
        "actual_prices": data[0],
        "sentiment_score": data[1] ,+
        "predicted_prices": data[2],
        "future_predictions": data[3]
    })

@app.route('/lump_sum', methods=['GET'])
def lump_sum_investment():
    try:
        initial_investment = float(request.args.get('initial_investment', 0))
        annual_rate = float(request.args.get('annual_rate', 0))
        years = int(request.args.get('years', 0))
        future_value = initial_investment * ((1 + annual_rate) * years)
        return jsonify({'future_value': future_value})
    except ValueError:
        return jsonify({'error': 'Invalid input'}), 400

@app.route('/sip', methods=['GET'])
def sip_investment():
    try:
        monthly_investment = float(request.args.get('monthly_investment', 0))
        annual_rate = float(request.args.get('annual_rate', 0))
        years = int(request.args.get('years', 0))
        months = years * 12
        monthly_rate = annual_rate / 12
        future_value = monthly_investment * ((1 + monthly_rate) ** months - 1) / monthly_rate
        return jsonify({'future_value': future_value})
    except ValueError:
        return jsonify({'error': 'Invalid input'}), 400

@app.route('/portfolio', methods=['POST'])
def diversified_portfolio():
    try:
        data = request.json
        allocations = data.get('allocations', [])
        annual_rates = data.get('annual_rates', [])
        years = data.get('years', 0)

        if len(allocations) != len(annual_rates):
            return jsonify({'error': 'Allocations and annual rates must have the same length'}), 400

        total_future_value = sum(
            allocation * ((1 + annual_rate) ** years)
            for allocation, annual_rate in zip(allocations, annual_rates)
        )
        return jsonify({'future_value': total_future_value})
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid input'}), 400

@app.route('/real_estate', methods=['GET'])
def real_estate_investment():
    try:
        initial_investment = float(request.args.get('initial_investment', 0))
        annual_appreciation = float(request.args.get('annual_appreciation', 0))
        years = int(request.args.get('years', 0))
        future_value = initial_investment * ((1 + annual_appreciation) ** years)
        return jsonify({'future_value': future_value})
    except ValueError:
        return jsonify({'error': 'Invalid input'}), 400

@app.route('/monte_carlo', methods=['POST'])
def monte_carlo_simulation():
    try:
        data = request.json
        initial_investment = data.get('initial_investment', 0)
        annual_rate_mean = data.get('annual_rate_mean', 0)
        annual_rate_std = data.get('annual_rate_std', 0)
        years = data.get('years', 0)
        simulations = data.get('simulations', 1000)

        # Vectorized simulation using NumPy
        annual_rates = np.random.normal(annual_rate_mean, annual_rate_std, (simulations, years))
        future_values = initial_investment * np.prod(1 + annual_rates, axis=1)

        return jsonify({'mean_future_value': np.mean(future_values).item()})
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid input'}), 400

if __name__ == '__main__':
    app.run(debug=True)
