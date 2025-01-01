import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define backend API endpoints
API_BASE = "http://127.0.0.1:5000"

API_STOCK = f"{API_BASE}/api/stock"
API_GOLD = f"{API_BASE}/api/gold"
API_LUMP_SUM = f"{API_BASE}/lump_sum"
API_SIP = f"{API_BASE}/sip"
API_PORTFOLIO = f"{API_BASE}/portfolio"
API_REAL_ESTATE = f"{API_BASE}/real_estate"
API_MONTE_CARLO = f"{API_BASE}/monte_carlo"

# [theme]
primaryColor = "#F02E65"  # Matches the pinkish Appwrite accent
backgroundColor = "#1E1E1E"  # Dark background
secondaryBackgroundColor = "#292929"  # Lighter card background
textColor = "#FFFFFF"  # White text
font = "sans-serif"

st.markdown(
    """
    <style>
    /* Navigation Sidebar */
    [data-testid="stSidebar"] {
        background-color: #292929;
        color: white;
    }

    /* Titles and Subtitles */
    .css-18e3th9 {
        font-family: 'Helvetica', sans-serif;
        font-size: 24px;
        color: #F02E65;
        font-weight: bold;
    }

    .css-1d391kg {
        font-family: 'Arial', sans-serif;
        color: white;
    }

    /* Background */
    .main {
        background-color: #1E1E1E;
        color: white;
    }

    /* Cards */
    .card {
        background-color: #292929;
        padding: 20px;
        margin: 10px 0;
        border-radius: 8px;
    }

    /* Charts */
    .chart {
        border-radius: 8px;
        background: #292929;
        padding: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Sidebar Navigation
st.sidebar.title("OPTIONS")
page = st.sidebar.radio("Go to", ["Home" , "Invest in Bank","Invest in Stock", "Invest in Gold"])
with st.sidebar:
    messages = st.container(height=200)
    if prompt := st.chat_input("Say something"):
        messages.chat_message("user").write(prompt)
        messages.chat_message("assistant").write(f"Echo: {"i would surely assist u until but my backend services are low sorry for inconvience  "}")
# Home Page
if page == "Home":
    st.snow()
    st.title("Investment Portfolio")
    st.subheader("Start Your Investment")

    amt = st.number_input("Enter the amount to be invested:", min_value=0, step=1)

    if amt > 0:
        st.write("Adjust your investments in various fields using the sliders below.")
        investment = st.slider(
            "Set your investment distribution:",
            min_value=0.0,step=(amt/20),
            max_value=float(amt),
            value=(amt / 4, amt / 2),
        )

        Bank_invest = investment[0]
        stock_investment = investment[1] - investment[0]
        gold_investment = amt - investment[1]

        st.write("### Investment Breakdown:")
        st.write(f"**Bank**: ${Bank_invest:.2f}")
        st.write(f"**Stocks**: ${stock_investment:.2f}")
        st.write(f"**Gold**: ${gold_investment:.2f}")
    else:
        st.warning("Please enter a valid amount to be invested.")

# Stock Prices Page
elif page == "Invest in Stock":
    st.snow()
    st.title("Stock Data and Analysis")
    
    # Input field for the Google Drive link
    file_url = "https://drive.google.com/file/d/1ZZp8DgTMt41ZiyXwl8q5wjYaPXdRzQdq/view?usp=sharing"
    
    if file_url:
        try:
            # Extract the file ID from the Google Drive link
            if "/file/d/" in file_url:
                file_id = file_url.split("/d/")[1].split("/")[0]
            elif "id=" in file_url:
                file_id = file_url.split("id=")[1].split("&")[0]
            else:
                st.error("Invalid Google Drive link format.")
                file_id = None

            if True:
                # Generate the download link
                download_url = f"https://drive.google.com/uc?id={file_id}"
                # Load the CSV file into a DataFrame
                df = pd.read_csv(download_url)
                
                # Display the DataFrame in Streamlit
                st.subheader("Stock Data Table")
                st.dataframe(df)
                
                # Display summary statistics
                st.subheader("Summary Statistics")
                st.write(df.describe())
                
                # Graphical analysis
                st.subheader("Graphical Analysis")
                # numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
                
                # Check for numerical columns
                numerical_columns = df.select_dtypes(include=['number']).columns
                
                if not numerical_columns.empty:
                    # Extract data for plotting (assuming your df structure)
                    actual_prices = df['actual_prices'] 
                    live_predictions = df['live_predictions']  
                    future_predictions = df['future_predictions'] 
                    ticker = "Apple"
                    
                    # Create and display plot
                    fig, ax = plt.subplots(figsize=(14, 7))
                    ax.plot(actual_prices, label="Actual Prices", color="blue")
                    ax.plot(live_predictions, label="Predicted Prices (Live)", color="red")
                    ax.plot(range(len(actual_prices), len(actual_prices) + len(future_predictions)),
                            future_predictions, label="Predicted Prices (Future)", color="green", linestyle="--")
                    ax.set_title(f"Live and Future Stock Price Prediction for {ticker}")
                    ax.set_xlabel("Time")
                    ax.set_ylabel("Stock Price")
                    ax.legend()
                    
                    st.pyplot(fig)  
                    column_to_plot = st.selectbox("Select a column to visualize:", numerical_columns)
                    # Line chart
                    st.line_chart(df[column_to_plot])
                    
                    # Bar chart
                    st.bar_chart(df[column_to_plot])
                    
                    # Histogram
                    st.subheader("Histogram")
                    fig, ax = plt.subplots()
                    ax.hist(df[column_to_plot], bins=20, color='blue', alpha=0.7)
                    ax.set_title(f"Histogram of {column_to_plot}")
                    ax.set_xlabel(column_to_plot)
                    ax.set_ylabel("Frequency")
                    st.pyplot(fig)
                else:
                    st.warning("No numerical columns available for plotting.")
        except Exception as e:
            st.error(f"An error occurred while loading the file: {e}")

# elif page == "Invest in Stock":
#     st.snow()
#     file_url = st.text_input("https://drive.google.com/file/d/1ZZp8DgTMt41ZiyXwl8q5wjYaPXdRzQdq/view?usp=drive_link")
#     try:
#         # Load the CSV file
#         df = pd.read_csv(file_url)
#         st.dataframe(df)
        
#     except Exception as e:
#         st.error(f"An error occurred: {e}")
#     st.title("Stock Prices")
    # ir=st.number_input("enter recent interest rate",max_value=10)
    # ifl=st.number_input("enter recent Inflation",max_value=20)
    # gg=st.number_input("enter recent GDP growth",max_value=10)
    # st.write(f"interest_rate: {ir},inflation:{ifl} ,gdp_growth: {gg} ")
    # response = requests.get(API_STOCK).json()
    # dates, prices = response["dates"], response["prices"]

    # fig, ax = plt.subplots()
    # ax.plot(dates, prices, label="Stock Prices")
    # ax.set_xlabel("Date")
    # ax.set_ylabel("Price")
    # ax.legend()
    # st.pyplot(fig)

    # st.write(f"Predicted Value (after growth): **${response['predicted_value']:.2f}**")
    # st.write(f"Live Sentiment: **{response['sentiment']:.2f}**")
    # features_df = pd.DataFrame([response["features"]])
    # st.table(features_df)

# Gold Prices Page
elif page == "Invest in Gold":
    st.snow()
    st.title("Gold Prices and Predictions")
    file_url = "https://drive.google.com/file/d/1iYmn4TrWDWkIbAsm5q1beka3XRQl0Z6_/view?usp=sharing"
    
    if file_url:
        try:
            # Extract the file ID from the Google Drive link
            if "/file/d/" in file_url:
                file_id = file_url.split("/d/")[1].split("/")[0]
            elif "id=" in file_url:
                file_id = file_url.split("id=")[1].split("&")[0]
            else:
                st.error("Invalid Google Drive link format.")
                file_id = None

            if True:
                # Generate the download link
                download_url = f"https://drive.google.com/uc?id={file_id}"
                # Load the CSV file into a DataFrame
                df = pd.read_csv(download_url)
                
                # Display the DataFrame in Streamlit
                st.subheader("Stock Data Table")
                st.dataframe(df)
                
                # Display summary statistics
                st.subheader("Summary Statistics")
                st.write(df.describe())
                
                # Graphical analysis
                st.subheader("Graphical Analysis")
                numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
                
                if not numerical_columns.empty:
                    # Dropdown to select column for visualization
                    column_to_plot = st.selectbox("Select a column to visualize:", numerical_columns)
                    
                    # Line chart
                    st.line_chart(df[column_to_plot])
                    
                    # Bar chart
                    st.bar_chart(df[column_to_plot])
                    
                    # Histogram
                    st.subheader("Histogram")
                    fig, ax = plt.subplots()
                    ax.hist(df[column_to_plot], bins=20, color='blue', alpha=0.7)
                    ax.set_title(f"Histogram of {column_to_plot}")
                    ax.set_xlabel(column_to_plot)
                    ax.set_ylabel("Frequency")
                    st.pyplot(fig)
                else:
                    st.warning("No numerical columns available for plotting.")
        except Exception as e:
            st.error(f"An error occurred while loading the file: {e}")

    # Fetch data from Flask API
    # response = requests.post(API_GOLD).json()
    # actual_time = response["actual_time"]
    # actual_prices = response["actual_prices"]
    # predicted_time = response["predicted_time"]
    # predicted_prices = response["predicted_prices"]
    # future_time = response["future_time"]
    # future_predictions = response["future_prediction"]
    # st.header("Gold-Related News Sentiment")

    # # Visualization using Matplotlib
    # plt.figure(figsize=(14, 7))
    # plt.plot(actual_time, actual_prices, label="Actual Gold Prices", color="blue")
    # plt.plot(predicted_time, predicted_prices, label="Predicted Gold Prices", color="red", linestyle="dashed")
    # plt.plot(future_time, future_predictions, label="Future Predicted Prices", color="green")
    # plt.title("Gold Price Prediction with Sentiment Adjustment Using LSTM")
    # plt.xlabel("Time")
    # plt.ylabel("Gold Price")
    # plt.legend()
    # plt.grid(True)

    # # Show the plot in Streamlit
    # st.pyplot(plt)

    # # Fetch and display gold-related news
    # news_response = requests.get(API_NEWS).json()
    # headlines = news_response["headlines"]
    # for i, headline in enumerate(headlines):
    #     st.write(f"{i + 1}. {headline}")
        
# Bank Investments Page
elif page == "Invest in Bank":
    st.title("Bank Investment Strategies")
    strategy = st.selectbox(
        "Select an Investment Strategy",
        ["Lump-Sum Investment", "SIP Investment", "Monte Carlo Simulation"],
    )
    years = st.number_input("Enter number of years:", min_value=1, value=10)

    if strategy == "Lump-Sum Investment":
        initial_investment = st.number_input("Enter initial investment:")
        annual_rate = st.number_input("Enter annual growth rate (decimal):")
        if st.button("Calculate"):
            response = requests.get(
                API_LUMP_SUM,
                params={"initial_investment": initial_investment, "annual_rate": annual_rate, "years": years},
            ).json()
            st.write(f"Future Value: **${response['future_value']:.2f}**")

    elif strategy == "SIP Investment":
        monthly_investment = st.number_input("Enter monthly investment:")
        annual_rate = st.number_input("Enter annual growth rate (decimal):")
        if st.button("Calculate"):
            response = requests.get(
                API_SIP,
                params={"monthly_investment": monthly_investment, "annual_rate": annual_rate, "years": years},
            ).json()
            st.write(f"Future Value: **${response['future_value']:.2f}**")

    elif strategy == "Monte Carlo Simulation":
        initial_investment = st.number_input("Enter initial investment:")
        annual_rate_mean = st.number_input("Enter mean growth rate (decimal):")
        annual_rate_std = st.number_input("Enter standard deviation (decimal):")
        simulations = st.number_input("Number of simulations:", min_value=100, value=1000)
        if st.button("Simulate"):
            response = requests.post(
                API_MONTE_CARLO,
                json={
                    "initial_investment": initial_investment,
                    "annual_rate_mean": annual_rate_mean,
                    "annual_rate_std": annual_rate_std,
                    "years": years,
                    "simulations": simulations,
                },
            ).json()
            results = response["results"]
            st.write(f"Mean Future Value: **${np.mean(results):,.2f}**")
            fig, ax = plt.subplots()
            ax.hist(results, bins=50, color="skyblue", edgecolor="black")
            st.pyplot(fig)

def calculate_savings(monthly_deposit, annual_rate, years):
    """
    Calculate the future value of savings in a Savings Account.
    """
    rate_per_month = annual_rate / 12
    months = years * 12
    total_savings = 0

    for i in range(months):
        total_savings += monthly_deposit * ((1 + rate_per_month) ** (months - i))
    
    return total_savings

# Streamlit App
investment_type = st.sidebar.selectbox(
    "Select Investment Type",
    ["Savings Account", "Fixed Deposit", "Recurring Deposit"]
)

def calculate_rd(monthly_investment, annual_rate, months):
    """
    Calculate Recurring Deposit Maturity Value.
    """
    rate_per_month = annual_rate / 12
    maturity_value = 0
    for i in range(months):
        maturity_value += monthly_investment * ((1 + rate_per_month) ** (months - i))
    return maturity_value

def calculate_fd(principal, annual_rate, years, compounding_frequency):
    """Calculate Fixed Deposit Maturity Value."""
    rate_per_period = annual_rate / compounding_frequency
    total_periods = years * compounding_frequency
    maturity_value = principal * ((1 + rate_per_period) ** total_periods)
    return maturity_value

if investment_type == "Savings Account":
        
    # Input fields for Savings Account
    st.header("Calculate Future Savings")

    monthly_deposit = st.number_input("Monthly Deposit ($):", min_value=0.0, step=10.0, value=100.0)
    annual_rate = st.number_input("Annual Interest Rate (%):", min_value=0.0, step=0.1, value=4.0) / 100
    years = st.number_input("Investment Period (Years):", min_value=1, step=1, value=5)

    # Calculation
    if st.button("Calculate Future Value"):
        future_value = calculate_savings(monthly_deposit, annual_rate, years)
        st.success(f"Future Value of Savings Account: ${future_value:,.2f}")
        
elif investment_type == "Fixed Deposit":

    # --- Streamlit App ---
    st.header("Fixed Deposit Calculator")

    # Input fields for Fixed Deposit
    principal = st.number_input("Principal Amount ($):", min_value=0.0, step=100.0, value=100000000.0)
    annual_rate = st.number_input("Annual Interest Rate (%):", min_value=0.0, step=0.1, value=5.0) / 100
    years = st.number_input("Investment Period (Years):", min_value=1, step=1, value=5)
    compounding_frequency = st.selectbox("Compounding Frequency:", ["Annually", "Semi-Annually", "Quarterly", "Monthly"])

    # Map compounding frequency to numeric value
    frequency_map = {
        "Annually": 1,
        "Semi-Annually": 2,
        "Quarterly": 4,
        "Monthly": 12
    }
    compounding_value = frequency_map[compounding_frequency]

    if st.button("Calculate Maturity Value"):
        maturity_value = calculate_fd(principal, annual_rate, years, compounding_value)
        st.success(f"Maturity Value of Fixed Deposit: ${maturity_value:,.2f}")
   
elif investment_type == "Recurring Deposit":
    
    monthly_investment = st.number_input("Monthly Investment ($):", min_value=0.0, step=10.0, value=100.0)
    annual_rate = st.number_input("Annual Interest Rate (%):", min_value=0.0, step=0.1, value=5.0) / 100
    years = st.number_input("Investment Period (Years):", min_value=1, step=1, value=5)
    
    st.header("Recurring Deposit Calculator")
    
    # Input fields for Recurring Deposit

    months = int(years * 12)
    
    if st.button("Calculate Maturity Value"):
        maturity_value = calculate_rd(monthly_investment, annual_rate, months)
        st.success(f"Maturity Value of Recurring Deposit: ${maturity_value:,.2f}")
 

    # monthly_deposit = st.number_input("Monthly Deposit ($)", min_value=0.0, value=500.0)
    # rate = st.number_input("Annual Interest Rate (%)", min_value=0.0, max_value=100.0, value=4.0) / 100 / 12
    # months = st.number_input("Term (Months)", min_value=1, value=24)
    # if st.button("Calculate"):
    #     result = calculate_recurring_deposit(monthly_deposit, rate, months)
    #     st.success(f"Total Balance after {months} month(s): ${result:.2f}")
