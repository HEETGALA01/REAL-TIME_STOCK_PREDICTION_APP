# Real-Time Stock Price Prediction Using LSTM & GRU

## **1. Introduction**
Stock market prediction is a crucial application of machine learning that helps investors and analysts make informed decisions. This project leverages **Long Short-Term Memory (LSTM)** and **Gated Recurrent Units (GRU)** neural networks to predict stock prices in real-time using historical stock data. The application is built using **Streamlit** for an interactive user experience.

---

## **2. What Are LSTM & GRU?**
### **LSTM (Long Short-Term Memory)**
LSTM is a type of recurrent neural network (RNN) designed to handle long-term dependencies in time-series data by using memory cells to retain important information.

### **GRU (Gated Recurrent Unit)**
GRU is a simplified version of LSTM that uses fewer parameters while achieving similar performance. It controls memory updates through reset and update gates.

### **Equation for LSTM & GRU Predictions:**
#### LSTM:
```math
h_t = \sigma(W_hx_t + U_hh_{t-1} + b_h)
```

#### GRU:
```math
z_t = \sigma(W_zx_t + U_zh_{t-1} + b_z)
```

where:
- `x_t` = Input at time step `t`
- `h_t` = Hidden state
- `z_t` = Update gate (for GRU)
- `W, U, b` = Learnable parameters

---

## **3. Dataset for Stock Price Prediction**
This project utilizes the **Yahoo Finance API (yfinance)** to fetch real-time stock market data. The dataset includes features such as:

- **Open Price** – The price at the beginning of the trading session.
- **Close Price** – The price at the end of the trading session.
- **High & Low Prices** – The highest and lowest price during the trading session.
- **Volume** – The number of shares traded.

---

## **4. Steps for Stock Price Prediction Using LSTM & GRU**
### **Step 1: Data Preprocessing**
- Fetch stock data using `yfinance`.
- Scale the data using `MinMaxScaler`.
- Prepare the dataset for time-series prediction.

### **Step 2: Train the Models**
- Load pre-trained **LSTM and GRU models** (`lstm_model.h5` and `gru_model.keras`).
- Train on historical stock price data.

### **Step 3: Make Predictions**
- Predict stock prices using **both LSTM and GRU models**.
- Visualize predictions using `plotly`.

### **Step 4: Evaluate Performance**
- Compare predicted vs. actual stock prices.
- Use **Mean Squared Error (MSE)** for evaluation.

---

## **5. Why Use LSTM & GRU for Stock Prediction?**
- **Captures Time-Series Patterns:** Remembers long-term dependencies in financial data.
- **Handles Volatility:** Works well with fluctuating stock prices.
- **Performs Well on Sequential Data:** Unlike traditional ML models, RNNs (LSTM/GRU) can learn patterns from sequences of data.

---

## **6. Limitations of LSTM & GRU in Stock Prediction**
- **Market Unpredictability:** Stock prices depend on unpredictable factors like news, global events, and politics.
- **Computational Complexity:** Training deep learning models is resource-intensive.
- **Data Sensitivity:** Performance depends heavily on data quality and feature selection.

---

## **7. Installation and Setup**
### **Prerequisites**
Ensure you have **Python 3.x** installed.

### **Installation Steps**
```bash
git clone https://github.com/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
```

### **Run the Application**
```bash
streamlit run app.py
```

---

## **8. Usage**
- Enter a **stock ticker symbol** (e.g., `AAPL`, `TSLA`).
- View **real-time stock data** and technical indicators.
- Compare **LSTM vs. GRU** predictions.
- Analyze **future stock price trends**.

---

## **9. Future Enhancements**
- **Integrate More Machine Learning Models** – Compare CNN, Transformers, and Reinforcement Learning.
- **Deploy as a Web App** – Host the application using AWS/GCP.
- **Enhance Prediction Accuracy** – Incorporate external factors like news sentiment analysis.
- **Develop an Alert System** – Notify users of market trends and signals.

---
