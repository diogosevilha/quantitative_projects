import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import pandas
import numpy


data = {
    'Date': pd.date_range(start='2020-10-31', end='2024-09-30', freq='M'),
    'Price': [10.2, 11.1, 12.5, 13.2, 14.0, 15.3, 16.2, 17.5, 18.0, 19.2, 20.1, 21.4, 22.0, 23.2, 24.1, 25.3, 26.5, 27.4, 28.2, 29.0, 30.2, 31.0, 32.4, 33.1]
}

df = pd.DataFrame(data)

def estimate_gas_price(input_date):
    # Convert the input date to a datetime object
    input_date = datetime.strptime(input_date, '%Y-%m-%d')
    
    # Create a DataFrame with the given input date and previous 12 months
    end_date = input_date + timedelta(days=365)
    date_range = pd.date_range(start=end_date - timedelta(days=365), end=end_date, freq='M')
    input_df = df[df['Date'].isin(date_range)]

    # Perform linear regression
    X = input_df.index.values.reshape(-1, 1)
    y = input_df['Price']
    model = LinearRegression()
    model.fit(X, y)

    # Predict gas price for the future date
    future_date = pd.date_range(start=input_date, end=input_date + timedelta(days=365), freq='M')
    future_x = len(input_df) + len(future_date) - 1
    future_price = model.predict([[future_x]])[0]
    
    return future_price