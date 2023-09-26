import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


class Pricing(object):
    def __init__(self) -> None:
        pass
    
    def _load_historical_gas_price(self, input_gaspricecsv):
        # # #
        # input_gaspricecsv: csv that contains the commoditie's historical price columns {"Dates" : '%m/%d/%y', "Prices" : "1.09E+01"}
        # # #

        # load into a dataframe the gas price data from a CSV file
        df = pd.read_csv('input_gaspricecsv.csv', 
                         parse_dates=['Dates'],
                         date_parser=lambda x: datetime.strptime(x, '%m/%d/%y')
                        )
        
        # Rename column Dates to Date
        df = df.rename(columns = {"Dates" : "Date"})

        return df

    def estimate_gas_price(self, input_datestring):
        # # #
        # Input_date: date that corresponds to the wanted price, preference to be %Y-%m-%d
        # # #

        # Transform input date in a %Y-%m-%d striptime
        input_date = datetime.strptime(input_datestring, '%Y-%m-%d')
    
        # Set a end_date to the range of 365 dc
        end_date = input_date + timedelta(days=365)

        # Define a date_range that is defined Monthly
        date_range = pd.date_range(start=end_date - timedelta(days=365), end=end_date, freq='M')
    
        # Load historical data
        df = self.load_historical_gas_price()
        
        # Define a conditional Boolean to comparing Date and date_range and create a df only True matchs
        input_df = df[df['Date'].isin(date_range)]

        X = input_df.index.values
        y = input_df['Price']

        n = len(X)
        sum_x = sum(X)
        sum_y = sum(y)
        sum_xy = sum(X * y)
        sum_x_squared = sum(X**2)

        # Calculate the coefficients of the linear regression equation
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x**2)
        intercept = (sum_y - slope * sum_x) / n

        # Predict gas price for the future date
        future_x = len(df) + (input_date.year - df['Date'].iloc[-1].year) * 12 + (input_date.month - df['Date'].iloc[-1].month)
        future_price = intercept + slope * future_x

        return future_price

    def estimate_gas_price_np(self, input_datestring):
        
        input_date = datetime.strptime(input_datestring, '%Y-%m-%d')
        end_date = input_date + timedelta(days=365)
        date_range = pd.date_range(start=end_date - timedelta(days=365), end=end_date, freq='M')
        df = self.load_historical_gas_price()
        input_df = df[df['Date'].isin(date_range)]

        X = np.arange(len(input_df))
        y = input_df['Price'].values

        # Construir a matriz X adicionando uma coluna de uns para o termo constante
        X_matrix = np.column_stack((np.ones(len(X)), X))

        # Calcular os coeficientes da regressão linear usando álgebra de matrizes
        coeficientes = np.linalg.lstsq(X_matrix, y, rcond=None)[0]

        # Prever o preço do gás para a data futura
        future_x = len(df) + (input_date.year - df['Date'].iloc[-1].year) * 12 + (input_date.month - df['Date'].iloc[-1].month)
        future_price = coeficientes[0] + coeficientes[1] * future_x
        
        return future_price


if __name__ == "__main__":

    # Input date for prediction
    input_date = '2023-09-23'
    estimated_price = Pricing().estimate_gas_price(input_date)
    print(f"Estimated gas price for {input_date}: ${estimated_price:.2f}")

    # Plotting historical data and the estimated future price
    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df['Price'], label='Historical Prices', marker='o')
    plt.plot(pd.date_range(start=input_date, end=input_date + timedelta(days=365), freq='M'), [estimated_price] * 13, label='Estimated Future Prices', linestyle='--', marker='o')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Gas Price Analysis')
    plt.legend()
    plt.grid(True)
    plt.show()