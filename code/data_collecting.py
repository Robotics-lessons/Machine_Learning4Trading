import pandas as pd
import os
import matplotlib.pyplot as plt

def test_run0():
	start_date = '2015-12-20'
	end_date = '2015-12-31'
	dates = pd.date_range(start_date, end_date)
	print(dates)
	df1 = pd.DataFrame(index=dates)
	print(df1)
	dfSPY = pd.read_csv('../data/SPY.csv', index_col='Date', 
		parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
#	print(dfSPY)
	df1 = df1.join(dfSPY, how='inner')
	df1 = df1.dropna()
#	print(df1)

	symbols = ['IBM', 'GOOG', 'GLD']
	for symbol in symbols:
		df_temp = pd.read_csv('../data/{}.csv'.format(symbol), index_col = 'Date'
				, parse_dates=True, usecols = ['Date', 'Adj Close'], na_values = ['nan'])

		df_temp = df_temp.rename(columns={'Adj Close':symbol})
		df1 = df1.join(df_temp)

	print(df1)  

def plot_selected(df, columns, start_index, end_index):
	pdf = df.ix[start_index:end_index, columns]  
	pdf = normalize_data(pdf)
	ax = pdf.plot(title='Stock Price')
	ax.set_xlabel('Date')
	ax.set_ylabel('Price')
	plt.show()

def normalize_data(df):
	return df / df.ix[0, :]

def symbol_to_path(symbol, base_path = '../data'):
	return os.path.join(base_path, "{}.csv".format(str(symbol)))	

def get_data(symbols, dates):
	df = pd.DataFrame(index=dates)
	for symbol in symbols:
		df_temp = pd.read_csv(symbol_to_path(symbol), index_col = 'Date'
			, parse_dates=True, usecols = ['Date', 'Adj Close'], na_values = ['nan'])
		df_temp = df_temp.rename(columns={'Adj Close':symbol})
		df = df.join(df_temp, how='inner')
	return df



def test_run(start_date, end_date, symbols):

	dates = pd.date_range(start_date, end_date)	
#	print(dates)
	
	if 'SPY' not in symbols:
		symbols.insert(0, 'SPY')
	df = get_data(symbols, dates)
	print(df.mean())
	print(df.median())
	print(df.std())
	return df


if __name__ == "__main__":
	start_date = '2015-01-01'
	end_date = '2015-12-31'
	symbols = ['IBM', 'GOOG', 'GLD','XOM']
	df = test_run(start_date, end_date, symbols)
	plot_selected(df, symbols, '2015-01-01', '2015-12-31')
#	print(df.ix['2015-02-04':'2015-02-20',['IBM','GLD']])
#	test_run0()

