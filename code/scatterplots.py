import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

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
#	pdf = df.ix[start_index:end_index, columns]  
#	pdf = normalize_data(pdf)
	
	pdf = df['SPY']
	
	rm_SPY = get_rolling_mean(pdf)
	rstd_SPY = get_rolling_std(pdf)
	upper_band, lower_band = get_bollinger_bands(rm_SPY, rstd_SPY)
	ax = pdf.plot(title='SPY rolling mean', label='SPY')

	rm_SPY.plot(label='Rolling mean', ax=ax)
	upper_band.plot(label='upper band', ax=ax)
	lower_band.plot(label='lower band', ax=ax)
	ax.set_xlabel('Date')
	ax.set_ylabel('Price')
	ax.legend(loc='upper left')
	plt.show()

def computer_daily_returns(df):
	daily_returns = df.copy()
	daily_returns[1:] = (daily_returns[1:] / daily_returns[:-1].values) - 1
	daily_returns.ix[0,:] = 0
	return daily_returns

def get_rolling_mean(df, window=20):
	return pd.rolling_mean(df, window=window)

def get_rolling_std(df, window=20):
	return pd.rolling_std(df, window=window)

def get_bollinger_bands(rmdf, rstddf):
	upper_band = rmdf + rstddf * 2
	lower_band = rmdf - rstddf * 2
	return upper_band, lower_band

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
#	print(df.mean())
#	print(df.median())
#	print(df.std())
	return df


def show_histograms(df, symbols):
	daily_returns = computer_daily_returns(df)
#	ax = daily_returns.plot(title='Histograms', label='SPY')
#	ax.set_xlabel('SPY')
	for symbol in symbols:
		daily_returns[symbol].hist(bins=20, label=symbol)
#		ax.set_ylabel(symbol)

#	ax.set_xlabel('SPY')
	
#	ax.legend(loc='upper left')
	plt.show()

def show_scatterplots(df, symbols):
	daily_returns = computer_daily_returns(df)
#	ax = daily_returns.plot(title='Histograms', label='SPY')
#	ax.set_xlabel('SPY')
	for symbol in symbols[1:]:
		daily_returns.plot(kind='scatter', x=symbols[0], y=symbol)
		beta, alpha = np.polyfit(daily_returns[symbols[0]], daily_returns[symbol], 1)
		print('beta = ', beta)
		print('alpha = ', alpha)

		plt.plot(daily_returns[symbols[0]], beta*daily_returns[symbols[0]] + alpha, '-', color='r')
		plt.show()
#		ax.set_ylabel(symbol)

#	ax.set_xlabel('SPY')
	print('correlation coefficient: \n', daily_returns.corr(method='pearson'))
#	ax.legend(loc='upper left')


if __name__ == "__main__":
	start_date = '2000-01-01'
	end_date = '2015-12-31'
	symbols = ['XOM','GLD'] # ['IBM', 'GOOG', 'GLD','XOM']
	df = test_run(start_date, end_date, symbols)
	show_scatterplots(df, symbols)
#	plot_selected(df, symbols, '2000-01-01', '2015-12-31')
#	print(df.ix['2015-02-04':'2015-02-20',['IBM','GLD']])
#	test_run0()

