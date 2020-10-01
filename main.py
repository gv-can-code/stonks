import yfinance as yf
from pandas_datareader import data as pdr
import pandas as pd
from pandas.tseries.offsets import BDay
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mpl
import datetime
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from pandas.plotting import register_matplotlib_converters
from matplotlib.dates import DateFormatter
import warnings
from investopedia_simulator_api.Investopedia import InvestopediaHelper
from textwrap import dedent
import pickle


today = datetime.date.today()


def startup_stuff():
    warnings.filterwarnings("ignore")
    yf.pdr_override()
    register_matplotlib_converters()


class Investor3000(object):
    def __init__(self, credentials, symbols=None, index_file=None, save_file=None):
        self.client = InvestopediaHelper(credentials)
        self.symbols = symbols
        self.symbol_scores = {}
        self.stocks_analysed = 0
        if index_file is None:
            self.index_file = INDEX.txt
        else:
            self.index_file = index_file
        if save_file is None:
            self.saved = "saved.pic"
        else:
            self.saved = save_file

    def load_symbols_from_index(self):
        self.symbols = []
        with open(self.index_file, 'r') as f:
            for line in f.readlines():
                smbl = ""
                for l in line:
                    if l == '\t':
                        break
                    smbl += l
                self.symbols.append(smbl)
        self.symbols.pop(0)
        print(f"Loaded {len(self.symbols)} symbols from '{file}'")

    def print_account(self):
        print(dedent("""
            {}
            """).format(self.client.portfolio))

    def analyse_stock(self, smbl, start, future_unix):
        df = pdr.get_data_yahoo(smbl, start=start, end=datetime.date.today(), interval="1m", progress=False)
        if df.empty:
            return 1
        print(df)
        df["Prediction"] = df[["Adj Close"]].shift(-future_unix)
        x = np.array(df.drop(["Prediction"], 1))[:-future_unix]
        if len(x) == 0:
            return
        print("x =", x)
        y = np.array(df["Prediction"])[:-future_unix]
        print("y =", y)
        x_train, x_test, y_train, y_test = train_test_split(x, y)

        x_future = df.drop(["Prediction"], 1)[:-future_unix]
        x_future = x_future.tail(future_unix)
        x_future = np.array(x_future)

        tree = DecisionTreeRegressor().fit(x_train, y_train)
        tree_prediction = tree.predict(x_future)
        tree_confidence = tree.score(x_train, y_train)

        lr = LinearRegression().fit(x_train, y_train)
        lr_prediction = lr.predict(x_future)
        lr_confidence = lr.score(x_test, y_test)

        svr_rbf = SVR(kernel="rbf", C=1e3, gamma=0.1)
        svr_rbf.fit(x_train, y_train)
        svr_prediction = svr_rbf.predict(x_future)
        svr_confidence = svr_rbf.score(x_test, y_test)

        self.symbol_scores[smbl] = {"tree": {"confidence": tree_confidence,
                                          "prediction": tree_prediction},
                                 "lr": {"confidence": lr_confidence,
                                        "prediction": lr_prediction},
                                 "svr": {"confidence": svr_confidence,
                                         "prediction": svr_prediction}}
        self.stocks_analysed += 1
        return 0

    def main(self):
        if self.symbols is None:
            self.load_symbols()
        start = today - BDay(3)
        future_unix = 30
        i = 0
        for smbl in self.symbols:
            try:
                success = self.analyse_stock(smbl, start, future_unix)
                if success == 1:
                    self.symbols.pop(i)
                i += 1
            except Exception as e:
                print("Error on '{}':{}".format(smbl, e))
                self.symbols.pop(i)
                i += 1

        print(self.symbol_scores)
        print(f"Analysed {len(self.symbols)}/{self.stocks_analysed} symbols")
        print("Saving symbols to '{}'".format(self.saved))
        with open(self.saved, "wb") as f:
            f.write(pickle.dumps(self.symbol_scores))



startup_stuff()
muney = Investor3000({"username" : "kandiotisa@gmail.com", "password" : "BlackList144"})
muney.main()
"""
print("Enter stock symbol:")
smbl = input("> ")

start = datetime.date.today() - BDay(5)
end = datetime.date.today() - BDay(1)
df = pdr.get_data_yahoo(smbl, start=start, end=end, interval="1m", progress=False)
# df = df.drop(["Open", "High", "Low", "Close", "Volume"], axis=1)
print(df.head(6))
# df.plot()
# plt.show()

future_unix = 60
df["Prediction"] = df[["Adj Close"]].shift(-future_unix)

print(df.tail(4))

x = np.array(df.drop(["Prediction"], 1))[:-future_unix]
y = np.array(df["Prediction"])[:-future_unix]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

tree = DecisionTreeRegressor().fit(x_train, y_train)
lr = LinearRegression().fit(x_train, y_train)

x_future = df.drop(["Prediction"], 1)[:-future_unix]
x_future = x_future.tail(future_unix)
x_future = np.array(x_future)

tree_prediction = tree.predict(x_future)
tree_confidence = tree.score(x_train, y_train)
lr_prediction = lr.predict(x_future)
lr_confidence = lr.score(x_test, y_test)

print("tree confidence: {}\nlinear regression confidence: {}".format(tree_confidence, lr_confidence))

plt.figure(figsize=(16,8))

higher = ""

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.01)
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_rbf.fit(x_train, y_train)

svm_confidence = svr_rbf.score(x_test, y_test)

if tree_confidence > 0.7 and tree_confidence > lr_confidence:
    higher = "tree"
    print("Using tree for predictions")
    predictions = tree_prediction
    valid = df[x.shape[0]:]
    valid["Predictions"] = predictions
    plt.plot(df["Adj Close"])
    plt.plot(valid[["Adj Close", "Predictions"]])
    plt.legend(["Train", "Val", "Tree Prediction"], loc="lower right")
    # plt.show()
elif lr_confidence > 0.7 and lr_confidence > tree_confidence:
    higher = "lr"
    predictions = lr_prediction
    valid = df[x.shape[0]:]
    valid["Predictions"] = predictions
    plt.plot(df["Adj Close"])
    plt.plot(valid[["Adj Close", "Predictions"]])
    plt.legend(["Train", "Val", "LR Prediction"], loc="lower right")
    # plt.show()
else:
    print("No confidence in {}, not investing".format(smbl))
    exit()

set = None
if higher == "tree":
    set = tree_prediction
else:
    set = lr_prediction

last_val = set[-1]

adj_close_avg = 0
i = 0
for n in df["Adj Close"]:
    adj_close_avg += n
    i += 1

adj_close_avg /= i

print("Last Adj Close = {}\nprediction val = {}".format(adj_close_avg, last_val))

difference = last_val - adj_close_avg
increase = difference / adj_close_avg * 100
print(f"Increased by {increase:.3f}%")
print("Show graph? y/n")
if input("> ") == 'y':
    plt.show()

print("Invest? y/n")
if input("> ") == 'y':
    credentials = {"username" : "kandiotisa@gmail.com", "password" : "BlackList144"}
    client = InvestopediaApi(credentials)
    p = client.portfolio
    print("Account value:", p.account_value)
    print("Cash:", p.cash)
    print("Buying power:", p.buying_power)

    quote = client.get_stock_quote(smbl)
    print(quote.__dict__)

    trade_type = 'buy'
    limit_1000 = 'limit 1000'

    print("How much do you want to invest?")
    amount = input("> ")

    trade = client.StockTrade(smbl, int(amount), trade_type, order_type=limit_1000)
    trade_info = trade.validate()
    if trade.validated:
        print(trade_info)
        trade.execute()
else:
    print('bye')




"""
