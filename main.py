import yfinance as yf
import pandas as pd
from pandas_datareader import data as pdr
from pandas.tseries.offsets import BDay
import datetime
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import pickle
import threading
import time
from WSTrade import WSTrade
from forex_python.converter import CurrencyRates
import warnings


today = datetime.date.today()
start = today - BDay(3)
c = CurrencyRates()
future_unix = 30
yf.pdr_override()
warnings.filterwarnings("ignore")


def compare(vals):
    highest_val = 0
    for v in vals:
        if v > highest_val and v < 1:
            highest_val = v
    if highest_val < 0.9:
        return None
    i = 0
    for v in vals:
        if v == highest_val:
            return i
        i += 1


class Investor3000(object):
    def __init__(self, credentials, index_file=None, save_file=None):
        if index_file is None and save_file is None:
            print("No index or save file specified")
            exit(1)
        elif index_file is None and save_file is not None:
            self.index_file = None
            self.save_file = save_file
        elif index_file is not None:
            self.index_file = index_file
            self.save_file = index_file.split('.')[0] + ".pickle"
        self.client = WSTrade()
        self.client.login(credentials)
        self.buying_power = self.client.get_buying_power()
        self.tickers = {}

    def get_ticker_data(self, ticker_list):
        data = pdr.get_data_yahoo(tickers=ticker_list, start=start, end=today, interval="1m",
                                  threads=True, auto_adjust=False, prepost=False, group_by="ticker", proxy=None)
        dataframes = data.T
        for ticker in ticker_list:
            try:
                df = dataframes.loc[(ticker,),].T
                df = df.dropna()
                if df.empty:
                    continue
                self.set_ticker_info(ticker, df, None, None, None, None, None, None)
            except KeyError:
                continue

    def set_ticker_info(self, ticker, df, tree_confidence, tree_prediction, lr_confidence, lr_prediction,
                        svr_confidence, svr_prediction):
        self.tickers[ticker] = {"dataframe": df,
                                      "tree": {"confidence": tree_confidence,
                                               "prediction": tree_prediction},
                                      "lr": {"confidence": lr_confidence,
                                             "prediction": lr_prediction},
                                      "svr": {"confidence": svr_confidence,
                                              "prediction": svr_prediction}}

    def load_tickers_from_index(self):
        tickers = []
        with open(self.index_file, 'r') as f:
            for line in f.readlines():
                smbl = ""
                for l in line:
                    if l == '\t':
                        break
                    smbl += l
                tickers.append(smbl)
        tickers.pop(0)
        return tickers

    def load_symbols_from_saved(self):
        with open(self.save_file, "rb") as f:
            self.tickers = pickle.loads(f.read())

    def save(self):
        with open(self.save_file, "wb") as f:
            f.write(pickle.dumps(self.tickers))

    def analyse_stock(self, ticker):
        df = self.tickers[ticker]["dataframe"]
        if df.empty:
            return False
        # df = df.dropna()
        df["Prediction"] = df[["Adj Close"]].shift(-future_unix)
        x = np.array(df.drop(["Prediction"], 1))[:-future_unix]
        if len(x) == 0:
            # Not enough info in dataframe
            return False
        y = np.array(df["Prediction"])[:-future_unix]
        # print("y =", y)
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

        self.set_ticker_info(ticker, df, tree_confidence, tree_prediction, lr_confidence, lr_prediction,
                             svr_confidence, svr_prediction)
        return True

    def decide_buy(self, ticker):
        confidences = [self.tickers[ticker]["tree"]["confidence"], self.tickers[ticker]["lr"]["confidence"],
                       self.tickers[ticker]["svr"]["confidence"]]
        if pd.isnull(confidences).any():
            return None
        models = ["tree", "lr", "svr"]
        choice = compare(confidences)
        if choice is None:
            return None
        model = models[choice]
        security, price, currency = self.client.get_security_n_price(ticker)
        if price is None or price < 1:
            return None
        if currency == "USD":
            price = c.convert("USD", "CAD", price)
        increase = round(float(self.tickers[ticker][model]["prediction"][-1]), 3) - price
        increase = round((increase / price) * 100, 3)
        if increase > 2.5:
            return {"security" : security, "price" : price, "increase" : increase}

    def get_quantity(self, price, increase):
        can_spend = increase  # * 5
        can_spend /= 100
        if can_spend > 100:
            can_spend = 100
        can_buy = self.buying_power / price
        if can_buy < 1:
            return 0
        will_buy = can_buy * can_spend
        if will_buy * price > self.buying_power / 2:
            return 0
        return int(will_buy)

    def invest(self, ticker):
        if not self.analyse_stock(ticker):
            return
        db = self.decide_buy(ticker)
        if db is None:
            return
        quantity = self.get_quantity(db["price"], db["increase"])
        if quantity == 0:
            return
        db["quantity"] = quantity
        self.tickers[ticker]["info"] = db
        # limit = round(db["price"] * 1.01, 3)
        # trade_info = self.client.place_order(db["security"], limit, quantity, "buy")
        # if "error" in trade_info:
        #     print(f"Error for {ticker}:{trade_info}")
        #     return
        # self.buying_power = self.client.get_buying_power()
        # time.sleep(future_unix * 60)
        # trade_info = self.client.place_order(db["security"], limit, quantity, "sell")
        # print(trade_info)

    def main(self):
        tickers = []
        if self.index_file is not None:
            tickers = self.load_tickers_from_index()
        else:
            self.load_symbols_from_saved()
            for ticker in self.tickers.keys():
                tickers.append(ticker)
        self.get_ticker_data(tickers)
        self.save()
        for ticker in self.tickers.keys():
            thread = threading.Thread(target=self.invest, args=(ticker,))
            thread.start()
        for ticker in self.tickers.keys():
            if not "info" in self.tickers[ticker]:
                continue
            if self.tickers[ticker]["info"]["increase"] > 2.5:
                print(f"Buy {ticker} at {self.tickers[ticker]['info']['price']} with predicted increase of "
                      f"{self.tickers[ticker]['info']['increase']}?")
                choice = input("y/n\n> ")
                if choice == 'y':
                    quantity = self.tickers[ticker]["info"]["quantity"]
                    print("Recommended quantity :", quantity)
                    print("Press enter to use this quantity, 'q' to cancel or enter your own amount")
                    choice = input("> ")
                    try:
                        quantity = int(choice)
                    except ValueError:
                        if choice == 'q':
                            continue
                    trade_info = self.client.place_order(self.tickers[ticker]["info"]["security"],
                                                         self.tickers[ticker]["info"]["price"] * 1.01, quantity, "buy")
                    print(trade_info)
                    if not "error" in trade_info:
                        self.buying_power = self.client.get_buying_power()






investor = Investor3000({"email" : "kandiotisa@gmail.com", "password" : "BlackList144"}, index_file="TSX.txt")
investor.main()
print(investor.tickers)