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
import threading
import time
from GUI import GUI
from WSTrade import WSTrade
from forex_python.converter import CurrencyRates


today = datetime.date.today()
c = CurrencyRates()


def startup_stuff():
    warnings.filterwarnings("ignore")
    yf.pdr_override()
    register_matplotlib_converters()


def compare(vals):
    print(vals)
    highest_val = 0
    for v in vals:
        if v > highest_val:
            highest_val = v
    i = 0
    for v in vals:
        if v == highest_val:
            return i
        i += 1


class Investor3000(object):
    def __init__(self, credentials, index_file=None, save_file=None):
        self.client = WSTrade()
        self.client.login(credentials)
        self.symbol_scores = {}
        self.stocks_analysed = 0
        self.pop_symbols = []
        self.to_invest = []
        self.active_symbols = {}
        if index_file is None and save_file is None:
            print("Either specify <index_file.txt> (format: <TICKER\\t...\\nTICKER\\t...\\n etc.>) "
                  "or a previously generated save file (pickle'd)")
            exit(1)
        elif index_file is None and save_file is not None:
            self.index_file = None
            self.save_file = save_file
        elif index_file is not None and save_file is None:
            self.index_file = index_file
            self.save_file = index_file.split('.')[0] + ".pickle"
        elif index_file is not None and save_file is not None:
            self.index_file = index_file
            self.save_file = save_file

    def load_symbols_from_index(self):
        symbols = []
        with open(self.index_file, 'r') as f:
            for line in f.readlines():
                smbl = ""
                for l in line:
                    if l == '\t':
                        break
                    smbl += l
                symbols.append(smbl)
        symbols.pop(0)
        print(f"Loaded {len(symbols)} symbols from '{self.index_file}'")
        return symbols

    def load_symbols_from_saved(self):
        with open(self.save_file, "rb") as f:
            self.symbol_scores = pickle.loads(f.read())

    def set_symbol_info(self, symbol, df, tree_confidence, tree_prediction, lr_confidence, lr_prediction,
                        svr_confidence, svr_prediction):
        self.symbol_scores[symbol] = {"dataframe": df,
                                    "tree": {"confidence": tree_confidence,
                                             "prediction": tree_prediction},
                                    "lr": {"confidence": lr_confidence,
                                           "prediction": lr_prediction},
                                    "svr": {"confidence": svr_confidence,
                                            "prediction": svr_prediction}}

    def set_valid_symbols(self, symbols, start):
        for smbl in symbols:
            df = pdr.get_data_yahoo(smbl, start=start, end=datetime.date.today(), interval="1m", progress=False)
            if df.empty:
                continue
            self.set_symbol_info(smbl, df, None, None, None, None, None, None)
            print(f"\nStoring '{smbl}'")

    def analyse_stock(self, smbl, start, interval, future_unix):
        df = pdr.get_data_yahoo(smbl, start=start, end=datetime.date.today(), interval=interval, progress=False)
        if df.empty:
            print("Empty dataframe")
            return False
        df["Prediction"] = df[["Adj Close"]].shift(-future_unix)
        x = np.array(df.drop(["Prediction"], 1))[:-future_unix]
        if len(x) == 0:
            print("Not enough info in dataframe")
            return False
        # print("x =", x)
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

        self.set_symbol_info(smbl, df, tree_confidence, tree_prediction, lr_confidence, lr_prediction,
                             svr_confidence, svr_prediction)

        self.stocks_analysed += 1
        print(f"'{smbl}' added")
        return True

    def decide_buy(self, smbl):
            confidences = [self.symbol_scores[smbl]["tree"]["confidence"], self.symbol_scores[smbl]["lr"]["confidence"],
                           self.symbol_scores[smbl]["svr"]["confidence"]]
            if np.isnan(confidences).any():
                return None
            models = ["tree", "lr", "svr"]
            choice = compare(confidences)
            model = models[choice]
            print(f"Using {model} model for {smbl}")
            price = 0
            security = ""
            currency = ""
            info = {}
            security, price, currency = self.client.get_security_n_price(smbl)
            if price is None:
                self.pop_symbols.append(smbl)
                return None
            info["security"] = security
            info["price"] = price
            info["currency"] = currency
            difference = self.symbol_scores[smbl][model]["prediction"][-1] - price
            increase = round(difference / price * 100, 3)
            print(f"Predicting an increase of {increase}% in the next 30 minutes for {smbl}")
            if increase > 1.5:
                info["increase"] = increase
                return info
            else:
                return None


    def invest(self, smbl):
        print("New thread for symbol", smbl)
        info = self.decide_buy(smbl)
        if info is None:
            return
        price = info["price"]
        increase = info["increase"]
        buying_power = 0
        buying_power = self.client.get_buying_power()
        if info["currency"] == "USD":
            c.get_rates("USD")
            price *= c["CAD"]
        buy_amount = float(buying_power) / price
        buy_amount /= price
        buy_amount *= (increase * 2)
        buy_amount /= price
        print(f"buy amount for {smbl} : {buy_amount}")
        buy_amount = int(buy_amount)
        print(f"Buying {buy_amount} shares of {smbl}")
        # limit = round(price * 1.01, 3)
        print("Attempting buy for", smbl)
        trade_info = 0
        trade_info = self.client.place_order(info["security"], None, buy_amount, "buy")
        print(trade_info)
        time.sleep(1800)
        trade_info = self.client.place_order(info["security"], None, buy_amount, "sell")
        print(info)


    def main(self):
        start = today - BDay(3)
        future_unix = 30
        if self.index_file is not None:
            symbols = self.load_symbols_from_index()
            self.set_valid_symbols(symbols, start)
        else:
            self.load_symbols_from_saved()

        # for smbl in self.symbol_scores.keys():
        #     try:
        #         success = self.analyse_stock(smbl, start, "1m", future_unix)
        #         if not success:
        #             self.pop_symbols.append(smbl)
        #             print(f"'{smbl}' failed, gonna get popped")
        #     except Exception as e:
        #         print("Error on '{}':'{}' -> gonna get popped".format(smbl, e))
        #         self.pop_symbols.append(smbl)
        # for smbl in self.pop_symbols:
        #     self.symbol_scores.pop(smbl)

        print(f"Analysed {self.stocks_analysed} symbols successfully")
        print("Saving symbols to '{}'".format(self.save_file))
        with open(self.save_file, "wb") as f:
            f.write(pickle.dumps(self.symbol_scores))
        while True:
            print(dedent("""
            1 - Auto invest
            2 - Manual invest
            3 - Quit
            """))

            choice = input("> ")
            if choice == '1':
                print(len(self.symbol_scores))
                try:
                    for smbl in self.symbol_scores.keys():
                        thread = threading.Thread(target=self.invest, args=(smbl,))
                        thread.start()
                except Exception as e:
                    print(f"error for {smbl} : {e}")
            elif choice == '2':
                pass
                # self.manual_invest
            elif choice == '3':
                return
            else:
                print("Invalid input")


def simulator():
    startup_stuff()
    muney = Investor3000({"username": "georgevarahidis@gmail.com", "password": "WelcomeEleven11"}, save_file="TSX.pickle")
    muney.main()
    
    
def trade():
    


def gui():
   pass 


if __name__ == "__main__":
    simulator()
