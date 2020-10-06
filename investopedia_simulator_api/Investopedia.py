from .investopedia_api import InvestopediaApi
import datetime

class InvestopediaHelper(object):
    def __init__(self, credentials):
        self.client = InvestopediaApi(credentials)
        print("signed in")
        self.portfolio = self.client.portfolio

    def get_quote(self, smbl):
        quote = self.client.get_stock_quote(smbl, )
        if quote is None:
            print(f"'{smbl}' is not supported by investopedia")
            return None
        return quote.__dict__

    def buy_stock(self, smbl, amount, limit):
        trade = self.client.StockTrade(symbol=smbl, quantity=amount, trade_type="buy")
        trade_info = trade.validate()
        trade.duration = "day_order"
        if trade.validated:
            trade.execute()
            return trade_info

    def sell_stock(self, smbl, amount, limit):
        trade = self.client.StockTrade(symbol=smbl, quantity=amount, trade_type="sell")
        trade_info = trade.validate()
        trade.duration = "day_order"
        if trade.validated:
            trade.execute()
            return trade_info


    def option_trade(self, smbl):
        lookup = self.client.get_option_chain(smbl)
        for chain in lookup.search_by_daterange(datetime.datetime.now(), datetime.datetime(2100, 1, 1)):
            print("--------------------------------")
            print("calls expiring on %s" % chain.expiration_date_str)
            for call in chain.calls:
                print(call)
            print("puts expiring on %s" % chain.expiration_date_str)
            for put in chain.puts:
                print(put)
            print("--------------------------------")

        option_contract = lookup.get(smbl)
        option_trade = self.client.OptionTrade(option_contract, 10, trade_type="buy to open")
        trade_info = None
        try:
            trade_info = option_trade.validate()
        except Exception as e:
            print(e)
        if option_trade.validated:
            print(trade_info)

    def test_buy(self, smbl):
        trade1 = self.client.StockTrade(symbol=smbl, quantity=5, trade_type="buy",
                                        order_type="market", duration="good_till_cancelled", send_email=True)
        trade_info = trade1.validate()
        print(trade_info)
        if trade1.validated:
            print("trade is validated")
