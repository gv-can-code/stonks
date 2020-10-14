import requests as req
import json


class WSTrade(object):
    def __init__(self):
        self.refresh_token = None
        self.auth_header = {"Authorization" : None}
        self.account_list = None
        self.hist_data = None
        self.orders = None

    def update_tokens(self, r):
        access_token = r.headers["X-Access-Token"]
        if access_token == None:
            print("ERROR GETTING ACCESS TOKEN")
            exit(1)
        self.auth_header["Authorization"] = access_token

    def refresh(self):
        refresh = {"refresh_token" : self.refresh_token}
        self.update_tokens(req.post("https://trade-service.wealthsimple.com/auth/refresh", refresh))

    def login(self, credentials):
        r = req.post("https://trade-service.wealthsimple.com/auth/login", credentials)
        self.update_tokens(r)

    def get_security_n_price(self, symbol):
        r = req.get("https://trade-service.wealthsimple.com/securities?query="
                    + symbol, headers=self.auth_header)
        text = json.loads(r.text)
        if len(text["results"]) == 0:
            return (None, None, None)
        security = json.loads(r.text)["results"][-1]["id"]
        r = req.get("https://trade-service.wealthsimple.com/securities/" + security, headers=self.auth_header)
        text = json.loads(r.text)
        # print(text)
        price = text["quote"]["ask"]
        currency = text["quote"]["currency"]
        return (security, float(price), currency)

    def get_buying_power(self):
        r = req.get("https://trade-service.wealthsimple.com/account/list", headers=self.auth_header)
        return float(json.loads(r.text)["results"][0]["buying_power"]["amount"])


    def place_order(self, security, limit, quantity, buy_or_sell):
        content = {
            "security_id" : security,
            "limit_price" : limit,
            "quantity" : quantity,
            "order_type" : buy_or_sell + "_quantity",
            "order_sub_type" : "market",
            "time_in_force" : "day",
            "api_paste_format" : "python"
        }
        r = req.post("https://trade-service.wealthsimple.com/orders", json=content, headers=self.auth_header)
        return r.text

