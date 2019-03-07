import pandas as pd
from datetime import datetime
from ccxt.base import ExchangeError


class OrderBook:

    def __init__(self, exchange, symbol, threshold=None):
        self.exchange = exchange
        self.symbol = symbol
        self.threshold = threshold
        self.bids, self.asks = self.get_order_book()

    def get_order_book(self):
        order_book = self.exchange.fetch_order_book(symbol=self.symbol, params={"full": 1})
        columns = ["price", "size"]
        bids = pd.DataFrame(order_book["bids"], columns=columns, dtype=float)
        asks = pd.DataFrame(order_book["asks"], columns=columns, dtype=float)
        if self.threshold is not None:
            bids = bids.query("price > {}".format(bids.loc[0, "price"] - self.threshold)).copy(deep=True)
            asks = asks.query("price < {}".format(bids.loc[0, "price"] + self.threshold)).copy(deep=True)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        bids["date"] = now
        asks["date"] = now
        return bids, asks

    def add_order_book(self):
        self.bids, self.asks = self.get_order_book()

    def get_spread(self, size=None):
        if size is None:
            return self.asks.loc[0, "price"] - self.bids.loc[0, "price"]
        else:
            return self.get_ask_vwap(size) - self.get_bid_vwap(size)

    def get_bid_vwap(self, size):
        return self.get_vwap(self.bids, size)

    def get_ask_vwap(self, size):
        return self.get_vwap(self.asks, size)

    def get_vwap(self, df, size):
        """
        Get volume weighted average price
        :param df: bids or asks DataFrame
        :param size: target size for vwap
        :return: vwap per size
        """
        if df["size"].sum() < size:
            raise ExchangeError  # shortage in order book
        df["cumsum"] = df["size"].cumsum()
        df["diff"] = df["cumsum"] - size
        target_index = df[df["cumsum"] >= size].index.min()
        df = df.loc[0: target_index, :]
        df.loc[df["diff"] >= 0, "size"] = df["size"] - df["diff"]
        df["notional"] = df["price"] * df["size"]
        return df["notional"].sum() / size
