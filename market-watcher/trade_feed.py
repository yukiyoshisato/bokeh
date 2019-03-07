import pandas as pd
import ccxt


class TradeFeed:

    def __init__(self, exchange, symbol, seconds=None):
        self.exchange = exchange
        self.symbol = symbol
        self.df = self.get_trade_feed()
        if seconds is not None:
            self.remove_past_feeds(seconds)

    def get_trade_feed(self):
        trade_feed = self.exchange.fetch_trades(symbol=self.symbol, limit=1000)
        return pd.DataFrame(trade_feed, dtype=float)

    def add_trade_feeds(self, seconds=None):
        df = self.get_trade_feed().query("id not in ({})".format(list(self.df.loc[:, "id"].values))).copy(deep=True)
        self.df = df.append(self.df).reset_index(drop=True)
        if seconds is not None:
            self.remove_past_feeds(seconds)

    def remove_past_feeds(self, seconds):
        now = self.df.loc[:, "timestamp"].head(1).values[0]
        self.df = self.df.query("{} < timestamp".format(now - seconds)).copy(deep=True)


if __name__ == "__main__":

    def main():
        TradeFeed(ccxt.liquid(), "BTC/JPY").add_trade_feeds()

    main()
