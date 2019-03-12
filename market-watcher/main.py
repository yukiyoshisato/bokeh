import pandas as pd
from math import pi
from datetime import datetime, timedelta, timezone
import visualization as v
from bokeh.models import Slider, ColumnDataSource, Button, WidgetBox, RangeSlider
from bokeh.layouts import layout, Row, Column
from bokeh.io import curdoc


# data source
df_trade_feed = pd.read_csv("./data/trade_feed/trade_feed_2019-03-03_14-00-08.csv").drop(["id"], axis=1)
df_trade_feed["created_at"] = (df_trade_feed["created_at"] // 5) * 5  # grouping by 5 seconds
df_trade_feed["created_at"] = pd.to_datetime(df_trade_feed["created_at"], unit="s")
df_trade_feed["notional"] = df_trade_feed["price"] * df_trade_feed["quantity"]
df_trade_feed = df_trade_feed.groupby(["created_at", "taker_side"]).sum()
df_trade_feed["price"] = df_trade_feed["notional"] / df_trade_feed["quantity"]
df_trade_feed.reset_index(inplace=True)


def get_df_trade_feed(side):
    return df_trade_feed.query("taker_side == '{}'".format(side)).copy(deep=True)


def get_df_order_book(side):
    df_order_book = pd.read_csv("./data/order_book/order_book_{}_2019-03-03_14-00-08.csv".format(side))
    df_order_book["created_at"] = pd.to_datetime(df_order_book["created_at"], unit="ns").dt.floor("S")
    df_order_book["price"] = (df_order_book["price"] // 20) * 20  # grouping by 20 JPY
    df_order_book = df_order_book.groupby(["created_at", "price"]).sum().reset_index()
    return df_order_book


def get_df_hist(df_trade_feed_buy, df_trade_feed_sell_neg):
    df_trade_feed_hist = df_trade_feed_buy.append(df_trade_feed_sell_neg)
    df_trade_feed_hist["quantity"] = (df_trade_feed_hist["quantity"] * 10 // 1) * 1 * 0.1  # bins = 0.2BTC
    df_trade_feed_hist_groupby = df_trade_feed_hist.groupby("quantity").count().reset_index()
    df_trade_feed_hist_groupby.drop(["taker_side", "price", "notional"], axis=1, inplace=True)
    df_trade_feed_hist_groupby.columns = ["quantity", "count"]
    return df_trade_feed_hist.merge(df_trade_feed_hist_groupby, how="left", on="quantity")


def get_df_trade_feed_side(df_trade_feed):
    df_trade_feed_side = df_trade_feed.groupby("taker_side").sum()
    df_trade_feed_side["rate"] = df_trade_feed_side["quantity"] / df_trade_feed_side["quantity"].sum()
    df_trade_feed_side.sort_index(inplace=True)
    return df_trade_feed_side


df_trade_feed_buy = get_df_trade_feed(side="buy")
df_trade_feed_sell = get_df_trade_feed(side="sell")
df_trade_feed_sell_neg = df_trade_feed_sell.copy(deep=True)
df_trade_feed_sell_neg["quantity"] = df_trade_feed_sell_neg["quantity"] * -1
df_trade_feed_hist = get_df_hist(df_trade_feed_buy, df_trade_feed_sell_neg)
df_order_book_buy = get_df_order_book(side="buy")
df_order_book_sell = get_df_order_book(side="sell")
df_order_book_sell["quantity"] = df_order_book_sell["quantity"] * -1
df_trade_feed_side = get_df_trade_feed_side(df_trade_feed)

source_trade_feed_buy = ColumnDataSource(data=dict(
    created_at=df_trade_feed_buy["created_at"],
    price=df_trade_feed_buy["price"],
    quantity=df_trade_feed_buy["quantity"],
))

source_trade_feed_sell = ColumnDataSource(data=dict(
    created_at=df_trade_feed_sell["created_at"],
    price=df_trade_feed_sell["price"],
    quantity=df_trade_feed_sell["quantity"],
))

source_trade_feed_center = ColumnDataSource(data=dict(
    x0=[datetime(2019, 3, 3, 14, 30, 0, tzinfo=timezone.utc).timestamp() * 1000],
    y0=[df_trade_feed["price"].min()],
    x1=[datetime(2019, 3, 3, 14, 30, 0, tzinfo=timezone.utc).timestamp() * 1000],
    y1=[df_trade_feed["price"].max()],
))

source_trade_feed_distribution_buy = ColumnDataSource(data=dict(
    created_at=df_trade_feed_buy["created_at"],
    price=df_trade_feed_buy["price"],
    quantity=df_trade_feed_buy["quantity"],
))

source_trade_feed_distribution_sell = ColumnDataSource(data=dict(
    created_at=df_trade_feed_sell["created_at"],
    price=df_trade_feed_sell["price"],
    quantity=df_trade_feed_sell_neg["quantity"],
))

source_trade_feed_side = ColumnDataSource(data=dict(
    start_angle=[pi / 2, df_trade_feed_side.tail(1)["rate"].values[0] * 2 * pi + pi / 2],
    end_angle=[df_trade_feed_side.tail(1)["rate"].values[0] * 2 * pi + pi / 2, pi / 2],
    color=["red", "green"],
    side=["sell", "buy"],
))

source_trade_feed_hist_buy = ColumnDataSource(data=dict(
    quantity=df_trade_feed_hist.query("taker_side == 'buy'")["quantity"],
    count=df_trade_feed_hist.query("taker_side == 'buy'")["count"],
))

source_trade_feed_hist_sell = ColumnDataSource(data=dict(
    quantity=df_trade_feed_hist.query("taker_side == 'sell'")["quantity"],
    count=df_trade_feed_hist.query("taker_side == 'sell'")["count"],
))

source_order_book_buy = ColumnDataSource(data=dict(
    price=df_order_book_buy["price"],
    quantity=df_order_book_buy["quantity"],
))

source_order_book_sell = ColumnDataSource(data=dict(
    price=df_order_book_sell["price"],
    quantity=df_order_book_sell["quantity"],
))


# plotting
p_trade_feed = v.plot_trade_feed(source_trade_feed_buy, source_trade_feed_sell, source_trade_feed_center)
p_candlestick = v.plot_candlestick()
p_order_book = v.plot_order_book(source_order_book_buy, source_order_book_sell)
p_trade_feed_distribution = v.plot_trade_feed_distribution(source_trade_feed_distribution_buy, source_trade_feed_distribution_sell)
p_trade_feed_hist = v.plot_trade_feed_histogram(source_trade_feed_hist_buy, source_trade_feed_hist_sell)
p_buy_sell_pie = v.plot_side_trade_feed(source_trade_feed_side)

# link plots
p_trade_feed.y_range = p_order_book.y_range
p_trade_feed_distribution.y_range = p_trade_feed.y_range
p_candlestick.x_range = p_trade_feed.x_range
p_candlestick.y_range = p_trade_feed_distribution.y_range

start = 0
end = 0


def update_start(new):
    global start
    start = datetime.utcfromtimestamp(new * 0.001)


def update_end(new):
    global end
    end = datetime.utcfromtimestamp(new * 0.001)
    update()


def update():
    global start, end

    df_trade_feed_buy_new = df_trade_feed_buy.query("'{}' <= created_at <= '{}'".format(start, end))
    df_trade_feed_sell_new = df_trade_feed_sell.query("'{}' <= created_at <= '{}'".format(start, end))

    source_trade_feed_buy.data = dict(
        created_at=df_trade_feed_buy_new["created_at"],
        price=df_trade_feed_buy_new["price"],
        quantity=df_trade_feed_buy_new["quantity"],
    )

    source_trade_feed_sell.data = dict(
        created_at=df_trade_feed_sell_new["created_at"],
        price=df_trade_feed_sell_new["price"],
        quantity=df_trade_feed_sell_new["quantity"],
    )

    source_trade_feed_center.data = dict(
        x0=[start + (end - start) * 0.5],
        y0=[df_trade_feed["price"].min()],
        x1=[start + (end - start) * 0.5],
        y1=[df_trade_feed["price"].max()],
    )

    df_trade_feed_new = df_trade_feed.query("'{}' <= created_at <= '{}'".format(start, start + (end - start) * 0.5))
    df_trade_feed_buy_new = df_trade_feed_buy.query("'{}' <= created_at <= '{}'".format(start, start + (end - start) * 0.5))
    df_trade_feed_sell_new = df_trade_feed_sell.query("'{}' <= created_at <= '{}'".format(start, start + (end - start) * 0.5))
    df_trade_feed_side_new = get_df_trade_feed_side(df_trade_feed_new)
    df_trade_feed_sell_neg_new = df_trade_feed_sell_neg.query("'{}' <= created_at <= '{}'".format(start, start + (end - start) * 0.5))
    df_trade_feed_hist_new = get_df_hist(df_trade_feed_buy_new, df_trade_feed_sell_neg_new)

    source_trade_feed_distribution_buy.data = dict(
        created_at=df_trade_feed_buy_new["created_at"],
        price=df_trade_feed_buy_new["price"],
        quantity=df_trade_feed_buy_new["quantity"],
    )

    source_trade_feed_distribution_sell.data = dict(
        created_at=df_trade_feed_sell_new["created_at"],
        price=df_trade_feed_sell_new["price"],
        quantity=df_trade_feed_sell_neg_new["quantity"],
    )

    source_trade_feed_side.data = dict(
        start_angle=[pi / 2, df_trade_feed_side_new.tail(1)["rate"].values[0] * 2 * pi + pi / 2],
        end_angle=[df_trade_feed_side_new.tail(1)["rate"].values[0] * 2 * pi + pi / 2, pi / 2],
        color=["red", "green"],
        side=["sell", "buy"],
    )

    source_trade_feed_hist_buy.data = dict(
        quantity=df_trade_feed_hist_new.query("taker_side == 'buy'")["quantity"],
        count=df_trade_feed_hist_new.query("taker_side == 'buy'")["count"],
    )

    source_trade_feed_hist_sell.data = dict(
        quantity=df_trade_feed_hist_new.query("taker_side == 'sell'")["quantity"],
        count=df_trade_feed_hist_new.query("taker_side == 'sell'")["count"],
    )

    snapshot_start = start + (end - start) * 0.5 - timedelta(seconds=1)
    snapshot_end = start + (end - start) * 0.5 + timedelta(seconds=1)
    df_order_book_buy_new = df_order_book_buy.query("'{}' <= created_at <= '{}'".format(snapshot_start, snapshot_end))
    df_order_book_sell_new = df_order_book_sell.query("'{}' <= created_at <= '{}'".format(snapshot_start, snapshot_end))

    source_order_book_buy.data = dict(
        price=df_order_book_buy_new["price"],
        quantity=df_order_book_buy_new["quantity"],
    )

    source_order_book_sell.data = dict(
        price=df_order_book_sell_new["price"],
        quantity=df_order_book_sell_new["quantity"],
    )


p_trade_feed.x_range.on_change('start', lambda attr, old, new: update_start(new))
p_trade_feed.x_range.on_change('end', lambda attr, old, new: update_end(new))


animate_start = datetime(2019, 3, 3, 13, 59, 0, tzinfo=timezone.utc).timestamp() * 1000
animate_end = datetime(2019, 3, 3, 15, 0, 0, tzinfo=timezone.utc).timestamp() * 1000


def animate_update():
    global animate_start
    animate_start = animate_start + 2000
    p_trade_feed.x_range.start = animate_start
    p_trade_feed.x_range.end = animate_start + 600000


callback_id = None


def animate():
    global callback_id, start, end
    if button.label == '► Play':
        button.label = '❚❚ Pause'
        callback_id = curdoc().add_periodic_callback(animate_update, 1)
    else:
        button.label = '► Play'
        start = datetime(2019, 3, 3, 14, 0, 0, tzinfo=timezone.utc).timestamp() * 1000
        end = datetime(2019, 3, 3, 15, 0, 0, tzinfo=timezone.utc).timestamp() * 1000
        curdoc().remove_periodic_callback(callback_id)


button = Button(label='► Play', width=60)
button.on_click(animate)

left = Column(p_trade_feed, p_candlestick, p_trade_feed_hist, WidgetBox(button))
right = Column(p_order_book, p_trade_feed_distribution, p_buy_sell_pie)

curdoc().add_root(Row(left, right))
