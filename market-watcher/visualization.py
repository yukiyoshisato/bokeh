import ccxt
from math import pi
import pandas as pd
from trade_feed import TradeFeed
from order_book import OrderBook
from bokeh.layouts import row, widgetbox, column, gridplot
from bokeh.models import ColumnDataSource, HoverTool, CustomJS, NumeralTickFormatter
from bokeh.models.widgets import Button, DataTable, TableColumn, PreText, Panel, Tabs, Select, TextInput, Slider, CheckboxGroup
from bokeh.plotting import figure
from bokeh.io import show


def plot_candlestick():

    df = pd.read_csv("./data/trade_feed/trade_feed_2019-03-03_14-00-08.csv").drop(["id", "quantity", "taker_side"], axis=1)
    df["created_at"] = pd.to_datetime(df["created_at"], unit="s")
    df.set_index("created_at", inplace=True)
    df["price"] = df["price"].astype(float)
    df = df.resample("1T").ohlc()
    df.columns = df.columns.droplevel(0)

    inc = df.close > df.open
    dec = df.open > df.close
    w = 1 * 60 * 1000  # half day in ms

    TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

    p = figure(x_axis_type="datetime", tools=TOOLS, plot_height=250, plot_width=750, title="Candlestick")

    p.segment(df.index, df.high, df.index, df.low, color="black")
    p.vbar(df.index[inc], w, df.open[inc], df.close[inc], fill_color="#D5E1DD", line_color="black")
    p.vbar(df.index[dec], w, df.open[dec], df.close[dec], fill_color="#F2583E", line_color="black")

    p.grid.grid_line_alpha = 0.3
    p.left[0].formatter.use_scientific = False

    return p


def plot_trade_feed(source_buy, source_sell, source_center):

    p = figure(title="Trade Feed", x_axis_type="datetime", plot_height=250, plot_width=750, output_backend="webgl")
    p.circle(x="created_at", y="price", size="quantity", color="green", source=source_buy,  alpha=0.5)
    p.circle(x="created_at", y="price", size="quantity", color="red", source=source_sell, alpha=0.5)
    p.segment(x0="x0", y0="y0", x1="x1", y1="y1", source=source_center, color="navy")

    p.grid.grid_line_alpha = 0.3
    p.left[0].formatter.use_scientific = False

    return p


def plot_trade_feed_distribution(source_buy, source_sell):

    p = figure(title="Trade Feed", plot_height=250, plot_width=250)
    p.hbar(y="price", height=0.01, left=0, right="quantity", color="green", source=source_buy)
    p.hbar(y="price", height=0.01, left="quantity", right=0,  color="red", source=source_sell)

    p.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
    p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
    p.yaxis.major_label_text_font_size = '0pt'  # turn off y-axis tick labels
    p.grid.grid_line_alpha = 0.3

    return p


def plot_side_trade_feed():

    df = pd.read_csv("./data/trade_feed/trade_feed_2019-03-03_14-00-08.csv").drop(["id"], axis=1)
    df["created_at"] = pd.to_datetime(df["created_at"], unit="s")

    df = df.groupby("taker_side").sum()
    df["rate"] = df["quantity"] / df["quantity"].sum()
    df.sort_index(inplace=True)

    source = ColumnDataSource(data=dict(
        start_angle=[pi/2, df.tail(1)["rate"].values[0] * 2 * pi + pi/2],
        end_angle=[df.tail(1)["rate"].values[0] * 2 * pi + pi/2, pi/2],
        color=["green", "red"],
        side=["buy", "sell"],
    ))
    p = figure(title="Buy Sell", plot_height=250, plot_width=250)
    p.annular_wedge(x=0, y=0, inner_radius=0.4, outer_radius=0.9, start_angle="start_angle", end_angle="end_angle",
                    fill_color="color", source=source, alpha=0.5, legend="side")

    p.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
    p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
    p.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
    p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
    p.xaxis.major_label_text_font_size = '0pt'  # turn off y-axis tick labels
    p.yaxis.major_label_text_font_size = '0pt'  # turn off y-axis tick labels
    p.grid.grid_line_alpha = 0.3

    return p


def plot_order_book(source_buy, source_sell):

    p = figure(title="Order Book", plot_height=250, plot_width=250)
    p.hbar(y="price", height=0.01, left=0, right="quantity",  color="green", source=source_buy)
    p.hbar(y="price", height=0.01, left="quantity", right=0,  color="red", source=source_sell)

    p.x_range.start = -21
    p.x_range.end = 21
    p.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
    p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
    p.yaxis.major_label_text_font_size = '0pt'  # turn off y-axis tick labels
    p.grid.grid_line_alpha = 0.3

    return p


def _bk_plot_order_book():

    symbol = "BTC/JPY"
    threshold = 1000

    exchanges = ["liquid", "bitflyer", "bitbank", "zaif", "coincheck"]

    order_book_liquid = OrderBook(exchange=ccxt.liquid(), symbol=symbol, threshold=threshold)
    order_book_bitflyer = OrderBook(exchange=ccxt.bitflyer(), symbol=symbol, threshold=threshold)
    order_book_bitbank = OrderBook(exchange=ccxt.bitbank(), symbol=symbol, threshold=threshold)
    order_book_zaif = OrderBook(exchange=ccxt.zaif(), symbol=symbol, threshold=threshold)
    order_book_coincheck = OrderBook(exchange=ccxt.coincheck(), symbol=symbol, threshold=threshold)

    sources_bid = []
    sources_ask = []
    for order_book in [order_book_liquid, order_book_bitflyer, order_book_bitbank, order_book_zaif,
                       order_book_coincheck]:
        source_bid = ColumnDataSource(data=dict(
            price=order_book.bids["price"],
            size=order_book.bids["size"].cumsum(),
        ))
        source_ask = ColumnDataSource(data=dict(
            price=order_book.asks["price"],
            size=order_book.asks["size"].cumsum(),
        ))
        sources_bid.append(source_bid)
        sources_ask.append(source_ask)

    colors = ["blue", "orange", "green", "purple", "red"]

    plots = []
    for exchange, s_bid, s_ask, color in zip(exchanges, sources_bid, sources_ask, colors):
        plot_order_book = figure(title=exchange, plot_width=1000, plot_height=150)
        plot_order_book.line(x="price", y="size", source=s_bid, color=color)
        plot_order_book.line(x="price", y="size", source=s_ask, color=color)
        plot_order_book.xaxis[0].formatter = NumeralTickFormatter(format="0,0")
        plots.append(plot_order_book)

    plots[0].x_range = plots[1].x_range = plots[2].x_range = plots[3].x_range = plots[4].x_range
    plots[0].y_range = plots[1].y_range = plots[2].y_range = plots[3].y_range = plots[4].y_range

    return plots


def blank_plot():

    return figure(plot_width=250, plot_height=250)
