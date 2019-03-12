import pandas as pd
from bokeh.plotting import figure


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


def plot_side_trade_feed(source):

    p = figure(title="Buy Sell (Trade Feed)", plot_height=200, plot_width=250)
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
    p.hbar(y="price", height=20, left=0, right="quantity",  color="green", source=source_buy)
    p.hbar(y="price", height=20, left="quantity", right=0,  color="red", source=source_sell)

    p.x_range.start = -5
    p.x_range.end = 5
    p.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
    p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
    p.yaxis.major_label_text_font_size = '0pt'  # turn off y-axis tick labels
    p.grid.grid_line_alpha = 0.3

    return p


def plot_trade_feed_histogram(source_buy, source_sell):

    p = figure(title="Trade Feed Histogram", plot_height=200, plot_width=750)
    p.vbar(x="quantity", top="count", width=0.1, color="green", alpha=0.5, source=source_buy)
    p.vbar(x="quantity", top="count", width=0.1, color="red", alpha=0.5, source=source_sell)

    return p
