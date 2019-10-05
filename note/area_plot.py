import requests
import pandas as pd
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, CrosshairTool, NumeralTickFormatter
from bokeh.io import show

# データ準備
data = requests.get("https://min-api.cryptocompare.com/data/v2/histoday?fsym=BTC&tsym=JPY&limit=150")
df = pd.DataFrame(data.json()["Data"]["Data"])
df["time"] = pd.to_datetime(df["time"], unit="s")

# Bokeh描画用データ
source = ColumnDataSource(df)

# 描画レイアウト
p = figure(
    title="BTC/JPY Close Price",
    plot_width=1500,
    plot_height=400,
    x_axis_type="datetime"
)

# メインチャート
p.varea(
    x="time",
    y1="close",
    y2=0,
    source=source,
    alpha=0.5
)

# 吹き出し
hover = HoverTool(
    tooltips=[
        ("Date", "@time{%F}"),
        ("Close", "@close{0,0}")
    ],
    formatters={"time": "datetime"},
    mode="vline",
    show_arrow=False,  # 矢印を消します
)
p.add_tools(hover)

# マウスの位置に合わせて●を表示
r_circle = p.circle(
    x="time",
    y="close",
    source=source,
    color=None,  # マウスの位置だけ表示するので基本は非表示
    hover_alpha=0.8,
    line_color=None,
    hover_line_color=None,
    size=8
)

# マウスの位置だけ表示
hover_circle = HoverTool(
    tooltips=None,
    renderers=[r_circle],
    mode="vline"
)
p.add_tools(hover_circle)

# マウスの位置に縦棒を表示
cross_hair = CrosshairTool(
    dimensions="height",
    line_alpha=0.2
)
p.add_tools(cross_hair)

# y軸の数字フォーマット
p.yaxis[0].formatter = NumeralTickFormatter(format="0,0")

# 背景グリッドの線を消す
p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None

# 表示
show(p)
