from os.path import dirname, join

import pandas as pd

from bokeh.layouts import row, widgetbox, column
from bokeh.models import ColumnDataSource, CustomJS
from bokeh.models.widgets import RangeSlider, Button, DataTable, TableColumn, NumberFormatter, PreText
from bokeh.io import curdoc

df_titanic = pd.read_csv("data/titanic/train.csv")
source_titanic = ColumnDataSource(data=dict())
source_titanic_stats = ColumnDataSource(data=dict())
source_titanic_missing = ColumnDataSource(data=dict())


def update_titanic():
    global df_titanic
    df_titanic = pd.read_csv("data/titanic/train.csv")

    data_titanic = {}
    df_titanic_data = df_titanic.reset_index()
    df_titanic_data = df_titanic_data.append(pd.DataFrame(data=[[""] * len(df_titanic_data.columns)], columns=df_titanic_data.columns))
    for column_titanic in df_titanic_data.columns:
        data_titanic[column_titanic] = df_titanic_data.loc[:, column_titanic]
    source_titanic.data = data_titanic

    data_titanic_stats = {}
    df_titanic_stats = df_titanic.describe()
    for column_titanic in df_titanic.columns:
        df_titanic_stats.loc["count", column_titanic] = df_titanic.loc[:, column_titanic].count()
    df_titanic_stats.reset_index(inplace=True)
    for column_titanic in df_titanic_stats.columns:
        try:
            data_titanic_stats[column_titanic] = df_titanic_stats.loc[:, column_titanic]
        except KeyError:
            pass
    source_titanic_stats.data = data_titanic_stats



button_titanic = Button(label="Read CSV", button_type="success")
button_titanic.on_click(update_titanic)

columns_titanic = []
for column_titanic in df_titanic.reset_index().columns:
    columns_titanic.append(TableColumn(field=column_titanic, title=column_titanic))

title_table_titanic = PreText(text="Data")
data_table_titanic = DataTable(source=source_titanic, columns=columns_titanic, width=1200)
title_table_titanic_stats = PreText(text="Statistics")
data_table_titanic_stats = DataTable(source=source_titanic_stats, columns=columns_titanic, width=1200)

curdoc().add_root(column(button_titanic, title_table_titanic, data_table_titanic, title_table_titanic_stats, data_table_titanic_stats))


