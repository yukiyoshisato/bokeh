import pandas as pd
from sklearn.datasets import load_iris
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Button
from bokeh.layouts import Column, Row
from bokeh.io import curdoc

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

sepal_button = Button(label="sepal", button_type="success")
petal_button = Button(label="petal", button_type="success")

source = ColumnDataSource(data=dict(length=[], width=[]))


def update_by_sepal():
    source.data = dict(
        length=df["sepal length (cm)"],
        width=df["sepal width (cm)"]
    )


def update_by_petal():
    source.data = dict(
        length=df["petal length (cm)"],
        width=df["petal width (cm)"]
    )


sepal_button.on_click(update_by_sepal)
petal_button.on_click(update_by_petal)

p = figure(title="Iris", plot_width=400, plot_height=400)
p.circle(x="length", y="width", source=source)

button_area = Column(sepal_button, petal_button)
layout = Row(button_area, p)

curdoc().add_root(layout)
