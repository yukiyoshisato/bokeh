from os.path import dirname, join

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from bokeh.layouts import row, widgetbox, column, gridplot
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.models.widgets import Button, DataTable, TableColumn, PreText, Panel, Tabs, Select, TextInput, Slider, CheckboxGroup
from bokeh.plotting import figure
from bokeh.io import curdoc


# Load Data

df_titanic = pd.read_csv("data/titanic/train.csv")
source_titanic = ColumnDataSource(data=dict())
source_titanic_stats = ColumnDataSource(data=dict())
source_titanic_missing = ColumnDataSource(data=dict())

columns_titanic = []
for column_titanic in df_titanic.reset_index().columns:
    columns_titanic.append(TableColumn(field=column_titanic, title=column_titanic))

button_read_csv = Button(label="Read CSV", button_type="success")
button_fill_missing_value = Button(label="Fill missing value", button_type="success")
column_fill_missing_value = Select(title="Column", value="", options=list(df_titanic.columns))
input_fill_missing_value = TextInput(title="Value")

title_table_titanic = PreText(text="Data")
data_table_titanic = DataTable(source=source_titanic, columns=columns_titanic, width=1000)
title_table_titanic_stats = PreText(text="Statistics")
data_table_titanic_stats = DataTable(source=source_titanic_stats, columns=columns_titanic, width=1000)


def read_csv():
    global df_titanic
    df_titanic = pd.read_csv("data/titanic/train.csv")
    update_titanic()


def update_titanic():
    global df_titanic

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
            print(column_titanic)
            pass
    source_titanic_stats.data = data_titanic_stats


def fill_missing_value():
    global df_titanic
    try:
        value = float(input_fill_missing_value.value)
    except ValueError:
        value = input_fill_missing_value.value
    df_titanic[column_fill_missing_value.value].fillna(value, inplace=True)
    update_titanic()


button_read_csv.on_click(read_csv)
button_fill_missing_value.on_click(fill_missing_value)

input_load_data = column(button_read_csv, column(button_fill_missing_value, column_fill_missing_value, input_fill_missing_value))
output_load_data = column(title_table_titanic, data_table_titanic, title_table_titanic_stats, data_table_titanic_stats)
load_titanic = row(input_load_data, output_load_data)
tab_load_titanic = Panel(child=load_titanic, title="Load Data")

# data visualization

survived = df_titanic["Survived"].drop_duplicates().values

groups = df_titanic.groupby('Survived')
q1 = groups.quantile(q=0.25)
q2 = groups.quantile(q=0.5)
q3 = groups.quantile(q=0.75)
iqr = q3 - q1
upper = q3 + 1.5*iqr
lower = q1 - 1.5*iqr


def outliers(group):
    survive_flag = group.name
    return group[(group.Age > upper.loc[survive_flag]['Age']) | (group.Age < lower.loc[survive_flag]['Age'])]['Age']


out = groups.apply(outliers).dropna()

if not out.empty:
    outx = []
    outy = []
    for keys in out.index:
        outx.append(keys[0])
        outy.append(out.loc[keys[0]].loc[keys[1]])

boxplot_age_survived = figure(background_fill_color="#efefef", x_range=(-2.5, 3.5), plot_height=600, title="boxplot")
# if no outliers, shrink lengths of stems to be no longer than the minimums or maximums
qmin = groups.quantile(q=0.00)
qmax = groups.quantile(q=1.00)
upper.Age = [min([x,y]) for (x,y) in zip(list(qmax.loc[:,'Age']),upper.Age)]
lower.Age = [max([x,y]) for (x,y) in zip(list(qmin.loc[:,'Age']),lower.Age)]

# stems
boxplot_age_survived.segment(survived, upper.Age, survived, q3.Age, line_color="black")
boxplot_age_survived.segment(survived, lower.Age, survived, q1.Age, line_color="black")

# boxes
boxplot_age_survived.vbar(survived, 0.7, q2.Age, q3.Age, fill_color="#E08E79", line_color="black")
boxplot_age_survived.vbar(survived, 0.7, q1.Age, q2.Age, fill_color="#3B8686", line_color="black")

# whiskers (almost-0 height rects simpler than segments)
boxplot_age_survived.rect(survived, lower.Age, 0.2, 0.01, line_color="black")
boxplot_age_survived.rect(survived, upper.Age, 0.2, 0.01, line_color="black")

# outliers
if not out.empty:
    boxplot_age_survived.circle(outx, outy, size=6, color="#F38630", fill_alpha=0.6)

boxplot_age_survived.xgrid.grid_line_color = None
boxplot_age_survived.ygrid.grid_line_color = "white"
boxplot_age_survived.grid.grid_line_width = 1
boxplot_age_survived.xaxis.axis_label = "Survived"
boxplot_age_survived.yaxis.axis_label = "Age"

# scatter plot

scatter_x = Select(title="x", value="Age", options=list(df_titanic.columns))
scatter_y = Select(title="y", value="Fare", options=list(df_titanic.columns))
scatter_button = Button(label="Refresh", button_type="success")

blank_data_scatter = dict(
    x=[],
    y=[],
    PassengerId=[],
    Survived=[],
    Pclass=[],
    Name=[],
    Sex=[],
    Age=[],
    SibSp=[],
    Parch=[],
    Ticket=[],
    Fare=[],
    Cabin=[],
    Embarked=[],
)
source_scatter_0_male = ColumnDataSource(data=dict(blank_data_scatter))
source_scatter_0_female = ColumnDataSource(data=dict(blank_data_scatter))
source_scatter_1_male = ColumnDataSource(data=dict(blank_data_scatter))
source_scatter_1_female = ColumnDataSource(data=dict(blank_data_scatter))


def update_scatter():
    x = scatter_x.value
    y = scatter_y.value
    query_0_male = "Survived == 0 and Sex == 'male'"
    data_0_male = dict(
        x=df_titanic.query(query_0_male)[x],
        y=df_titanic.query(query_0_male)[y],
        PassengerId=df_titanic.query(query_0_male)["PassengerId"],
        Survived=df_titanic.query(query_0_male)["Survived"],
        Pclass=df_titanic.query(query_0_male)["Pclass"],
        Name=df_titanic.query(query_0_male)["Name"],
        Sex=df_titanic.query(query_0_male)["Sex"],
        Age=df_titanic.query(query_0_male)["Age"],
        SibSp=df_titanic.query(query_0_male)["SibSp"],
        Parch=df_titanic.query(query_0_male)["Parch"],
        Ticket=df_titanic.query(query_0_male)["Ticket"],
        Fare=df_titanic.query(query_0_male)["Fare"],
        Cabin=df_titanic.query(query_0_male)["Cabin"],
        Embarked=df_titanic.query(query_0_male)["Embarked"],
    )
    source_scatter_0_male.data = data_0_male

    query_0_female = "Survived == 0 and Sex == 'female'"
    data_0_female = dict(
        x=df_titanic.query(query_0_female)[x],
        y=df_titanic.query(query_0_female)[y],
        PassengerId=df_titanic.query(query_0_female)["PassengerId"],
        Survived=df_titanic.query(query_0_female)["Survived"],
        Pclass=df_titanic.query(query_0_female)["Pclass"],
        Name=df_titanic.query(query_0_female)["Name"],
        Sex=df_titanic.query(query_0_female)["Sex"],
        Age=df_titanic.query(query_0_female)["Age"],
        SibSp=df_titanic.query(query_0_female)["SibSp"],
        Parch=df_titanic.query(query_0_female)["Parch"],
        Ticket=df_titanic.query(query_0_female)["Ticket"],
        Fare=df_titanic.query(query_0_female)["Fare"],
        Cabin=df_titanic.query(query_0_female)["Cabin"],
        Embarked=df_titanic.query(query_0_female)["Embarked"],
    )
    source_scatter_0_female.data = data_0_female

    query_1_male = "Survived == 1 and Sex == 'male'"
    data_1_male = dict(
        x=df_titanic.query(query_1_male)[x],
        y=df_titanic.query(query_1_male)[y],
        PassengerId=df_titanic.query(query_1_male)["PassengerId"],
        Survived=df_titanic.query(query_1_male)["Survived"],
        Pclass=df_titanic.query(query_1_male)["Pclass"],
        Name=df_titanic.query(query_1_male)["Name"],
        Sex=df_titanic.query(query_1_male)["Sex"],
        Age=df_titanic.query(query_1_male)["Age"],
        SibSp=df_titanic.query(query_1_male)["SibSp"],
        Parch=df_titanic.query(query_1_male)["Parch"],
        Ticket=df_titanic.query(query_1_male)["Ticket"],
        Fare=df_titanic.query(query_1_male)["Fare"],
        Cabin=df_titanic.query(query_1_male)["Cabin"],
        Embarked=df_titanic.query(query_1_male)["Embarked"],
    )
    source_scatter_1_male.data = data_1_male

    query_1_female = "Survived == 1 and Sex == 'female'"
    data_1_female = dict(
        x=df_titanic.query(query_1_female)[x],
        y=df_titanic.query(query_1_female)[y],
        PassengerId=df_titanic.query(query_1_female)["PassengerId"],
        Survived=df_titanic.query(query_1_female)["Survived"],
        Pclass=df_titanic.query(query_1_female)["Pclass"],
        Name=df_titanic.query(query_1_female)["Name"],
        Sex=df_titanic.query(query_1_female)["Sex"],
        Age=df_titanic.query(query_1_female)["Age"],
        SibSp=df_titanic.query(query_1_female)["SibSp"],
        Parch=df_titanic.query(query_1_female)["Parch"],
        Ticket=df_titanic.query(query_1_female)["Ticket"],
        Fare=df_titanic.query(query_1_female)["Fare"],
        Cabin=df_titanic.query(query_1_female)["Cabin"],
        Embarked=df_titanic.query(query_1_female)["Embarked"],
    )
    source_scatter_1_female.data = data_1_female


scatter_plot = figure(title="Scatter Plot", plot_width=1000, plot_height=600)
scatter_plot.x(x="x", y="y", source=source_scatter_0_male, color="blue", size=10, alpha=0.5, legend="Not Survived (Male)")
scatter_plot.x(x="x", y="y", source=source_scatter_0_female, color="red", size=10, alpha=0.5, legend="Not Survived (Female)")
scatter_plot.circle(x="x", y="y", source=source_scatter_1_male, color="blue", size=10, alpha=0.5, legend="Survived (Male)")
scatter_plot.circle(x="x", y="y", source=source_scatter_1_female, color="red", size=10, alpha=0.5, legend="Survived (Female)")
scatter_plot.add_tools(HoverTool(
    tooltips=[
        ("PassengerId", "@PassengerId"),
        ("Survived", "@Survived"),
        ("Pclass", "@Pclass"),
        ("Name", "@Name"),
        ("Sex", "@Sex"),
        ("Age", "@Age"),
        ("SibSp", "@SibSp"),
        ("Parch", "@Parch"),
        ("Ticket", "@Ticket"),
        ("Fare", "@Fare"),
        ("Cabin", "@Cabin"),
        ("Embarked", "@Embarked"),
        ("x", "$x"),
        ("y", "$y"),
    ]
))
scatter_plot.legend.click_policy = "hide"
scatter_button.on_click(update_scatter)
input_scatter = widgetbox(scatter_x, scatter_y, scatter_button, height=600)

# histogram
hist_x = Select(title="x", value="Age", options=list(df_titanic.columns))
hist_button = Button(label="Refresh", button_type="success")
source_hist = ColumnDataSource(data=dict(top=[], bottom=[], left=[], right=[],))


def update_histogram():
    measured = df_titanic[hist_x.value]
    hist, edges = np.histogram(measured, density=True, bins=50)
    source_hist.data = dict(
        top=hist,
        bottom=[0]*len(hist),
        left=edges[:-1],
        right=edges[1:],
    )


hist_plot = figure(title="Histogram", plot_width=1000, plot_height=300)
hist_plot.quad(top="top", bottom="bottom", left="left", right="right", source=source_hist, fill_color="navy", line_color="white", alpha=0.5)
hist_plot.legend.background_fill_color = "#fefefe"
hist_plot.grid.grid_line_color = "white"
hist_button.on_click(update_histogram)

data_visualization = gridplot([
    [input_scatter, scatter_plot],
    [widgetbox(hist_x, hist_button), hist_plot],
    [None, boxplot_age_survived]
])

tab_data_visualization = Panel(child=data_visualization, title="Data Visualization")

# train

x_titanic_title = PreText(text="X")
x_titanic = CheckboxGroup(labels=list(df_titanic.columns), active=[i for i in range(2, 12)])
y_titanic_title = PreText(text="y")
y_titanic = CheckboxGroup(labels=list(df_titanic.columns), active=[1])
train_test_slider = Slider(start=0.05, end=0.95, value=0.25, step=0.05, title="Test Size")
train_button = Button(label="Train", button_type="success")
train_message = PreText(text="")
input_x = column(x_titanic_title, x_titanic)
input_y = column(y_titanic_title, y_titanic)
input_others = column(train_test_slider, train_button, train_message)
controls_train = row(input_x, input_y, input_others)

source_importance = ColumnDataSource(dict(x=[], top=[],))


def train():
    if 1 < len(y_titanic.active):
        train_message.text = "please select only one item for y."
        return None
    X = df_titanic[[x_titanic.labels[i] for i in x_titanic.active]]
    #if X.isnull().all():
    #    train_message.text = "please fill missing values or please don't select the item including missing values."
    #    print(X.to_string())
    #    return None
    y = df_titanic[[y_titanic.labels[i] for i in y_titanic.active]]
    print(y.to_string())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(train_test_slider.value), random_state=42)
    random_forest = RandomForestClassifier()
    random_forest.fit(X_train, X_test)
    train_message.text = "Training..."
    importances = random_forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in random_forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    data_importance = dict(
        x=range(X.shape[1]),
        top=importances[indices],
    )
    train_message.text = "Completed."


importance_plot = figure(title="Feature Importance", plot_width=1000, plot_height=600)
importance_plot.vbar(x="x", top="top", bottom=0, width=5, source=source_importance)

train_button.on_click(train)

tab_train = Panel(child=column(controls_train, importance_plot), title="Train")


# consolidate all tabs

tabs = Tabs(tabs=[tab_load_titanic, tab_data_visualization, tab_train])

curdoc().add_root(tabs)


