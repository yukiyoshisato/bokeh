from os.path import dirname, join

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve

from bokeh.layouts import row, widgetbox, column, gridplot
from bokeh.models import ColumnDataSource, HoverTool, CustomJS
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
button_convert_categorical = Button(label="Convert categorical values", button_type="success")

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


def convert_categorical_value():
    global df_titanic
    df_titanic.loc[df_titanic["Sex"] == "male", "Sex"] = 0
    df_titanic.loc[df_titanic["Sex"] == "female", "Sex"] = 1
    df_titanic.loc[df_titanic["Embarked"] == "S", "Embarked"] = 0
    df_titanic.loc[df_titanic["Embarked"] == "C", "Embarked"] = 1
    df_titanic.loc[df_titanic["Embarked"] == "Q", "Embarked"] = 2
    update_titanic()


button_read_csv.on_click(read_csv)
button_fill_missing_value.on_click(fill_missing_value)
button_convert_categorical.on_click(convert_categorical_value)

input_load_data = column(button_read_csv,
                         button_convert_categorical,
                         column_fill_missing_value,
                         input_fill_missing_value,
                         button_fill_missing_value)
output_load_data = column(title_table_titanic,
                          data_table_titanic,
                          title_table_titanic_stats,
                          data_table_titanic_stats)
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
    query_0_male = "Survived == 0 and Sex == 0"
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

    query_0_female = "Survived == 0 and Sex == 1"
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

    query_1_male = "Survived == 1 and Sex == 0"
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

    query_1_female = "Survived == 1 and Sex == 1"
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
bins_slider = Slider(start=1, end=100, value=50, step=1, title="bins")
source_hist = ColumnDataSource(data=dict(top=[], bottom=[], left=[], right=[],))


def update_histogram():
    measured = df_titanic[hist_x.value]
    hist, edges = np.histogram(measured, density=False, bins=int(bins_slider.value))
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
hist_plot.add_tools(HoverTool(
    tooltips=[
        ("top", "@top"),
        ("bottom", "@bottom"),
        ("left", "@left"),
        ("right", "@right"),
    ],
    mode='vline'
))
hist_button.on_click(update_histogram)
bins_slider.on_change("value", lambda attr, old, new: update_histogram())

data_visualization = gridplot([
    [input_scatter, scatter_plot],
    [widgetbox(hist_x, hist_button, bins_slider), hist_plot],
    [None, boxplot_age_survived]
])

tab_data_visualization = Panel(child=data_visualization, title="Data Visualization")

# train

x_titanic_title = PreText(text="X")
x_titanic = CheckboxGroup(labels=list(df_titanic.columns), active=[i for i in range(2, 12)])
y_titanic_title = PreText(text="y")
y_titanic = CheckboxGroup(labels=list(df_titanic.columns), active=[1])
train_sizes_slider = Slider(start=1, end=20, value=10, step=1, title="Train Sizes")
cv_slider = Slider(start=2, end=20, value=10, step=1, title="CV")

train_button = Button(label="Train", button_type="success")
train_message = PreText(text="")
input_x = column(x_titanic_title, x_titanic)
input_y = column(y_titanic_title, y_titanic)
input_others = column(train_sizes_slider, cv_slider, train_button, train_message)
controls_train = column(row(input_x, input_y), input_others)

source_importance = ColumnDataSource(dict(x=[], top=[],))
source_train_scores = ColumnDataSource(dict(train_sizes=[], train_scores=[], test_scores=[]))
source_test_scores = ColumnDataSource(dict(train_sizes=[], train_scores=[], test_scores=[]))
source_train_std = ColumnDataSource(dict(x_train_std=[], y_train_std=[]))
source_test_std = ColumnDataSource(dict(x_test_std=[], y_test_std=[]))

# model
random_forest = RandomForestClassifier()


def train():
    train_message.text = "Training..."
    global importance_plot
    if 1 < len(y_titanic.active):
        train_message.text = "please select only one item for y."
        return None
    x_columns = [x_titanic.labels[i] for i in x_titanic.active]
    X = pd.get_dummies(df_titanic[x_columns])
    y = df_titanic[[y_titanic.labels[i] for i in y_titanic.active]].values.ravel()
    global random_forest
    random_forest.fit(X, y)

    # importance
    importances = random_forest.feature_importances_
    df_importances = pd.DataFrame(columns=["x", "importance"])
    for x, importance in zip(x_columns, importances):
        df_importances = df_importances.append(pd.DataFrame(data=[[x, importance]], columns=df_importances.columns))
    df_importances.sort_values("importance", ascending=False, inplace=True)
    std = np.std([tree.feature_importances_ for tree in random_forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    data_importance = dict(
        x=df_importances["x"],
        top=df_importances["importance"],
    )
    importance_plot.x_range.factors = list(df_importances["x"].values)
    source_importance.data = data_importance

    # learning curve
    train_sizes, train_scores, test_scores = learning_curve(estimator=random_forest,
                                                            X=X,
                                                            y=y,
                                                            train_sizes=np.linspace(.1, 1.0, int(train_sizes_slider.value)),
                                                            cv=int(cv_slider.value),
                                                            n_jobs=4)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    data_train_scores = dict(
        train_sizes=train_sizes,
        train_scores=train_mean,
        test_scores=["N/A"] * len(train_sizes)
    )
    source_train_scores.data = data_train_scores

    data_test_scores = dict(
        train_sizes=train_sizes,
        train_scores=["N/A"] * len(train_sizes),
        test_scores=test_mean
    )
    source_test_scores.data = data_test_scores

    data_train_std = dict(
        x_train_std=np.array([train_sizes, np.flipud(train_sizes)]).flatten(),
        y_train_std=np.array([train_mean + train_std, np.flipud(train_mean - train_std)]).flatten()
    )
    source_train_std.data = data_train_std

    data_test_std = dict(
        x_test_std=np.array([train_sizes, np.flipud(train_sizes)]).flatten(),
        y_test_std=np.array([test_mean + test_std, np.flipud(test_mean - test_std)]).flatten()
    )
    source_test_std.data = data_test_std

    train_message.text = "Completed."


importance_plot = figure(title="Feature Importance", x_range=[""], plot_width=600, plot_height=300)
importance_plot.vbar(x="x", top="top", bottom=0, width=0.5, source=source_importance)
metrics_plot = figure(title="Learning Curve", plot_width=600, plot_height=400)
metrics_plot.line(x="train_sizes", y="train_scores", source=source_train_scores, color="blue", legend="train scores")
metrics_plot.line(x="train_sizes", y="test_scores", source=source_test_scores, color="green", legend="test scores")
metrics_plot.patch(x="x_train_std", y="y_train_std", source=source_train_std, color="blue", alpha=0.15, legend="train std")
metrics_plot.patch(x="x_test_std", y="y_test_std", source=source_test_std, color="green", alpha=0.15, legend="test std")
metrics_plot.legend.location = "bottom_right"
metrics_plot.legend.orientation = "horizontal"
metrics_plot.add_tools(HoverTool(
    tooltips=[
        ("train sizes", "@train_sizes"),
        ("train scores", "@train_scores"),
        ("test scores", "@test_scores"),
    ],
    mode='vline'
))


output_train = column(importance_plot, metrics_plot)

train_button.on_click(train)

tab_train = Panel(child=row(controls_train, output_train), title="Train")

# inference

df_titanic_inference = pd.read_csv("data/titanic/test.csv")
source_titanic_inference = ColumnDataSource(data=dict())
source_inference = ColumnDataSource(data=dict())

columns_titanic_inference = []
for column_titanic_inference in df_titanic_inference.reset_index().columns:
    columns_titanic_inference.append(TableColumn(field=column_titanic_inference, title=column_titanic_inference))

columns_inference = []
columns_inference.append(TableColumn(field="PassengerId", title="PassengerId"))
columns_inference.append(TableColumn(field="Survived", title="Survived"))

button_read_csv_inference = Button(label="Read CSV", button_type="success")
button_fill_missing_value_inference = Button(label="Fill missing value", button_type="success")
column_fill_missing_value_inference = Select(title="Column", value="", options=list(df_titanic_inference.columns))
input_fill_missing_value_inference = TextInput(title="Value")
button_convert_categorical_inference = Button(label="Convert categorical values", button_type="success")
button_inference = Button(label="Inference", button_type="success")
button_download = Button(label="Download", button_type="success")

title_table_titanic_inference = PreText(text="Data")
data_table_titanic_inference = DataTable(source=source_titanic_inference, columns=columns_titanic_inference, width=1000)
title_table_inference = PreText(text="Inference")
data_table_inference = DataTable(source=source_inference, columns=columns_inference, width=500)


def read_csv_inference():
    global df_titanic_inference
    df_titanic_inference = pd.read_csv("data/titanic/test.csv")
    update_titanic_inference()


def update_titanic_inference():
    global df_titanic_inference

    data_titanic_inference = {}
    df_titanic_data_inference = df_titanic_inference.reset_index()
    for column_titanic_inference in df_titanic_data_inference.columns:
        data_titanic_inference[column_titanic_inference] = df_titanic_data_inference.loc[:, column_titanic_inference]
    source_titanic_inference.data = data_titanic_inference


def fill_missing_value_inference():
    global df_titanic_inference
    try:
        value = float(input_fill_missing_value_inference.value)
    except ValueError:
        value = input_fill_missing_value_inference.value
    df_titanic_inference[column_fill_missing_value_inference.value].fillna(value, inplace=True)
    update_titanic_inference()


def convert_categorical_value_inference():
    global df_titanic_inference
    df_titanic_inference.loc[df_titanic_inference["Sex"] == "male", "Sex"] = 0
    df_titanic_inference.loc[df_titanic_inference["Sex"] == "female", "Sex"] = 1
    df_titanic_inference.loc[df_titanic_inference["Embarked"] == "S", "Embarked"] = 0
    df_titanic_inference.loc[df_titanic_inference["Embarked"] == "C", "Embarked"] = 1
    df_titanic_inference.loc[df_titanic_inference["Embarked"] == "Q", "Embarked"] = 2
    update_titanic_inference()


def inference():
    global random_forest
    global df_titanic_inference

    x_columns = [x_titanic.labels[i] for i in x_titanic.active]
    X_inference = pd.get_dummies(df_titanic_inference[x_columns])
    p = random_forest.predict(X_inference)
    df_submit = pd.concat([pd.DataFrame(data=df_titanic_inference["PassengerId"].values, columns=["PassengerId"]),
                           pd.DataFrame(data=p, columns=["Survived"])], axis=1)
    data_inference = {}
    for column_inference in df_submit.columns:
        data_inference[column_inference] = df_submit.loc[:, column_inference]
    source_inference.data = data_inference


button_read_csv_inference.on_click(read_csv_inference)
button_fill_missing_value_inference.on_click(fill_missing_value_inference)
button_convert_categorical_inference.on_click(convert_categorical_value_inference)
button_inference.on_click(inference)
button_download.callback = CustomJS(args=dict(source=source_inference),
                                    code=open(join(dirname(__file__), "download.js")).read())

input_load_data_inference = column(button_read_csv_inference,
                                   button_convert_categorical_inference,
                                   column_fill_missing_value_inference,
                                   input_fill_missing_value_inference,
                                   button_fill_missing_value_inference)
output_load_data_inference = column(title_table_titanic_inference, data_table_titanic_inference)
load_titanic_inference = row(input_load_data_inference, output_load_data_inference)
show_inference = row(column(button_inference, button_download), data_table_inference)
tab_load_titanic_inference = Panel(child=column(load_titanic_inference, show_inference), title="Inference")


# consolidate all tabs

tabs = Tabs(tabs=[tab_load_titanic, tab_data_visualization, tab_train, tab_load_titanic_inference])

curdoc().add_root(tabs)


