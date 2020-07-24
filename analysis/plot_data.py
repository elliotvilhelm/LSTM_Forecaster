import plotly.graph_objects as go
from data_collection.yfinance_collector import get_ohlc
import pandas as pd
from datetime import datetime
from data_processing.data_processing import multivariate_data, get_datasets, add_features, split_multivariate
from config import HISTORY_SIZE, TARGET_DIS, STEP, FEATURES, STD_DENOMINATOR, TEST_MODEL
from tf_kit.model import get_lstm


class Eq:
    def __init__(self, data_set):
        self.data = data_set


def get_text_label(x):
    if x == 0:
        return "UP"
    elif x == 1:
        return ""
    elif x == 2:
        return "DOWN"


# 7 hours per day
df = get_ohlc('ba', period='2y', interval='1h', start="2018-07-27")
dates = df.index
new_dates = []
current_date = None
acc = 0
for i, date in enumerate(dates):
    if not current_date or current_date != str(date.date()):
        current_date = str(date.date())
        acc = 0
    else:
        acc += 1
    new_date = datetime(date.year, date.month, date.day, 6 + acc, 0, 0)
    new_dates.append(new_date)

df.index = new_dates
e = Eq(df)
e = add_features(e)
e.data = e.data[43:]
dataset = e.data[FEATURES].values
dataset = (dataset - dataset.min(axis=0)) / (dataset.max(axis=0) - dataset.min(axis=0))
std_close = dataset.std(axis=0)[0] / STD_DENOMINATOR
e.data['Date'] = e.data.index

dataset = dataset[-(7 * 5 * 4):]
e.data = e.data[-(7 * 5 * 4):]
x_train_single, y_train_single = multivariate_data(dataset, dataset[:, 0], 0,
                                                   None, HISTORY_SIZE,
                                                   TARGET_DIS, STEP, std_close)
data = e.data[HISTORY_SIZE + TARGET_DIS:]
e.data = e.data[HISTORY_SIZE:]
model = get_lstm()
model.load_weights(TEST_MODEL)
y_out = model.predict(x_train_single)
y_out_class = y_out.argmax(axis=1)
y_out_text = list(map(lambda x: get_text_label(x), y_out_class))

layout = go.Layout(
    xaxis=dict(
        type='category',
    )
)

fig = go.Figure(data=[go.Candlestick(x=e.data.index,
                                     open=e.data['Open'],
                                     high=e.data['High'],
                                     low=e.data['Low'],
                                     close=e.data['Close'])],
                layout=layout)

fig.add_trace(go.Scatter(
    x=e.data['Date'],
    y=e.data['Close'],
    mode="markers+text",
    name="Markers and Text",
    text=y_out_text,
    textposition="bottom center"
))

fig.show()
