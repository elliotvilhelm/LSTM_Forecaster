from discord import Client, Embed
from data_collection.yfinance_collector import get_ohlc
from config import HISTORY_SIZE, TEST_MODEL, FEATURES, TARGET_DIS
from ta import add_all_ta_features
from ta.utils import dropna
from tf_kit.model import get_lstm
from datetime import datetime
import tensorflow as tf


CHANNEL_ID = 733516071289487460
client = Client()

model = get_lstm()
model.load_weights(TEST_MODEL)

TFRAME = "1h"
INTERVAL = 3600
RED = 0xFF0000
GREEN = 0x21fc21
GREY = 0xe8fce8
P_MAP = {0: "UP", 1: "CHOP", 2: "DOWN"}
C_MAP = {0: GREEN, 1: GREY, 2: RED}


def get_single_sequence(df):
    df = dropna(df)
    df = add_all_ta_features(df,
                             open="Open",
                             high="High",
                             low="Low",
                             close="Close",
                             volume="Volume")
    df = df[50:]
    df = df[FEATURES]
    df = df.to_numpy()
    df = (df - df.min(axis=0)) / (df.max(axis=0) - df.min(axis=0))
    df = df[-HISTORY_SIZE:]
    df = df.reshape(1, df.shape[0], df.shape[1])
    print(FEATURES)
    print(df.squeeze())
    return df


def build_embed(ticker, pred, close):
    p_max = pred.argmax()
    trend = P_MAP[p_max]
    color = C_MAP[p_max]
    t = datetime.now().strftime("%I:%M %p") + " PT"
    rounded_pred = list(map(lambda x: round(x, 2), pred))

    e = Embed(title=f'${ticker}', colour=color)
    e.add_field(name="Current Price", value=f"${close}")
    e.add_field(name="Candle", value=TFRAME)
    e.add_field(name="Target", value=str(TARGET_DIS) + " hr")
    e.add_field(name="Trend", value=trend)
    e.add_field(name="Time", value=t)
    e.add_field(name="Prediction", value=str(rounded_pred))
    e.set_footer(text="©2020 by LSTM SQUAD", icon_url="https://media1.giphy.com/media/CVtNe84hhYF9u/giphy.gif")
    e.set_thumbnail(url="https://miro.medium.com/max/1400/1*rKWZiar6GfFh9jSAT9i62A.gif")
    return e


@client.event
async def on_ready():
    print(f'{client.user} has connected to Discord!')


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    content = message.content
    if content[0] == '~':
        contents = content[1:].split(" ")
        ticker = contents[0]
        look_back = 0
        if len(contents) > 1:
            look_back = int(contents[1])
        if look_back != 0:
            df = get_ohlc(ticker, period="2y", interval=TFRAME)[:look_back]
        else:
            df = get_ohlc(ticker, period="2y", interval=TFRAME)
        df_old = df
        x = get_single_sequence(df)
        with tf.device('/cpu:0'):
            pred = model.predict(x=x)[0]
            e = build_embed(ticker, pred, df_old['Close'][-1])
        await message.channel.send(embed=e)
