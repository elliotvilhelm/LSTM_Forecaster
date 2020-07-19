from discord_connector.discord_connector import CHANNEL_ID, client
from discord_connector.secrets import TOKEN
from data_collection.yfinance_collector import get_ohlc
from config import HISTORY_SIZE, TEST_TICKERS, TEST_MODEL, FEATURES
from tf_kit.model import get_lstm

from ta import add_all_ta_features
from ta.utils import dropna

from datetime import datetime
import discord
import asyncio
import tensorflow_addons as tfa
import tensorflow as tf

tfa.register_all()

INTERVAL = 3600
TFRAME = "1h"
RED = 0xFF0000
GREEN = 0x21fc21
GREY = 0xe8fce8
P_MAP = {0: "UP", 1: "NOTHING", 2: "DOWN"}
D_MAP = {0: "UP",
         1: "CHOP",
         2: "DOWN"}
C_MAP = {0: GREEN, 1: GREY, 2: RED}


def get_single_sequence(df):
    df = dropna(df)
    df = add_all_ta_features(df,
                             open="Open",
                             high="High",
                             low="Low",
                             close="Close",
                             volume="Volume")
    df = df[15:]
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

    e = discord.Embed(title=f'${ticker}', description=D_MAP[p_max], colour=color)
    e.add_field(name="Current Price", value=f"${close}")
    e.add_field(name="Time Frame", value=TFRAME)
    e.add_field(name="Trend", value=trend)
    e.add_field(name="Time", value=t)
    e.add_field(name="Prediction", value="\n".join(["{} -> {:.2f}\n".format(P_MAP[i],round(p, 2)) for i, p in enumerate(pred)]))
    e.set_footer(text="©2020 by LSTM SQUAD", icon_url="https://media1.giphy.com/media/CVtNe84hhYF9u/giphy.gif")
    e.set_thumbnail(url="https://miro.medium.com/max/1400/1*rKWZiar6GfFh9jSAT9i62A.gif")
    return e


async def stock_watch_job():
    await client.wait_until_ready()
    while 1:
        try:
            # allows to train and test live at same time
            with tf.device('/cpu:0'):
                model = get_lstm()
                model.load_weights(TEST_MODEL)

                print('-' * 80)
                for ticker in TEST_TICKERS:
                    print("[{}]".format(ticker))
                    df = get_ohlc(ticker, period="1mo", interval=TFRAME)
                    df_old = df
                    x = get_single_sequence(df)

                    pred = model.predict(x=x)[0]
                    e = build_embed(ticker, pred, df_old['Close'][-1])

                    channel = client.get_channel(CHANNEL_ID)
                    print('-' * 80)

                    await channel.send(embed=e)
        except Exception as e:
            print(f"Exception:{e}")
            exit(1)
        await asyncio.sleep(INTERVAL)

client.loop.create_task(stock_watch_job())
client.run(TOKEN)
