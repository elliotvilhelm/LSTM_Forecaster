from discord_connector.discord_connector import CHANNEL_ID, client
from discord_connector.secrets import TOKEN
from data_collection.yfinance_collector import get_ohlc
from datetime import datetime
import discord
import asyncio
from config import HISTORY_SIZE
import tensorflow as tf


interval = 5  # * 60 * 1
tickers = ["SPY", "ROKU", "QQQ", "MSFT", "NVDA"]
tickers = ["NOC"]
time_frame = "1h"
red = 0xFF0000
green = 0x008000


def get_single_sequence(df):
    df = df[['Close', 'Volume']]
    df['MA_short'] = df['Close'].rolling(window=7).mean()
    df['Change_1'] = df['Close'] - df['Close'].shift(1)
    df['Change_4'] = df['Close'] - df['Close'].shift(4)
    df['Change_8'] = df['Close'] - df['Close'].shift(8)
    df = df[8:]
    df = df.to_numpy()
    df = (df - df.min(axis=0)) / (df.max(axis=0) - df.min(axis=0))
    df = df[-HISTORY_SIZE:]
    df = df.reshape(1, df.shape[0], df.shape[1])
    return df


async def stock_watch_job():
    p_map = {0: "UP > 1/10th STD", 1: "UP < 1/10th STD", 2: "DOWN > 1/10th STD", 3: "DOWN < 1/10th STD"}
    c_map = {0: green, 1: green, 2: red, 3: red}
    await client.wait_until_ready()

    while not client.is_closed():
        try:
            model = tf.keras.models.load_model('checkpoints/multivariate_single_model')
            for ticker in tickers:
                df = get_ohlc(ticker, period="1mo", interval=time_frame)
                df_old = df
                x = get_single_sequence(df)

                channel = client.get_channel(CHANNEL_ID)
                print("predicting")
                prediction = model.predict(x=x)[0]
                p_max = prediction.argmax()
                trend = p_map[p_max]
                color = c_map[p_max]
                time = datetime.now().strftime("%I:%M %p") + " PT"
                em1 = discord.Embed(title=f'${ticker}', description="LSTM Classification", colour=color)
                em1.add_field(name="Current Price", value=f"${df_old['Close'][-1]}")
                em1.add_field(name="Time Frame", value=time_frame)
                em1.add_field(name="Trend", value=trend)
                em1.add_field(name="Time", value=time)
                em1.add_field(name="Prediction", value=str(prediction))
                em1.set_footer(text="©2020 by LSTM SQUAD", icon_url="https://media1.giphy.com/media/CVtNe84hhYF9u/giphy.gif")
                em1.set_thumbnail(url="https://media1.giphy.com/media/CVtNe84hhYF9u/giphy.gif")
                await channel.send(embed=em1)
            # await asyncio.sleep(interval)
        except Exception as e:
            print(f"Exception:{e}")

        break

client.loop.create_task(stock_watch_job())
client.run(TOKEN)
