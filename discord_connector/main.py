from discord_connector.discord_connector import CHANNEL_ID, client
from discord_connector.secrets import TOKEN
import tensorflow as tf


async def stock_watch_job():
    await client.wait_until_ready()

    while not client.is_closed():
        try:
            channel = client.get_channel(CHANNEL_ID)
            model = tf.keras.models.load_model('checkpoints/multivariate_single_model')
            model.predict([[[]]])

            await channel.send("hey fran @here")
        except Exception as e:
            print(f"Exception:{e}")

        break

client.loop.create_task(stock_watch_job())
client.run(TOKEN)
