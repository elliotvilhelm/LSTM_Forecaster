from discord import Client

CHANNEL_ID = 732851314228330537

client = Client()


@client.event
async def on_ready():
    print(f'{client.user} has connected to Discord!')
