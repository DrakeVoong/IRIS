import discord
from discord.ext import commands 
from discord import Intents
import responser
import configparser

def bot():
    intents = Intents.all()
    client = commands.Bot(command_prefix = '.', intents = intents)
    config = configparser.ConfigParser()
    config.read(".//config.ini")

    token = config['discord']['token']

    @client.event 
    async def on_ready():
        print('Bot is ready.')

    @client.command()
    async def res(ctx):
        #Get message contents
        message_content = ctx.message.content

        message_content = message_content.replace('.hello', '')

        response = responser.response_msg(message_content)


        await ctx.send(response)

    client.run(token)


bot()