import os
import re
import asyncio
import discord
from discord.ext import commands
from llama_cpp import Llama
from colorama import init, Fore, Style
init(autoreset=True)

TOKEN = "i never use dotenv , so this is , fr"
MODEL_PATH = "delta-model.Q4_K_M.gguf"
USER_START = "<user>"
USER_END = "</user>\n"
ASSISTANT_START = "<assistant>"
ASSISTANT_END = "</assistant>\n"
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)
histories = {}
llm = None

def get_llm():
    global llm
    if llm is None:
        if not os.path.exists(MODEL_PATH):
            print(f"{Fore.RED}Error: Model file '{MODEL_PATH}' not found.")
            return None
        
        print(f"{Fore.CYAN}Loading model {MODEL_PATH}...")
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=2048,
            n_threads=os.cpu_count() or 4,
            verbose=False
        )
        print(f"{Fore.GREEN}Model loaded successfully!")
    return llm

@bot.event
async def on_ready():
    print(f"{Fore.MAGENTA}{Style.BRIGHT}--- Epsilon Discord Bot ---")
    print(f"{Fore.WHITE}Logged in as: {bot.user} (ID: {bot.user.id})")
    get_llm()
    print("-" * 40)

@bot.command()
async def reset(ctx):
    histories[ctx.channel.id] = ""
    await ctx.reply("✨ Context history reset.")
    print(f"{Fore.YELLOW}Context reset for channel {ctx.channel.id}")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    await bot.process_commands(message)
    is_pinged = bot.user in message.mentions
    is_reply = (
        message.reference 
        and message.reference.resolved 
        and message.reference.resolved.author == bot.user
    )

    if (is_pinged or is_reply) and not message.content.startswith("!"):
        model = get_llm()
        if not model:
            await message.reply("❌ Error: Model not loaded on server.")
            return

        content = message.content.replace(f"<@{bot.user.id}>", "").replace(f"<@!{bot.user.id}>", "").strip()
        
        if not content:
            return

        channel_id = message.channel.id
        history = histories.get(channel_id, "")
        prompt = f"{history}{USER_START}{content}{USER_END}{ASSISTANT_START}"

        async with message.channel.typing():
            try:
                loop = asyncio.get_event_loop()
                
                def generate():
                    return model(
                        prompt,
                        max_tokens=1024,
                        stop=[USER_START, "</assistant>"],
                        temperature=0.7,
                        repeat_penalty=1.1
                    )

                response = await loop.run_in_executor(None, generate)
                response_text = response["choices"][0]["text"].strip()

                if not response_text:
                    response_text = "..."

                history += f"{USER_START}{content}{USER_END}{ASSISTANT_START}{response_text}{ASSISTANT_END}"
                turns = re.findall(r"(<user>.*?</user>\n<assistant>.*?</assistant>\n)", history, re.DOTALL)
                if len(turns) > 5:
                    history = "".join(turns[-5:])
                
                histories[channel_id] = history

                if len(response_text) > 2000:
                    for i in range(0, len(response_text), 2000):
                        await message.channel.send(response_text[i:i+2000])
                else:
                    await message.reply(response_text)
                
                print(f"{Fore.GREEN}Responded to {message.author} in {channel_id}")

            except Exception as e:
                print(f"{Fore.RED}Inference Error: {e}")
                await message.reply(f"⚠️ An error occurred during inference: {str(e)}")

if __name__ == "__main__":
    bot.run(TOKEN)
