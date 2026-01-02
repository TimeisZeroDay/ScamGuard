import os
import sys
import discord
from discord.ext import commands
from discord import app_commands
from dotenv import load_dotenv
from knowledge_loader import get_cached_embeddings, search_knowledge

import asyncio
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from collections import Counter
import datetime

load_dotenv()

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
LOG_CHANNEL_ID = int(os.getenv("LOG_CHANNEL_ID"))
SENIOR_ROLE_ID = int(os.getenv("SENIOR_ROLE_ID", "0"))  # Role ID from env, default 0 if missing

if not DISCORD_BOT_TOKEN:
    print("❌ DISCORD_BOT_TOKEN is not set.")
    exit(1)

intents = discord.Intents.default()
intents.message_content = True
intents.reactions = True
intents.members = True  # Needed to access member roles

bot = commands.Bot(command_prefix="!", intents=intents)

knowledge = []
embeddings = None
index = None

bot.answer_cache = {}
pending_clarifications = {}

usage_stats = {
    "ask_count": 0,
    "clarification_count": 0,
    "negative_ratings": 0,
    "questions_counter": Counter()
}

class KnowledgeFileHandler(FileSystemEventHandler):
    def __init__(self):
        self.last_modified = os.path.getmtime("department_knowledge.txt")

    def on_modified(self, event):
        if event.src_path.endswith("department_knowledge.txt"):
            current_time = os.path.getmtime("department_knowledge.txt")
            if current_time != self.last_modified:
                print("📄 department_knowledge.txt modified. Reloading cache...")
                asyncio.run_coroutine_threadsafe(reload_cache(), bot.loop)
                self.last_modified = current_time

async def reload_cache():
    global knowledge, embeddings, index
    knowledge, embeddings, index = await get_cached_embeddings()
    print("✅ Knowledge cache updated.")

async def send_periodic_reports():
    await bot.wait_until_ready()
    log_channel = bot.get_channel(LOG_CHANNEL_ID)
    if not log_channel:
        print("⚠️ Log channel not found, report sending disabled.")
        return
    
    while not bot.is_closed():
        now = datetime.datetime.now(datetime.timezone.utc)
        if now.hour == 0 and now.minute == 0:
            report = generate_report()
            await log_channel.send(report)
            usage_stats["ask_count"] = 0
            usage_stats["clarification_count"] = 0
            usage_stats["negative_ratings"] = 0
            usage_stats["questions_counter"].clear()
        await asyncio.sleep(60)

def generate_report():
    top_questions = usage_stats["questions_counter"].most_common(5)
    report = (
        f"📊 **Bot Usage Report (Daily)** 📊\n"
        f"Total /ask commands: {usage_stats['ask_count']}\n"
        f"Clarifying questions asked: {usage_stats['clarification_count']}\n"
        f"Negative answer ratings: {usage_stats['negative_ratings']}\n\n"
        f"**Top Questions:**\n"
    )
    if top_questions:
        for q, count in top_questions:
            report += f"• `{q}` — {count} times\n"
    else:
        report += "No questions asked today.\n"
    return report

@bot.event
async def on_ready():
    print(f"✅ Bot is online as {bot.user}")
    try:
        synced = await bot.tree.sync()
        print(f"🔁 Synced {len(synced)} slash commands.")
    except Exception as e:
        print(f"❌ Failed to sync commands: {e}")

    observer = Observer()
    event_handler = KnowledgeFileHandler()
    observer.schedule(event_handler, ".", recursive=False)
    observer.start()
    print("👁️ Monitoring department_knowledge.txt for changes...")

    asyncio.create_task(send_periodic_reports())

    async def shutdown_observer():
        while not bot.is_closed():
            await asyncio.sleep(1)
        observer.stop()
        observer.join()

    asyncio.create_task(shutdown_observer())

@bot.tree.command(name="ask", description="Ask the Scam Department AI a question.")
async def ask(interaction: discord.Interaction, question: str):
    await interaction.response.defer()

    user_id = interaction.user.id

    if user_id in pending_clarifications:
        original_question = pending_clarifications.pop(user_id)
        combined_question = f"{original_question} {question}"
        relevant_info = await search_knowledge(combined_question, knowledge, index)
        used_question = combined_question
        usage_stats["clarification_count"] += 1
    else:
        relevant_info = await search_knowledge(question, knowledge, index)
        used_question = question

    if not relevant_info.strip():
        clarifying_prompt = (
            "I’m not sure I fully understood your question. "
            "Could you please provide more details or clarify what you mean?"
        )
        pending_clarifications[user_id] = question
        await interaction.followup.send(clarifying_prompt)
        return

    usage_stats["ask_count"] += 1
    usage_stats["questions_counter"][used_question] += 1

    if len(relevant_info) > 1000:
        try:
            from openai import OpenAI
            OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
            client = OpenAI(api_key=OPENAI_API_KEY)

            summary_prompt = f"Summarize the following internal knowledge so it's concise and focused for answering a question:\n\n{relevant_info}"
            summary_response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert summarizer."},
                    {"role": "user", "content": summary_prompt}
                ],
                temperature=0.3
            )
            relevant_info = summary_response.choices[0].message.content.strip()
            print("📝 Relevant info was summarized before sending to GPT.")
        except Exception as e:
            print("⚠️ Summarization failed, using full relevant info.", e)

    prompt = f"""
You are a helpful assistant for the Scam Department. Answer based only on internal department knowledge.

Relevant info: {relevant_info}
User's question: {used_question}
Answer:"""

    try:
        from openai import OpenAI
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=OPENAI_API_KEY)

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response.choices[0].message.content.strip()
        if not answer:
            answer = "No internal information found."

        answer_message = await interaction.followup.send(answer)
        await answer_message.add_reaction("✅")
        await answer_message.add_reaction("❌")

        bot.answer_cache[answer_message.id] = {
            "question": used_question,
            "answer": answer,
            "asked_by": interaction.user.id
        }

        log_channel = bot.get_channel(LOG_CHANNEL_ID)
        if log_channel:
            await log_channel.send(
                f"🧠 {interaction.user.mention} asked `/ask`: `{used_question}`\n📄 AI used: `{relevant_info.strip()}`\n📬 AI said:\n{answer}"
            )

    except Exception as e:
        await interaction.followup.send("❌ An error occurred.")
        print(e)

@bot.event
async def on_reaction_add(reaction, user):
    if user.bot:
        return

    message = reaction.message
    if message.id not in bot.answer_cache:
        return

    if reaction.emoji not in ("✅", "❌"):
        return

    entry = bot.answer_cache[message.id]
    question = entry["question"]
    answer = entry["answer"]
    asked_by_id = entry["asked_by"]

    rating = "Positive" if reaction.emoji == "✅" else "Negative"

    if rating == "Negative":
        usage_stats["negative_ratings"] += 1

    log_channel = bot.get_channel(LOG_CHANNEL_ID)
    if log_channel:
        await log_channel.send(
            f"📝 Answer rating by {user.mention}: **{rating}**\n"
            f"👤 Asked by <@{asked_by_id}>\n"
            f"❓ Question: {question}\n"
            f"💬 Answer: {answer}"
        )

@bot.tree.command(name="reload", description="Manually reload the knowledge cache.")
async def reload(interaction: discord.Interaction):
    await interaction.response.defer()
    await reload_cache()
    await interaction.followup.send("✅ Knowledge cache manually reloaded.")

@bot.tree.command(name="shutdown", description="Shutdown the bot (Role restricted).")
async def shutdown(interaction: discord.Interaction):
    if SENIOR_ROLE_ID == 0:
        await interaction.response.send_message("❌ Senior role ID not configured.", ephemeral=True)
        return

    if not isinstance(interaction.user, discord.Member):
        await interaction.response.send_message("❌ Command must be used in a server.", ephemeral=True)
        return

    has_role = any(role.id == SENIOR_ROLE_ID for role in interaction.user.roles)
    if not has_role:
        await interaction.response.send_message("❌ You do not have permission to use this command.", ephemeral=True)
        return

    await interaction.response.send_message("⚠️ Shutting down... Goodbye!", ephemeral=True)
    print(f"🛑 Shutdown initiated by {interaction.user} ({interaction.user.id})")
    
    await bot.close()
    await asyncio.sleep(1)  # Give time for close to complete
    print("Bot has closed connection, now exiting process.")
    sys.exit(0)

async def main():
    global knowledge, embeddings, index
    knowledge, embeddings, index = await get_cached_embeddings()
    await bot.start(DISCORD_BOT_TOKEN)

if __name__ == "__main__":
    asyncio.run(main())
