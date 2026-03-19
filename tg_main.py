import asyncio
import logging
import os
import argparse
from typing import Dict

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove
from dotenv import load_dotenv

from pipeline.chatbot import VideoState, build_chatbot_tools, _get_llm, SYSTEM_PROMPT
from pipeline.formatting import md_to_tg_html
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# --- ССЫЛКИ НА ФОТО ---
PHOTO_WELCOME = "https://i.postimg.cc/VNxNXybh/welcome-Pic.png"
PHOTO_PROCESSING = "https://i.postimg.cc/KvpxnX64/proces-Pic.png"
PHOTO_FINISH = "https://i.postimg.cc/4xM4TCwf/bye-Pic.png"
PHOTO_ERROR = "https://i.postimg.cc/qMPZfjnW/error-Pic.png"

def parse_bot_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-dir", default="output")
    parser.add_argument("--whisper-model", default="large-v3-turbo-q5_0")
    parser.add_argument("--llm-model", default="nvidia/nemotron-3-nano-30b-a3b:free") 
    parser.add_argument("--embedding-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--language", default="ru")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cookies-from-browser", default="chrome")
    return parser.parse_args()

args = parse_bot_args()
bot = Bot(token=os.getenv("TELEGRAM_TOKEN"))
dp = Dispatcher()

# --- КЛАВИАТУРЫ ---

def get_main_kb():
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="🔄 Новое видео"), KeyboardButton(text="🛑 Остановить выполнение")],
            [KeyboardButton(text="🏁 Завершить работу")]
        ],
        resize_keyboard=True,
        persistent=True
    )

def get_analysis_kb():
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="📝 Саммари")],
            [KeyboardButton(text="🔄 Новое видео"), KeyboardButton(text="🏁 Завершить работу")]
        ],
        resize_keyboard=True,
        persistent=True
    )

# --- СОСТОЯНИЯ ---

user_contexts: Dict[int, dict] = {}

async def safe_edit_status(chat_id: int, user_id: int, text: str):
    ctx = user_contexts.get(user_id)
    if ctx and ctx.get("status_msg_id"):
        try:
            await bot.edit_message_text(
                chat_id=chat_id, 
                message_id=ctx["status_msg_id"], 
                text=f"⏳ Статус: {text}"
            )
        except Exception: pass

def create_ctx(user_id: int, chat_id: int):
    state = VideoState()
    loop = asyncio.get_event_loop()
    def status_cb(text: str):
        asyncio.run_coroutine_threadsafe(safe_edit_status(chat_id, user_id, text), loop)
    state.status_callback = status_cb
    llm = _get_llm(args.llm_model)
    tools = build_chatbot_tools(state, args)
    agent = create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)
    ctx = {"agent": agent, "state": state, "history": [], "status_msg_id": None}
    user_contexts[user_id] = ctx
    return ctx

# --- ХЕНДЛЕРЫ ---

@dp.message(Command("start"))
async def cmd_start(message: Message):
    user_id = message.from_user.id
    create_ctx(user_id, message.chat.id)
    await message.answer_photo(
        photo=PHOTO_WELCOME,
        caption="👋 Привет! Я твой ИИ-ассистент.\n\nПришли мне ссылку на видео, и я его проанализирую.",
        parse_mode="Markdown",
        reply_markup=get_main_kb()
    )

@dp.message(F.text == "🔄 Новое видео")
async def cmd_new(message: Message):
    user_id = message.from_user.id
    if user_id in user_contexts:
        user_contexts[user_id]["state"].is_stopped = True
    create_ctx(user_id, message.chat.id)
    # БЕЗ ФОТО
    await message.answer(
        "🧹 Контекст очищен.\nЖду новую ссылку на YouTube видео!",
        reply_markup=get_main_kb()
    )

@dp.message(F.text == "🛑 Остановить выполнение")
async def cmd_stop(message: Message):
    user_id = message.from_user.id
    if user_id in user_contexts:
        user_contexts[user_id]["state"].is_stopped = True
        create_ctx(user_id, message.chat.id)
    # БЕЗ ФОТО
    await message.answer(
        "🛑 Выполнение прервано.\nЯ готов к работе с новым видео.",
        reply_markup=get_main_kb()
    )

@dp.message(F.text == "🏁 Завершить работу")
async def cmd_finish(message: Message):
    user_id = message.from_user.id
    if user_id in user_contexts:
        user_contexts[user_id]["state"].is_stopped = True
    user_contexts.pop(user_id, None)
    await message.answer_photo(
        photo=PHOTO_FINISH,
        caption="🏁 Работа завершена.\nЧтобы начать заново, нажми /start",
        reply_markup=ReplyKeyboardRemove()
    )

@dp.message(F.text == "📝 Саммари")
async def handle_summary(message: Message):
    user_id = message.from_user.id
    ctx = user_contexts.get(user_id)
    if not ctx: return await message.answer("⚠️ Нажми /start")
    if not ctx["state"].processed_url: return await message.answer("⚠️ Пришли ссылку!")

    await handle_logic(message, ctx, "Сделай краткое содержание видео")

@dp.message(F.text.regexp(r'(https?://[^\s]+)'))
async def handle_link(message: Message):
    user_id = message.from_user.id
    ctx = user_contexts.get(user_id)
    if not ctx: return await message.answer("⚠️ Нажми /start")

    if ctx["state"].processed_url:
        ctx = create_ctx(user_id, message.chat.id)

    await message.answer_photo(
        photo=PHOTO_PROCESSING,
        caption="🚀 Ссылка принята в работу!\nСейчас я изучу содержание. Пожалуйста, подожди..."
    )

    status_msg = await message.answer("⏳ Подключаюсь к YouTube...")
    ctx["status_msg_id"] = status_msg.message_id
    
    try:
        ctx["history"].append(HumanMessage(content=message.text.strip()))
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: ctx["agent"].invoke({"messages": ctx["history"]}))
        
        if ctx["state"].is_stopped: return

        ctx["history"] = result["messages"]
        await message.answer(
            md_to_tg_html(ctx["history"][-1].content),
            parse_mode="HTML",
            reply_markup=get_analysis_kb(),
        )
        try:
            await bot.delete_message(message.chat.id, ctx["status_msg_id"])
        except: pass
    except Exception as e:
        if not ctx["state"].is_stopped:
            # ДОБАВЛЕНО ФОТО ПРИ ОШИБКЕ
            await message.answer_photo(
                photo=PHOTO_ERROR,
                caption=f"❌ Произошла ошибка при анализе ссылки: {e}"
            )

@dp.message()
async def handle_question(message: Message):
    user_id = message.from_user.id
    ctx = user_contexts.get(user_id)
    if not ctx or not ctx["state"].processed_url:
        return await message.answer("⚠️ Сначала нажми /start и пришли ссылку.")
    await handle_logic(message, ctx, message.text)

async def handle_logic(message: Message, ctx: dict, text: str):
    thinking_msg = await message.answer("⏳ Обрабатываю ваш вопрос...")
    await bot.send_chat_action(message.chat.id, "typing")
    loop = asyncio.get_running_loop()
    try:
        ctx["history"].append(HumanMessage(content=text))
        result = await loop.run_in_executor(None, lambda: ctx["agent"].invoke({"messages": ctx["history"]}))
        ctx["history"] = result["messages"]
        await bot.edit_message_text(
            chat_id=message.chat.id,
            message_id=thinking_msg.message_id,
            text=md_to_tg_html(ctx["history"][-1].content),
            parse_mode="HTML",
        )
    except Exception as e:
        # ПРИ ОШИБКЕ В Q&A УДАЛЯЕМ "ДУМАЮ" И ШЛЕМ КАРТИНКУ
        try: await bot.delete_message(message.chat.id, thinking_msg.message_id)
        except: pass
        await message.answer_photo(
            photo=PHOTO_ERROR,
            caption=f"😵 Ошибка при генерации ответа: {e}"
        )

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())