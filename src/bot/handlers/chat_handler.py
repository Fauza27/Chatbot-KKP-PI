import logging

from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import ContextTypes, MessageHandler, filters

from src.bot import messages
from src.services.ai_services import chat

logger = logging.getLogger(__name__)

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    
    await update.message.reply_text(
        messages.WELCOME.format(first_name=user.first_name),
        parse_mode=ParseMode.HTML,
    )

async def handle_text_chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return

    text = (update.message.text or "").strip()
    if not text:
        return

    chat_id = update.effective_chat.id
    user_id = str(update.effective_user.id)

    try:
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

        response = chat(
            query=text,
            session_id=user_id,
        )

        reply_text = (response.get("answer", "")).strip()
        if not reply_text:
            reply_text = "Maaf, saya belum bisa menjawab sekarang."

        await update.message.reply_text(
            f"🤖 {reply_text}",
            parse_mode=ParseMode.HTML,
        )

        num_docs = response.get("num_docs", 0)
        if num_docs > 0:
            logger.info(f"Chat response sent to user {user_id}, used {num_docs} documents")

    except Exception as exc:
        logger.exception(f"Unexpected error in text chat handler for user {user_id}")
        await update.message.reply_text(
            "Maaf, terjadi kesalahan. Silakan coba lagi.",
            parse_mode=ParseMode.HTML,
        )

def build_text_chat_handler() -> MessageHandler:
    return MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_chat)