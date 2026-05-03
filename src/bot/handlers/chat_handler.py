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
        
        # Kirim pesan loading sementara
        loading_message = await update.message.reply_text(
            "⏳ Sedang mencari jawaban...",
            parse_mode=ParseMode.HTML,
        )

        response = chat(
            query=text,
            session_id=user_id,
        )

        reply_text = (response.get("answer", "")).strip()
        if not reply_text:
            reply_text = "Maaf, saya belum bisa menjawab sekarang."

        sources = response.get("sources", [])
        if sources:
            source_text = "\n\n📚 Sumber:\n"
            for s in sources:
                section = s.get("section", "")
                title = s.get("title", "")
                
                # Biar rapi, gabungkan section dan title jika beda
                if section and title and section != title:
                    source_name = f"{section} — {title}"
                else:
                    source_name = title or section or "Buku Panduan"
                
                # Cek panduan mana
                parent_id = s.get("parent_id", "")
                panduan_type = "PI" if "pi" in parent_id.lower() else "KKP"
                
                source_text += f"  • {source_name} (Buku Panduan {panduan_type})\n"
            
            reply_text += source_text

        # Update pesan loading dengan jawaban akhir
        await loading_message.edit_text(
            f"🤖 {reply_text}",
            parse_mode=ParseMode.HTML,
        )

        num_docs = response.get("num_docs", 0)
        if num_docs > 0:
            logger.info(f"Chat response sent to user {user_id}, used {num_docs} documents")

    except Exception as exc:
        logger.exception(f"Unexpected error in text chat handler for user {user_id}")
        error_text = "Maaf, terjadi kesalahan. Silakan coba lagi."
        
        try:
            if 'loading_message' in locals():
                await loading_message.edit_text(error_text, parse_mode=ParseMode.HTML)
            else:
                await update.message.reply_text(error_text, parse_mode=ParseMode.HTML)
        except Exception:
            pass

def build_text_chat_handler() -> MessageHandler:
    return MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_chat)