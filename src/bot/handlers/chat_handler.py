import logging
from datetime import datetime

from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import ContextTypes, MessageHandler, filters
from supabase import create_client

from config.settings import get_settings
from src.bot import messages
from src.services.ai_services import chat

logger = logging.getLogger(__name__)

DAILY_LIMIT = 13

def check_and_update_quota(user_id: str) -> bool:
    settings = get_settings()
    supabase = create_client(settings.supabase_url, settings.supabase_service_key)
    
    today = datetime.now().strftime("%Y-%m-%d")
    
    try:
        response = supabase.table("user_quotas").select("*").eq("user_id", user_id).eq("date", today).execute()
        data = response.data
        
        if not data:
            supabase.table("user_quotas").insert({
                "user_id": user_id,
                "date": today,
                "message_count": 1
            }).execute()
            return True
        
        current_count = data[0]["message_count"]
        if current_count >= DAILY_LIMIT:
            return False
            
        supabase.table("user_quotas").update({
            "message_count": current_count + 1
        }).eq("user_id", user_id).eq("date", today).execute()
        
        return True
    except Exception as e:
        logger.error(f"Error checking quota for user {user_id}: {e}")
        # Jika gagal cek database, beri izin agar tidak memblokir sistem
        return True

def log_chat_to_db(user_id: str, username: str, question: str, answer: str) -> None:
    try:
        settings = get_settings()
        supabase = create_client(settings.supabase_url, settings.supabase_service_key)
        
        supabase.table("chat_logs").insert({
            "user_id": user_id,
            "username": username,
            "question": question,
            "answer": answer
        }).execute()
    except Exception as e:
        logger.error(f"Gagal menyimpan log chat untuk user {user_id}: {e}")

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

    # Cek limit harian sebelum memproses pertanyaan
    has_quota = check_and_update_quota(user_id)
    if not has_quota:
        await update.message.reply_text(
            "⚠️ <b>Batas Limit Harian Tercapai</b>\n\n"
            "Maaf, Anda telah menggunakan jatah 13 pertanyaan untuk hari ini "
            "guna menghemat biaya server. Silakan kembali lagi besok hari ya! 🎓",
            parse_mode=ParseMode.HTML
        )
        return

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

        # Catat ke database untuk monitoring
        username = update.effective_user.username or update.effective_user.first_name or "Unknown"
        # Gunakan bot.loop (jika ada) atau panggil secara sinkron, supabase-py by default sinkron
        log_chat_to_db(user_id, username, text, reply_text)

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