import logging
from contextlib import asynccontextmanager

from slowapi import Limiter
from slowapi.util import get_remote_address

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from telegram import Update

from config.settings import get_settings
from src.bot.application import create_bot, post_init
from src.api import ai

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    if settings.TELEGRAM_WEBHOOK_URL:
        bot_app = create_bot()
        await bot_app.initialize()

        await post_init(bot_app)

        webhook_url = f"{settings.TELEGRAM_WEBHOOK_URL}{settings.TELEGRAM_WEBHOOK_PATH}"
        await bot_app.bot.set_webhook(
            url=webhook_url,
            secret_token=settings.TELEGRAM_WEBHOOK_SECRET or None,
            drop_pending_updates=True,
        )
        await bot_app.start()

        app.state.bot_app = bot_app
    
    yield

    if hasattr(app.state, "bot_app"):
        try:
            await app.state.bot_app.stop()
            await app.state.bot_app.shutdown()
        except Exception:
            logger.exception("Error shutting down Telegram bot")

def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.VERSION,
        description="chatbot assistan yang mampu menjawab pertanyaan terkai kkp/pi",
        docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
        redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
        lifespan=lifespan,
    )

    limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])
    
    app.state.limiter = limiter

    _register_middleware(app, settings)
    _register_routers(app)

    return app

def _register_middleware(app: FastAPI, settings):
    app.add_middleware(SlowAPIMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

def _register_routers(app: FastAPI):
    API_PREFIX = "/api"

    app.include_router(ai.router, prefix=API_PREFIX)


    @app.post(
        "/api/telegram/webhook",
        tags=["Telegram"],
        summary="Telegram webhook receiver",
        include_in_schema=False,
    )
    async def telegram_webhook(request: Request):
        settings = get_settings()

        if settings.TELEGRAM_WEBHOOK_SECRET:
            incoming_token = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
            if incoming_token != settings.TELEGRAM_WEBHOOK_SECRET:
                raise HTTPException(status_code=403, detail="Invalid secret token")

        if not hasattr(request.app.state, "bot_app"):
            raise HTTPException(status_code=503, detail="Bot not initialized")

        data = await request.json()
        bot_app = request.app.state.bot_app
        update = Update.de_json(data=data, bot=bot_app.bot)
        await bot_app.process_update(update)

        return JSONResponse(content={"ok": True})

    @app.get("/health", tags=["System"], summary="Health check")
    async def health_check():
        settings = get_settings()
        health_status = {
            "status": "healthy",
            "app": settings.APP_NAME,
            "version": settings.VERSION,
            "environment": settings.ENVIRONMENT,
        }

        if hasattr(app.state, "bot_app"):
            health_status["telegram_bot"] = "running"
        else:
            health_status["telegram_bot"] = "not started"
        
        return health_status
    
    @app.get("/")
    async def root():
        settings = get_settings()
        return {
            "message": f"Welcome to {settings.APP_NAME}",
            "version": settings.VERSION,
            "docs": "/docs"
        }