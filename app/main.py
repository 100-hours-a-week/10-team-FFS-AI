import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.core.database import check_health, close_databases, init_databases

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("start server")
    try:
        await init_databases()
        logger.info("Application startup complete")
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise

    yield

    logger.info("shut down server")
    try:
        await close_databases()
        logger.info("Application shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


app = FastAPI(
    title="KlosetLab",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    return {
        "status": "running"
    }


@app.get("/health")
async def health_check():
    health_status = await check_health()

    all_connected = all(status == "connected" for status in health_status.values())
    overall_status = "healthy" if all_connected else "degraded"

    response_data = {
        "status": overall_status,
        "services": health_status,
    }
    status_code = 200 if all_connected else 503

    return JSONResponse(content=response_data, status_code=status_code)
