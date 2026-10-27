
import os
import time
import threading
import logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import MarianMTModel, MarianTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TraductorAPI")

MODELO_ID = "Helsinki-NLP/opus-mt_tiny_eng-spa"

tokenizador = None
modelo = None
bloqueo_modelo = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizador, modelo

    logger.info("Cargando modelo de traducción: %s", MODELO_ID)

    torch.set_num_threads(2)

    tokenizador = MarianTokenizer.from_pretrained(MODELO_ID)
    modelo = MarianMTModel.from_pretrained(MODELO_ID)
    modelo.eval()

    logger.info("Modelo cargado y listo para traducir.")

    yield

    logger.info("API finalizando.")


app = FastAPI(
    title="Traductor inglés a español",
    version="1.0.0",
    lifespan=lifespan,
)


class SolicitudTraduccion(BaseModel):
    texto: str = Field(min_length=1, max_length=1000)


@app.get("/")
def inicio():
    return {
        "estado": "activo",
        "servicio": "Traductor inglés a español",
        "modelo": MODELO_ID,
    }


@app.get("/salud")
def salud():
    return {
        "estado": "listo" if modelo is not None else "cargando"
    }


@app.post("/traducir")
def traducir(solicitud: SolicitudTraduccion):
    if modelo is None or tokenizador is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo todavía no está listo.",
        )

    ingles = solicitud.texto.strip()

    if not ingles:
        raise HTTPException(
            status_code=400,
            detail="El texto está vacío.",
        )

    inicio = time.perf_counter()

    try:
        with bloqueo_modelo:
            entradas = tokenizador(
                ingles,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            )

            with torch.inference_mode():
                resultado = modelo.generate(
                    **entradas,
                    max_new_tokens=128,
                    num_beams=1,
                )

            espanol = tokenizador.decode(
                resultado[0],
                skip_special_tokens=True,
            ).strip()

        duracion = round(time.perf_counter() - inicio, 4)

        logger.info(
            "Traducción completada en %.4f segundos",
            duracion,
        )

        return {
            "ingles": ingles,
            "espanol": espanol,
            "tiempo_segundos": duracion,
        }

    except Exception:
        logger.exception("Error durante la traducción.")
        raise HTTPException(
            status_code=500,
            detail="No se pudo traducir el texto.",
        )
