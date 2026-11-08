
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

MODELO_EN_ES = "Helsinki-NLP/opus-mt_tiny_eng-spa"
MODELO_ES_ZH = "Helsinki-NLP/opus-tatoeba-es-zh"

tokenizador = None
modelo = None

tokenizador_es_zh = None
modelo_es_zh = None

bloqueo_modelo = threading.Lock()


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     global tokenizador, modelo

#     logger.info("Cargando modelo de traducción: %s", MODELO_ID)

#     torch.set_num_threads(2)

#     tokenizador = MarianTokenizer.from_pretrained(MODELO_ID)
#     modelo = MarianMTModel.from_pretrained(MODELO_ID)
#     modelo.eval()

#     logger.info("Modelo cargado y listo para traducir.")

#     yield

#     logger.info("API finalizando.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizador, modelo
    global tokenizador_es_zh, modelo_es_zh

    torch.set_num_threads(2)

    # MODELO 1: INGLÉS -> ESPAÑOL
    logger.info("Cargando modelo inglés-español: %s", MODELO_EN_ES)

    tokenizador = MarianTokenizer.from_pretrained(MODELO_EN_ES)
    modelo = MarianMTModel.from_pretrained(MODELO_EN_ES)
    modelo.eval()

    logger.info("Modelo inglés-español cargado.")

    # MODELO 2: ESPAÑOL -> MANDARÍN
    logger.info("Cargando modelo español-mandarín: %s", MODELO_ES_ZH)

    tokenizador_es_zh = MarianTokenizer.from_pretrained(MODELO_ES_ZH)
    modelo_es_zh = MarianMTModel.from_pretrained(MODELO_ES_ZH)
    modelo_es_zh.eval()

    logger.info("Modelo español-mandarín cargado.")

    logger.info("Todos los modelos están listos.")

    yield

    logger.info("API finalizando.")



app = FastAPI(
    title="Traductor inglés a español",
    version="1.0.0",
    lifespan=lifespan,
)


# class SolicitudTraduccion(BaseModel):
#     texto: str = Field(min_length=1, max_length=1000)
class SolicitudTraduccion(BaseModel):
    texto: str = Field(min_length=1, max_length=1000)
    idioma_origen: str = "en"
    idioma_destino: str = "es"

# @app.get("/")
# def inicio():
#     return {
#         "estado": "activo",
#         "servicio": "Traductor inglés a español",
#         "modelo": MODELO_ID,
#     }


# @app.get("/salud")
# def salud():
#     return {
#         "estado": "listo" if modelo is not None else "cargando"
#     }


app = FastAPI(
    title="Traductor multidioma",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/")
def inicio():
    return {
        "estado": "activo",
        "servicio": "Traductor multidioma",
        "modelos": {
            "en-es": MODELO_EN_ES,
            "es-zh-CN": MODELO_ES_ZH
        }
    }


@app.get("/salud")
def salud():
    return {
        "estado": (
            "listo"
            if modelo is not None and modelo_es_zh is not None
            else "cargando"
        ),
        "ingles_espanol": modelo is not None,
        "espanol_mandarin": modelo_es_zh is not None
    }



# @app.post("/traducir")
# def traducir(solicitud: SolicitudTraduccion):
#     if modelo is None or tokenizador is None:
#         raise HTTPException(
#             status_code=503,
#             detail="El modelo todavía no está listo.",
#         )

#     ingles = solicitud.texto.strip()

#     if not ingles:
#         raise HTTPException(
#             status_code=400,
#             detail="El texto está vacío.",
#         )

#     inicio = time.perf_counter()

#     try:
#         with bloqueo_modelo:
#             entradas = tokenizador(
#                 ingles,
#                 return_tensors="pt",
#                 truncation=True,
#                 max_length=256,
#             )

#             with torch.inference_mode():
#                 resultado = modelo.generate(
#                     **entradas,
#                     max_new_tokens=128,
#                     num_beams=1,
#                 )

#             espanol = tokenizador.decode(
#                 resultado[0],
#                 skip_special_tokens=True,
#             ).strip()

#         duracion = round(time.perf_counter() - inicio, 4)

#         logger.info(
#             "Traducción completada en %.4f segundos",
#             duracion,
#         )

#         return {
#             "ingles": ingles,
#             "espanol": espanol,
#             "tiempo_segundos": duracion,
#         }

#     except Exception:
#         logger.exception("Error durante la traducción.")
#         raise HTTPException(
#             status_code=500,
#             detail="No se pudo traducir el texto.",
#         )


@app.post("/traducir")
def traducir(solicitud: SolicitudTraduccion):

    global modelo, tokenizador
    global modelo_es_zh, tokenizador_es_zh

    texto = solicitud.texto.strip()

    if not texto:
        raise HTTPException(
            status_code=400,
            detail="El texto está vacío."
        )

    origen = solicitud.idioma_origen.lower().strip()
    destino = solicitud.idioma_destino.lower().strip()

    inicio = time.perf_counter()

    try:

        with bloqueo_modelo:

            # ==========================================
            # MODELO 1: INGLÉS -> ESPAÑOL
            # ==========================================

            if origen == "en" and destino == "es":

                if modelo is None or tokenizador is None:
                    raise HTTPException(
                        status_code=503,
                        detail="Modelo inglés-español no disponible."
                    )

                entradas = tokenizador(
                    texto,
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

                duracion = round(
                    time.perf_counter() - inicio, 4
                )

                logger.info(
                    "Inglés-español completado en %.4f segundos",
                    duracion
                )

                return {
                    "ingles": texto,
                    "espanol": espanol,
                    "tiempo_segundos": duracion
                }

            # ==========================================
            # MODELO 2: ESPAÑOL -> MANDARÍN
            # ==========================================

            elif origen == "es" and destino in ("zh-cn", "zh"):

                if modelo_es_zh is None or tokenizador_es_zh is None:
                    raise HTTPException(
                        status_code=503,
                        detail="Modelo español-mandarín no disponible."
                    )

                texto_modelo = ">>cmn_Hans<< " + texto

                entradas = tokenizador_es_zh(
                    texto_modelo,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256,
                )

                with torch.inference_mode():
                    resultado = modelo_es_zh.generate(
                        **entradas,
                        max_new_tokens=128,
                        num_beams=1,
                    )

                chino = tokenizador_es_zh.decode(
                    resultado[0],
                    skip_special_tokens=True,
                ).strip()

                duracion = round(
                    time.perf_counter() - inicio, 4
                )

                logger.info(
                    "Español-mandarín completado en %.4f segundos",
                    duracion
                )

                return {
                    "espanol": texto,
                    "traduccion": chino,
                    "tiempo_segundos": duracion
                }

            else:

                raise HTTPException(
                    status_code=400,
                    detail=f"Combinación de idiomas no soportada: {origen}-{destino}"
                )

    except HTTPException:
        raise

    except Exception:

        logger.exception("Error durante la traducción.")

        raise HTTPException(
            status_code=500,
            detail="No se pudo traducir el texto."
        )