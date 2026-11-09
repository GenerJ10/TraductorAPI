

import time
import threading
import logging
from contextlib import asynccontextmanager

import ctranslate2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import MarianTokenizer
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TraductorAPI")

MODELO_ID = "Helsinki-NLP/opus-tatoeba-es-zh"
MODELO_CT2 = "GENERJOSE10/modelo-es-zh-int8"

tokenizador = None
traductor = None
bloqueo_modelo = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizador, traductor

    logger.info("Cargando tokenizador español-mandarín...")

    tokenizador = MarianTokenizer.from_pretrained(MODELO_ID)

    logger.info("Descargando modelo CTranslate2 INT8...")

    ruta_modelo = snapshot_download(
        repo_id=MODELO_CT2
    )

    logger.info("Cargando modelo INT8...")

    traductor = ctranslate2.Translator(
        ruta_modelo,
        device="cpu",
        compute_type="int8",
        inter_threads=1,
        intra_threads=2
    )

    logger.info("Modelo español-mandarín INT8 listo.")

    yield

    logger.info("API finalizando.")


app = FastAPI(
    title="Traductor espanol a mandarin",
    version="1.0.0",
    lifespan=lifespan,
)


class SolicitudTraduccion(BaseModel):
    texto: str = Field(min_length=1, max_length=1000)


@app.get("/")
def inicio():
    return {
        "estado": "activo",
        "servicio": "Traductor español a mandarín INT8",
        "modelo": MODELO_ID,
        "modelo_ct2": MODELO_CT2,
    }


@app.get("/salud")
def salud():
    return {
        "estado": "listo" if traductor is not None else "cargando"
    }


@app.post("/traducir")
def traducir(solicitud: SolicitudTraduccion):

    if traductor is None or tokenizador is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo todavía no está listo.",
        )

    espanol = solicitud.texto.strip()

    if not espanol:
        raise HTTPException(
            status_code=400,
            detail="El texto está vacío.",
        )

    inicio = time.perf_counter()

    try:
        with bloqueo_modelo:

            texto_modelo = ">>cmn_Hans<< " + espanol

            ids = tokenizador.encode(
                texto_modelo,
                add_special_tokens=True
            )

            tokens = tokenizador.convert_ids_to_tokens(ids)

            resultado = traductor.translate_batch(
                [tokens],
                beam_size=1,
                max_decoding_length=128
            )

            tokens_chinos = resultado[0].hypotheses[0]

            ids_chinos = tokenizador.convert_tokens_to_ids(
                tokens_chinos
            )

            chino = tokenizador.decode(
                ids_chinos,
                skip_special_tokens=True
            ).strip()

        duracion = round(time.perf_counter() - inicio, 4)

        logger.info(
            "Traducción INT8 español-mandarín completada en %.4f segundos",
            duracion,
        )

        return {
            "espanol": espanol,
            "traduccion": chino,
            "tiempo_segundos": duracion,
        }

    except Exception:
        logger.exception("Error durante la traducción.")

        raise HTTPException(
            status_code=500,
            detail="No se pudo traducir el texto.",
        )
