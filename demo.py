"""Dos consultas reales contra el servidor configurado en .env."""
import asyncio
import logging

from rag import UNKNOWN, get_rag_response


async def main():
    casos = [
        ("¿Cuánto dura el préstamo a domicilio?", "17 días corridos", "prestamos.md#"),
        ("¿Cuál es la contraseña del wifi de la biblioteca?", UNKNOWN, None),
    ]
    for pregunta, esperado, fuente in casos:
        respuesta = await get_rag_response(pregunta)
        print(pregunta, respuesta.model_dump_json(indent=2), sep="\n", flush=True)
        if fuente:
            if esperado not in respuesta.texto or not any(
                ref.startswith(fuente) for ref in respuesta.referencias
            ):
                raise AssertionError("Falló la respuesta documentada o su referencia.")
        elif respuesta.texto != UNKNOWN or respuesta.referencias:
            raise AssertionError("El modelo debía abstenerse sin citar documentos.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    asyncio.run(main())
