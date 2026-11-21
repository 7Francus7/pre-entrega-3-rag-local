import unittest

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from pydantic import ValidationError

from rag import Respuesta, crear_cadena, get_rag_response

CITA = "El préstamo dura 17 días."
VALIDO = '{"texto":"El préstamo dura 17 días.","referencias":["manual.md#1"]}'


class PruebasRAG(unittest.IsolatedAsyncioTestCase):
    def cadena(self, *salidas):
        return crear_cadena(FakeListChatModel(responses=list(salidas)),
                            lambda _: {"manual.md#1": CITA})

    async def test_reintento_json_malformado(self):
        with self.assertLogs("rag", level="INFO") as logs:
            respuesta = await self.cadena("JSON inválido", VALIDO).ainvoke("¿Qué plazo hay?")
        self.assertEqual(respuesta.texto, CITA)
        self.assertIn("intento=1", logs.output[0])
        self.assertIn("intento=2", logs.output[1])

    async def test_agota_dos_intentos(self):
        with self.assertRaises(OutputParserException):
            await self.cadena("inválido", "inválido").ainvoke("¿Qué plazo hay?")

    async def test_cita_inventada(self):
        for invalido in (VALIDO.replace("manual.md", "falso.md"), VALIDO.replace("17", "25")):
            with self.subTest(invalido=invalido), self.assertRaises(ValueError):
                await self.cadena(invalido).ainvoke("¿Qué plazo hay?")

    async def test_pregunta_vacia(self):
        with self.assertRaises(ValueError):
            await get_rag_response("  ")

    def test_esquema_estricto(self):
        for datos in [
            dict(texto="No lo sé", referencias=[], confianza=1),
            dict(texto="No lo sé", referencias=["manual.md#1"]),
            dict(texto=CITA, referencias=[]),
            dict(texto=CITA, referencias="manual.md#1"),
            dict(texto="", referencias=[]),
        ]:
            with self.subTest(datos=datos), self.assertRaises(ValidationError):
                Respuesta(**datos)

