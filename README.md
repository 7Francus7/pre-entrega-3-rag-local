# Pre-entrega 3 · Biblioteca La Estación

Sistema RAG asíncrono para consultar cuatro manuales ficticios de una biblioteca. ChromaDB guarda el índice en disco. Un modelo multilingüe calcula los embeddings y un LLM selecciona una oración de los manuales. Cada respuesta incluye el archivo y el número de fragmento para poder verificarla.

## Instalación

Desde PowerShell, con Git y Python 3.12 instalados, ejecutá lo siguiente.

```powershell
git clone https://github.com/7Francus7/pre-entrega-3-rag-local.git
cd pre-entrega-3-rag-local
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Si PowerShell restringe la activación, podés reemplazar `python` por `.\.venv\Scripts\python.exe` en todos los comandos. En Linux y macOS usá `python3.12 -m venv .venv`, `source .venv/bin/activate` y `cp .env.example .env`.

## Configurar el modelo local

El ejemplo usa Qwen2.5-1.5B-Instruct en CPU mediante el servidor de llama.cpp. En Windows x64, abrí **otra terminal** y ejecutá estos comandos. Guardan el servidor y el modelo fuera del repositorio. La primera descarga del modelo ocupa unos 1,1 GB.

```powershell
$dir = New-Item -Force -ItemType Directory "$env:LOCALAPPDATA/rag-curso"
Set-Location $dir
curl.exe -fL --retry 2 -o runtime.zip https://github.com/ggml-org/llama.cpp/releases/download/b12600/llama-b12600-bin-win-cpu-x64.zip
Expand-Archive runtime.zip runtime -Force
curl.exe -fL --retry 2 -o model.gguf https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf
.\runtime\llama-server.exe -m model.gguf --alias rag-local --port 8080 -c 4096 -np 1
```

Dejá esa terminal abierta durante las consultas. En otro sistema operativo elegí el binario de llama.cpp de tu plataforma y pasale los mismos argumentos. [Binarios de llama.cpp](https://github.com/ggml-org/llama.cpp/releases/tag/b12600) · [Modelo Qwen GGUF](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF)

| Variable del archivo `.env` | Valor para ejecución local | Uso |
|---|---|---|
| `OPENAI_BASE_URL` | `http://127.0.0.1:8080/v1` | Dirección de la API compatible con OpenAI |
| `OPENAI_MODEL` | `rag-local` | Alias configurado al iniciar el servidor |
| `OPENAI_API_KEY` | `local` | Valor de ejemplo que requiere el cliente. El servidor local acepta este valor. |

Para usar OpenAI alojado, configurá `OPENAI_BASE_URL=https://api.openai.com/v1`, un modelo de chat disponible en tu cuenta y tu clave en `.env`. En ese modo, el proveedor recibe la pregunta y los fragmentos recuperados. El índice y los embeddings siguen en tu computadora. Si ya definiste estas variables en la terminal, sus valores tienen prioridad sobre `.env`.

## Indexar y consultar

En la terminal del proyecto ejecutá estos comandos en orden.

```powershell
python ingesta.py
python rag.py "¿Cuánto dura el préstamo a domicilio?"
python demo.py
python -m unittest -v
```

La primera ingesta descarga el modelo de embeddings y el vocabulario del tokenizador. Requiere conexión a internet. Podés volver a ejecutar la ingesta después de editar `data/`. Si el corpus coincide, reutiliza el índice existente.

`demo.py` llama dos veces a `get_rag_response` con `asyncio.run(main())` y termina con error si falla alguna comprobación.

| Pregunta de la demo | Resultado esperado |
|---|---|
| ¿Cuánto dura el préstamo a domicilio? | Una cita con «17 días corridos» y la referencia a `prestamos.md`. |
| ¿Cuál es la contraseña del wifi de la biblioteca? | El texto exacto «No lo sé» y `referencias: []`. Los manuales no indican esa contraseña. |

Los tests unitarios simulan la salida del LLM y corren sin servidor ni credenciales. Cubren un JSON mal formado seguido de una respuesta válida, dos fallos consecutivos, citas inventadas, preguntas vacías, listas de referencias inconsistentes y rechazo de campos adicionales.

## Cómo funciona

1. **Ingesta.** `ingesta.py` lee los archivos UTF-8 de `data/`. Usa `RecursiveCharacterTextSplitter` con presupuesto de 550 tokens medidos con `cl100k_base`, solapamiento objetivo de 50 tokens y cortes en límites de palabras. Los documentos cortos y el último fragmento pueden medir menos. Cada ID combina ruta y número de fragmento, por ejemplo `prestamos.md#1`.
2. **Persistencia.** La huella del corpus, el modelo y el esquema de fragmentación identifican la colección en `vectorstore/`. La ingesta activa la nueva colección después de guardar todos sus fragmentos. Las versiones anteriores se conservan en disco.
3. **Recuperación.** FastEmbed usa `minishlab/potion-multilingual-128M` tanto para documentos como para consultas. Chroma devuelve los tres fragmentos más próximos por distancia coseno. La búsqueda local corre en un hilo para liberar el bucle asíncrono.
4. **Generación.** La cadena LCEL recupera el contexto, arma el prompt y llama al LLM con `.ainvoke()`. Un `PydanticOutputParser` valida `texto` y `referencias`. Para responder, el modelo debe copiar una cita literal presente en cada fragmento citado. Si falta evidencia, debe abstenerse. El cotejo literal comprueba la procedencia de la cita. La selección de una cita pertinente depende del modelo.
5. **Errores y logs.** Un error de formato, esquema o cita admite un segundo intento. Si vuelve a fallar, la excepción llega al llamador. Los logs registran el identificador del pedido, el intento, la duración y las referencias aceptadas. Para observar el reintento controlado, ejecutá `python -m unittest -v test_rag.PruebasRAG.test_agota_dos_intentos`.

`.env`, los modelos descargados, el entorno virtual y la base vectorial quedan excluidos de Git.

