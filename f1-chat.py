import streamlit as st
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig, Tool, FunctionDeclaration, Part
import google.auth
import requests

# ----------------------------------
#  Definición de entorno
# ----------------------------------

# ----------------------------------
#  Configuración del nivel de Logs
# ----------------------------------
import warnings, os
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ----------------------------------
#  Definición de Funciones
# ----------------------------------

# --- Función Python que ejecutará el código
# --- Nos conectaríamos con una API real. Pero nosotros simulamos la respuesta.
def obtener_ultima_carrera(dummy: str = "ignorar"):
    """
    Consulta la API pública de F1 para obtener el ganador de la última carrera disputada.
    El parámetro 'dummy' es solo para que el agente tenga algo que pasar, aunque no se usa.
    """
    try:
        url = "https://ergast.com/api/f1/current/last.json"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()

            carrera = data["MRData"]["RaceTable"]["Races"][0]
            nombre_gp = carrera["raceName"]
            fecha = carrera["date"]

            return {
                "gran_premio": nombre_gp,
                "fecha": fecha,
            }

    except Exception as e:
        return f"Error al consultar la API: {str(e)}"

# --- Diccionario para mapear el nombre de la función con la ejecución real
herramientas_mapeo = {
    "obtener_ultima_carrera": obtener_ultima_carrera
}

# --- Creación de las herramientas para el Modelo (Tool)
herramientas_agente = Tool(
    function_declarations = [
        FunctionDeclaration.from_func(obtener_ultima_carrera)
    ]
)

# ----------------------------------
#  Configuración del Modelo en GCP
# ----------------------------------

contexto = """
Eres un comentarista deportivo experto en Fórmula 1.
Tu objetivo es informar a los fans sobre las carreras y estadísticas.

INSTRUCCIONES:
1. Si el usuario pregunta por el "ganador de la última carrera", "último resultado" o "cómo quedó la carrera", DEBES usar la herramienta 'obtener_ultima_carrera_f1'.
2. Si te preguntan cosas generales de F1 (historia, reglas, pilotos históricos como Senna o Schumacher), responde con tu propio conocimiento sin usar herramientas.
3. Si preguntan algo que no es deporte, responde educadamente que solo hablas de F1.
"""

parametros = {
    "temperature": 0.3,
    "max_output_tokens": 8192,
}

try:
    credentials, project_id = google.auth.default()
    vertexai.init(project=PROJECT_ID, location=REGION)
    model = GenerativeModel(
        MODEL_NAME,
        system_instruction = contexto,
        generation_config = GenerationConfig(**parametros),
        tools=[herramientas_agente]
    )
    estado = f"🟢 Conectado a {PROJECT_ID} ({MODEL_NAME})"
except Exception as e:
    estado = f"🔴 Error: {str(e)}"
    model = None

# ----------------------------------
# Diseño de la interfaz de usuario
# ----------------------------------

st.set_page_config(page_title="Chatbot con Gemini", page_icon="⚙️", layout="centered")
st.title("🧠 Chat con Gemini (Vertex-Agent)")
st.caption(estado)

if "chat" not in st.session_state and model:
    st.session_state.chat = model.start_chat()

if model:
    prompt = st.text_area("Introduce tu prompt:")
    if st.button("Generar Respuesta"):
        # --- Mostrar historial previo
        if prompt.strip():
            with st.chat_message("user"):
                st.write(prompt)
            # ----------------------------------
            #  Lógica del Agente
            # ----------------------------------
            with st.chat_message("assistant"):
                with st.spinner("El agente está pensando..."):
                    try:
                        # --- Envío del mensaje al modelo
                        response = st.session_state.chat.send_message(prompt)
                        # --- Comprobación de si el Agente necesita usar una herramienta (Function Calling)
                        part = response.candidates[0].content.parts[0]
                        if part.function_call:
                            st.info(f"🛠️ El agente está usando la herramienta: `{part.function_call.name}`")
                            # --- Extraer nombre de la función y argumentos
                            fn_name = part.function_call.name
                            fn_args = {key: val for key, val in part.function_call.args.items()}
                            # --- Ejecutar la función real de Python
                            if fn_name in herramientas_mapeo:
                                api_response = herramientas_mapeo[fn_name](**fn_args)
                                # --- Devolver el resultado de la función al modelo para que genere la respuesta final
                                response = st.session_state.chat.send_message(
                                    Part.from_function_response(
                                        name=fn_name,
                                        response={
                                            "content": api_response,
                                        },
                                    )
                                )
                        # --- Mostrar la respuesta final (texto natural)
                        st.write(response.text)
                    except Exception as e:
                        st.error(f"Error en la lógica del agente: {str(e)}")
else:
    st.error("Error de inicialización. Revisa logs.")
