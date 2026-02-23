# Asistente de Finanzas Personales

Asistente construido con **LangChain + Streamlit** usando **Gemini 2.5 Flash** como cerebro, `yfinance` para mercado y **Tavily** para noticias recientes.

## Funcionalidades

- Análisis completo de activos con combinación de:
  - Precio actual (`yfinance`)
  - Indicadores técnicos (SMA, RSI, MACD, volatilidad, ATR, Bollinger, ADX, nivel de riesgo)
  - Snapshot fundamental (valoración, crecimiento, deuda, márgenes, beta)
  - Calendario de resultados (`earnings`) y catalizadores temporales
  - Noticias recientes (`Tavily API`)
  - Señal simple de sentimiento de noticias (positivo/neutral/negativo)
- Validación lineal de calidad de datos (`DataValidator`) con reintentos automáticos básicos.
- Scoring de noticias con LLM (Gemini) para relevancia, impacto y horizonte.
- `confidence_report` (score 0-100 + etiqueta) para contextualizar la fiabilidad del análisis.
- Calculadora de ahorro con interés compuesto.
- Flujo interno (LangChain):
  - `Analyst`: decide y llama tools.
  - `Tools`: ejecuta Python para datos de mercado/noticias/cálculo.
  - `Advisor`: redacta consejo amigable y preventivo.
- Memoria por hilo con `InMemoryChatMessageHistory` para recordar contexto (por ejemplo, nombre del usuario).
- Interfaz Streamlit con paleta:
  - `#2b2d39`
  - `#73c59b`

## Requisitos

- Python 3.10+
- `GOOGLE_API_KEY` (Gemini)
- `TAVILY_API_KEY` (noticias)

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuración segura

1. Copia el archivo de ejemplo de entorno:

```bash
cp .env.example .env
```

2. Añade tus claves en `.env` (no se sube al repo):

- `GOOGLE_API_KEY`
- `TAVILY_API_KEY`

3. Si usas Streamlit secrets, guarda credenciales en `.streamlit/secrets.toml` (también ignorado por Git).

4. Activa validaciones locales antes de cada commit:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## Ejecución

```bash
streamlit run app.py
```

En la barra lateral introduce:
- `GOOGLE_API_KEY`
- `TAVILY_API_KEY`

## Seguridad del repositorio

Este repositorio incluye:
- `.gitignore` endurecido para evitar subir secretos y entornos locales.
- `SECURITY.md` para reporte responsable de vulnerabilidades.
- `Dependabot` para actualización de dependencias.
- Workflows de GitHub Actions para:
  - `CodeQL` (análisis estático)
  - `Bandit` + `pip-audit` (seguridad Python/dependencias)
  - `Gitleaks` (detección de secretos)

## Ejemplos de uso

- `Analiza AAPL con técnico y noticias recientes`
- `Dame análisis de BTC-USD`
- `Calcula ahorro 10000 al 4.5% durante 15 años`

## Estructura

- `app.py`: aplicación principal (tools, orquestación LangChain y UI)
- `requirements.txt`: dependencias

## Nota legal

El asistente es educativo y no sustituye asesoramiento financiero profesional.
