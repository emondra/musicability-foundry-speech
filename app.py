"""
MusicAbility – MVP Streamlit
Accesibilidad motora: el usuario escribe una instrucción y obtiene un MIDI simple.
"""

import io
import json
import os
import re
import struct
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

# Carga .env siempre relativo al archivo, sin depender del CWD
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

FOUNDRY_API_KEY  = os.getenv("FOUNDRY_API_KEY", "")
FOUNDRY_ENDPOINT = os.getenv("FOUNDRY_ENDPOINT", "").rstrip("/")
MODEL_DEPLOYMENT = os.getenv("MODEL_DEPLOYMENT_NAME", "")

# Campos mínimos que el JSON del modelo debe contener
REQUIRED_FIELDS = {"title", "tempo_bpm", "key", "length_bars",
                   "time_signature", "melody"}

# Rango MIDI permitido: C3 (48) – C5 (72)
PITCH_MIN = 48
PITCH_MAX = 72

# ─────────────────────────────────────────────────────────────────────────────
# 2. PROMPT DEL SISTEMA
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
Eres un compositor asistido por IA especializado en accesibilidad musical.
El usuario describe una idea musical. Debes devolver ÚNICAMENTE un objeto JSON
válido, sin texto adicional, sin bloques de código, sin explicaciones.

Esquema obligatorio:
{
  "title": "string",
  "tempo_bpm": int,
  "key": "string",
  "length_bars": int,
  "time_signature": "string",
  "melody": [
    {
      "pitch": "string",
      "start_beat": float,
      "duration_beats": float,
      "velocity": int
    }
  ],
  "assumptions": ["string"]
}

Restricciones:
- Rango de notas: C3 a C5 exclusivamente (pitch como "C4", "D#4", "Bb3").
- Melodía cantable: evita saltos mayores a una sexta (9 semitonos) consecutivos.
- length_bars máximo 8 si el usuario no indica otro valor.
- tempo_bpm entre 40 y 200. Default 90.
- velocity entre 40 y 110. Default 80.
- time_signature default "4/4".
- SOLO devuelves JSON válido, nada más.
""".strip()

# ─────────────────────────────────────────────────────────────────────────────
# 3. INTEGRACIÓN CON AZURE AI FOUNDRY
# ─────────────────────────────────────────────────────────────────────────────

def _clean_model_response(raw: str) -> str:
    """Elimina bloques ```json ... ``` que el modelo pueda agregar."""
    raw = raw.strip()
    match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    if raw.startswith("{"):
        return raw
    match = re.search(r"(\{.*\})", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw


def call_foundry_for_music_json(user_text: str) -> dict:
    """
    Llama a Azure AI Foundry (chat completions) con el texto del usuario.
    Devuelve el dict Python del JSON musical o lanza ValueError / RuntimeError.
    """
    if not FOUNDRY_API_KEY or not FOUNDRY_ENDPOINT or not MODEL_DEPLOYMENT:
        raise RuntimeError(
            "Variables FOUNDRY_API_KEY, FOUNDRY_ENDPOINT o "
            "MODEL_DEPLOYMENT_NAME no están configuradas."
        )

    url = (
        f"{FOUNDRY_ENDPOINT}"
        f"/openai/deployments/{MODEL_DEPLOYMENT}"
        f"/chat/completions?api-version=2024-10-21"
    )
    headers = {
        "Content-Type": "application/json",
        "api-key": FOUNDRY_API_KEY,
    }
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_text},
        ],
        # gpt-5-nano es un modelo de razonamiento: usa max_completion_tokens
        # (incluye reasoning tokens internos + tokens de respuesta).
        # Con 8000 hay margen suficiente para generar el JSON musical completo.
        "max_completion_tokens": 8000,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=90)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(
            f"Error HTTP {resp.status_code}: {resp.text[:400]}\n\n"
            f"URL llamada: `{url}`"
        ) from e
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error de conexión: {e}") from e

    raw_content = resp.json()["choices"][0]["message"]["content"]
    clean = _clean_model_response(raw_content)

    try:
        data = json.loads(clean)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"El modelo no devolvió JSON válido.\n"
            f"Respuesta recibida:\n{raw_content[:500]}\n\nError: {e}"
        )

    missing = REQUIRED_FIELDS - set(data.keys())
    if missing:
        raise ValueError(f"Faltan campos en el JSON: {', '.join(sorted(missing))}")

    if not isinstance(data["melody"], list) or len(data["melody"]) == 0:
        raise ValueError("El campo 'melody' debe ser una lista no vacía.")

    return data


# ─────────────────────────────────────────────────────────────────────────────
# 4. GENERACIÓN MIDI (solo stdlib + mido)
# ─────────────────────────────────────────────────────────────────────────────

_NOTE_MAP  = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_ACCID_MAP = {"#": 1, "b": -1}


def pitch_to_midi(pitch_str: str) -> int:
    """
    Convierte 'C4', 'D#4', 'Bb3' a número MIDI.
    Si está fuera del rango C3-C5 lo transpone por octavas hasta encajar.
    """
    match = re.fullmatch(r"([A-G])([#b]?)(-?\d+)", pitch_str.strip())
    if not match:
        raise ValueError(f"Pitch inválido: '{pitch_str}'")
    note, acc, octave = match.group(1), match.group(2), int(match.group(3))
    midi = (octave + 1) * 12 + _NOTE_MAP[note] + _ACCID_MAP.get(acc, 0)
    while midi < PITCH_MIN:
        midi += 12
    while midi > PITCH_MAX:
        midi -= 12
    return midi


def _encode_varint(value: int) -> bytes:
    """Codifica entero como MIDI variable-length quantity."""
    buf = [value & 0x7F]
    value >>= 7
    while value:
        buf.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(buf))


def _meta_tempo(bpm: int) -> bytes:
    """Mensaje meta FF 51 03 <3 bytes μs/beat>."""
    us = int(60_000_000 / bpm)
    return b"\xFF\x51\x03" + struct.pack(">I", us)[1:]


def build_midi(music_json: dict) -> bytes:
    """
    Construye un archivo MIDI tipo 0 desde el dict musical.
    ticks_per_beat = 480, canal 0, program 0 (piano).
    """
    tpb  = 480
    bpm  = max(40, min(200, int(music_json.get("tempo_bpm", 90))))
    notes = sorted(music_json["melody"], key=lambda n: float(n["start_beat"]))

    # Eventos: (tick_absoluto, prioridad, status, p1, p2)
    # prioridad 0 = note_off primero cuando coinciden ticks
    events: list[tuple[int, int, int, int, int]] = []
    for note in notes:
        try:
            midi_note = pitch_to_midi(str(note["pitch"]))
        except ValueError:
            continue
        vel   = max(0, min(127, int(note.get("velocity", 80))))
        start = int(float(note["start_beat"])      * tpb)
        dur   = max(1, int(float(note["duration_beats"]) * tpb))
        events.append((start,       1, 0x90, midi_note, vel))   # note_on
        events.append((start + dur, 0, 0x80, midi_note, 0))     # note_off

    events.sort(key=lambda e: (e[0], e[1]))

    track = bytearray()
    # set_tempo
    track += _encode_varint(0) + _meta_tempo(bpm)
    # program_change → piano
    track += _encode_varint(0) + bytes([0xC0, 0x00])

    current_tick = 0
    for tick, _, status, p1, p2 in events:
        delta = tick - current_tick
        current_tick = tick
        track += _encode_varint(delta) + bytes([status, p1, p2])

    # End of track
    track += b"\x00\xFF\x2F\x00"

    track_bytes = bytes(track)
    header  = b"MThd" + struct.pack(">IHHH", 6, 0, 1, tpb)
    chunk   = b"MTrk" + struct.pack(">I", len(track_bytes)) + track_bytes
    return header + chunk


# ─────────────────────────────────────────────────────────────────────────────
# 5. INTERFAZ STREAMLIT
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="MusicAbility", page_icon="🎵", layout="centered")

st.title("🎵 MusicAbility")
st.caption("Accesibilidad musical — describe tu idea y descarga un MIDI.")

# Verificar variables de entorno (nunca mostrar los valores)
_missing_env = [v for v in ("FOUNDRY_API_KEY", "FOUNDRY_ENDPOINT", "MODEL_DEPLOYMENT_NAME")
                if not os.getenv(v)]
if _missing_env:
    st.error(
        f"❌ Faltan variables en `.env`: `{'`, `'.join(_missing_env)}`\n\n"
        "Copia `.env.example` a `.env` y completa los valores."
    )
    st.stop()

# ── Entrada ───────────────────────────────────────────────────────────────────
st.subheader("💬 Describe tu melodía")
user_input = st.text_area(
    label="Instrucción musical:",
    placeholder=(
        "Ej: Una melodía tranquila en Do mayor, tempo lento, 8 compases, "
        "que suene esperanzadora y fácil de tararear."
    ),
    height=120,
)

generate = st.button("🎼 Generar melodía", type="primary", use_container_width=True)

# ── Procesamiento ─────────────────────────────────────────────────────────────
if generate:
    if not user_input.strip():
        st.warning("⚠️ Por favor escribe una instrucción antes de generar.")
        st.stop()

    with st.spinner("🧠 Analizando con Azure AI Foundry…"):
        try:
            music_data = call_foundry_for_music_json(user_input.strip())
        except ValueError as e:
            st.error(f"❌ JSON inválido del modelo:\n\n{e}")
            st.stop()
        except RuntimeError as e:
            st.error(f"❌ Error de conexión con Foundry:\n\n{e}")
            st.stop()

    st.success(f"✅ Melodía generada: **{music_data.get('title', 'Sin título')}**")

    # Métricas resumen
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tonalidad",  music_data.get("key", "–"))
    col2.metric("Tempo",      f"{music_data.get('tempo_bpm', '–')} BPM")
    col3.metric("Compases",   music_data.get("length_bars", "–"))
    col4.metric("Compás",     music_data.get("time_signature", "–"))

    # Supuestos del compositor
    if music_data.get("assumptions"):
        with st.expander("💡 Supuestos del compositor"):
            for a in music_data["assumptions"]:
                st.write(f"• {a}")

    # JSON completo
    with st.expander("📄 Ver JSON musical completo"):
        st.json(music_data)

    # Tabla de notas
    st.subheader("🎼 Partitura (notas)")
    st.dataframe(
        [
            {
                "Nota":     n.get("pitch", "?"),
                "Inicio":   n.get("start_beat", 0),
                "Duración": n.get("duration_beats", 1),
                "Velocidad":n.get("velocity", 80),
            }
            for n in music_data["melody"]
        ],
        use_container_width=True,
        hide_index=True,
    )

    # Generar MIDI
    with st.spinner("🎹 Construyendo archivo MIDI…"):
        try:
            midi_bytes = build_midi(music_data)
        except Exception as e:
            st.error(f"❌ Error al generar MIDI: {e}")
            st.stop()

    st.subheader("⬇️ Descarga tu MIDI")
    st.download_button(
        label="💾 Descargar musicability.mid",
        data=midi_bytes,
        file_name="musicability.mid",
        mime="audio/midi",
        use_container_width=True,
    )