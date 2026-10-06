"""Crop Recommendation Chatbot (Streamlit).

Run (from the folder that contains crop_model.py):
    pip install streamlit
    streamlit run app.py

Type "hii" to wake the bot. It then collects N, P, K, temperature, humidity,
pH and rainfall (all at once in plain English, or one by one) and replies with
the top crops plus the soil analysis, using your trained PyTorch model.
"""
import json
import re
from pathlib import Path

import streamlit as st

import crop_model as cm

st.set_page_config(page_title="CropBot", page_icon="🌾", layout="centered")

# ----------------------------------------------------------------------------
# Field metadata + NLP (regex slot filling)
# ----------------------------------------------------------------------------
FIELD_INFO = {
    "N": ("Nitrogen (N)", "ratio of nitrogen in soil, e.g. 90", (0, 400)),
    "P": ("Phosphorus (P)", "ratio of phosphorus in soil, e.g. 42", (0, 400)),
    "K": ("Potassium (K)", "ratio of potassium in soil, e.g. 43", (0, 400)),
    "temperature": ("Temperature", "in °C, e.g. 21", (-10, 60)),
    "humidity": ("Humidity", "relative humidity in %, e.g. 82", (0, 100)),
    "ph": ("Soil pH", "between 0 and 14, e.g. 6.5", (0, 14)),
    "rainfall": ("Rainfall", "in mm, e.g. 203", (0, 5000)),
}
ORDER = [f for f in cm.FEATURES]  # keep the model's own feature order

NUM = r"(-?\d+(?:\.\d+)?)"
ALIASES = {
    "N": r"nitrogen|n",
    "P": r"phosphor(?:ous|us)|phosphate|p",
    "K": r"potassium|potash|k",
    "temperature": r"temperature|temp",
    "humidity": r"humidity|humid",
    "ph": r"ph",
    "rainfall": r"rainfall|rain",
}
KEYWORD_RE = {
    f: re.compile(
        rf"(?<![a-z])(?:{a})(?![a-z])\s*(?:level|value|content|ratio)?\s*"
        rf"(?:is|are|was|=|:|of|at|about|around|-)?\s*{NUM}",
        re.I,
    )
    for f, a in ALIASES.items()
}
UNIT_RE = {
    "temperature": re.compile(rf"{NUM}\s*(?:°\s*c?|degrees?|deg|celsius)", re.I),
    "humidity": re.compile(rf"{NUM}\s*%", re.I),
    "rainfall": re.compile(rf"{NUM}\s*(?:mm|millimet\w+)", re.I),
}
NPK_RE = re.compile(rf"npk\D*{NUM}\D+{NUM}\D+{NUM}", re.I)
GREET_RE = re.compile(r"^\s*(?:h+i+|hel+o+|hey+|hi+i*|namaste|start)\b", re.I)
TOPK_RE = re.compile(r"top\s*(\d)", re.I)
BYE_WORDS = {"bye", "exit", "quit", "stop", "goodbye", "close"}
RESET_WORDS = {"reset", "clear", "restart", "cancel"}
AGAIN_WORDS = {"again", "new", "another", "next", "yes", "y", "more", "start"}
HELP_WORDS = {"help", "?", "how", "what can you do"}


def extract_values(text: str) -> dict:
    """Pull any recognised soil/climate numbers out of a free-text message."""
    found = {}
    m = NPK_RE.search(text)
    if m:
        found.update({"N": float(m.group(1)), "P": float(m.group(2)), "K": float(m.group(3))})
    for field, rx in KEYWORD_RE.items():
        if field in found:
            continue
        m = rx.search(text)
        if m:
            found[field] = float(m.group(1))
    for field, rx in UNIT_RE.items():
        if field not in found:
            m = rx.search(text)
            if m:
                found[field] = float(m.group(1))
    return {k: v for k, v in found.items() if k in ORDER}


def validate(values: dict):
    ok, errors = {}, []
    for k, v in values.items():
        lo, hi = FIELD_INFO[k][2]
        if lo <= v <= hi:
            ok[k] = v
        else:
            errors.append(f"**{FIELD_INFO[k][0]}** = {v:g} looks out of range ({lo} to {hi}), so I ignored it.")
    return ok, errors


def fmt(v: float) -> str:
    return f"{v:g}"


# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model…")
def load_model(path: str):
    return cm.load_bundle(path)


@st.cache_data(show_spinner=False)
def load_df(path: str):
    return cm.load_dataframe(path)


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
def init_state():
    ss = st.session_state
    ss.setdefault("messages", [])
    ss.setdefault("active", False)
    ss.setdefault("slots", {})
    ss.setdefault("pending", None)
    ss.setdefault("top_k", 3)


init_state()
ss = st.session_state

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title("🌾 CropBot")
    model_path = st.text_input("Model file", "artifacts/crop_model.pt")
    csv_path = st.text_input("Dataset CSV", "data/Crop_recommendation.csv")
    ss.top_k = st.slider("Recommendations to show", 1, 5, ss.top_k)

    metrics_file = Path(model_path).parent / "metrics.json"
    if metrics_file.exists():
        try:
            met = json.loads(metrics_file.read_text())
            st.caption(f"Model test accuracy: **{met['test_accuracy'] * 100:.1f}%** "
                       f"| macro-F1 **{met['test_macro_f1']:.3f}**")
        except Exception:
            pass

    st.markdown("**Current sample**")
    if ss.active:
        for f in ORDER:
            v = ss.slots.get(f)
            st.write(f"{'✅' if v is not None else '⬜'} {FIELD_INFO[f][0]}: "
                     f"{fmt(v) if v is not None else '—'}")
    else:
        st.caption("Bot is asleep. Type **hii** in the chat.")

    if st.button("🗑️ Clear chat", use_container_width=True):
        for k in ("messages", "active", "slots", "pending"):
            ss.pop(k, None)
        init_state()
        st.rerun()

# ----------------------------------------------------------------------------
# Bot logic
# ----------------------------------------------------------------------------
def ask_for(field: str) -> str:
    label, hint, _ = FIELD_INFO[field]
    return f"What's the **{label}**? _({hint})_"


def missing_fields():
    return [f for f in ORDER if f not in ss.slots]


def run_prediction() -> str:
    sample = {f: ss.slots[f] for f in ORDER}
    try:
        model, bundle = load_model(model_path)
        results = cm.recommend(model, bundle, sample, top_k=ss.top_k)
        df = load_df(csv_path)
        report = cm.soil_report(sample, df, results[0][0])
    except FileNotFoundError as e:
        return (f"⚠️ I couldn't find a file: `{e.filename or e}`. Check the paths in the sidebar "
                "(did you run `python train.py`?).")
    except Exception as e:  # keep the chat alive on any model error
        return f"⚠️ Something went wrong while predicting: `{e}`"

    bars = "\n".join(f"{crop:<12} {'█' * round(p * 20):<20} {p * 100:5.1f}%" for crop, p in results)
    lines = "\n".join(f"- {line}" for line in report)
    inputs = " · ".join(f"{k}={fmt(v)}" for k, v in sample.items())
    msg = (f"🌱 **Top recommendations**\n\n```\n{bars}\n```\n"
           f"**Soil analysis for {results[0][0]}**\n\n{lines}\n\n")
    if results[0][1] < 0.6:
        msg += ("_Heads-up: the model isn't very confident here. Some inputs may be unusual "
                "compared to the training data._\n\n")
    msg += f"<sub>Inputs used: {inputs}</sub>\n\nSay **again** for a new sample, or **bye** to end."
    ss.slots = {}
    ss.pending = None
    return msg


HELP_TEXT = (
    "I recommend crops from soil and climate data. You can give values in plain English, e.g.\n\n"
    "`N 90, P 42, K 43, temperature 21, humidity 82, pH 6.5, rainfall 203`\n\n"
    "or `NPK 90 42 43, 21°C, 82%, pH 6.5, 203 mm`, or just answer my questions one by one.\n\n"
    "Commands: **reset** (clear values), **top 5** (show more crops), **again**, **bye**."
)


def respond(text: str) -> str:
    low = text.strip().lower()

    # --- sleeping: wake on greeting ----------------------------------------
    if not ss.active:
        if GREET_RE.search(text):
            ss.active = True
            ss.slots = {}
            ss.pending = ORDER[0]
            return ("Hii! 👋 I'm **CropBot**. Tell me about your soil and weather and I'll suggest "
                    "the best crops.\n\nYou can send everything in one message "
                    "(`N 90, P 42, K 43, temp 21, humidity 82, pH 6.5, rain 203`), "
                    "or answer one by one. Type **help** anytime.\n\n" + ask_for(ORDER[0]))
        return "😴 I'm asleep. Type **hii** to wake me up."

    # --- commands ------------------------------------------------------------
    if low in BYE_WORDS:
        ss.active, ss.slots, ss.pending = False, {}, None
        return "Bye! 🌾 Type **hii** whenever you want me again."
    if low in RESET_WORDS:
        ss.slots, ss.pending = {}, ORDER[0]
        return "Cleared. Let's start over.\n\n" + ask_for(ORDER[0])
    if low in HELP_WORDS:
        return HELP_TEXT
    if GREET_RE.search(text) and not extract_values(text) and low not in AGAIN_WORDS:
        ss.slots, ss.pending = {}, ORDER[0]
        return "Hii again! 👋 Let's begin a new sample.\n\n" + ask_for(ORDER[0])
    if low in AGAIN_WORDS and not ss.slots:
        ss.pending = ORDER[0]
        return "Sure! 🌾 New sample.\n\n" + ask_for(ORDER[0])

    m = TOPK_RE.search(text)
    if m:
        ss.top_k = max(1, min(5, int(m.group(1))))

    # --- slot filling ----------------------------------------------------------
    found = extract_values(text)
    if ss.pending and ss.pending not in found:
        nums = re.findall(NUM, text)
        if len(nums) == 1 and not found:  # bare number answers the pending question
            found[ss.pending] = float(nums[0])
    found, errors = validate(found)

    if not found and not errors:
        if m and not ss.slots:
            return f"Okay, I'll show the top {ss.top_k} crops. Now send me your values."
        if ss.pending:
            return ("Hmm, I couldn't find a number in that. 🤔\n\n" + ask_for(ss.pending))
        return ("I couldn't spot any soil values in that. Try something like "
                "`N 90, P 42, K 43, temp 21, humidity 82, pH 6.5, rain 203`, or say **again** "
                "and I'll ask step by step.")

    ss.slots.update(found)
    parts = []
    if found:
        parts.append("Got it: " + ", ".join(f"**{k}** = {fmt(v)}" for k, v in found.items()) + ".")
    parts.extend(errors)

    todo = missing_fields()
    if todo:
        ss.pending = todo[0]
        parts.append(ask_for(todo[0]) if len(todo) == 1 else
                     f"{len(todo)} left: {', '.join(todo)}.\n\n" + ask_for(todo[0]))
        return "\n\n".join(parts)

    parts.append("That's everything, analysing… 🔍")
    return "\n\n".join(parts) + "\n\n" + run_prediction()


# ----------------------------------------------------------------------------
# Chat UI
# ----------------------------------------------------------------------------
st.title("🌾 CropBot")
st.caption("Soil & climate based crop recommendation chatbot")

if not ss.messages:
    ss.messages.append({"role": "assistant",
                        "content": "😴 I'm asleep. Type **hii** to wake me up."})

for msg in ss.messages:
    with st.chat_message(msg["role"], avatar="🌾" if msg["role"] == "assistant" else None):
        st.markdown(msg["content"], unsafe_allow_html=True)

if prompt := st.chat_input("Type 'hii' to start…"):
    ss.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    reply = respond(prompt)
    ss.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant", avatar="🌾"):
        st.markdown(reply, unsafe_allow_html=True)
    st.rerun()  # refresh the sidebar's "current sample" checklist
