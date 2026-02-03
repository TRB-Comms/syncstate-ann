import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix


# =========================================================
# DEMO MODE (public-safe)
# =========================================================
PUBLIC_DEMO = True  # True = hide dataset preview + training details


# -----------------------------
# Session state init (SAFE)
# -----------------------------
if "sync_result" not in st.session_state:
    st.session_state["sync_result"] = None

if "last_inputs" not in st.session_state:
    st.session_state["last_inputs"] = None

if "demo_override" not in st.session_state:
    st.session_state["demo_override"] = "auto"

# Next-step log + feedback memory (session-only)
if "step_log" not in st.session_state:
    st.session_state["step_log"] = []

# feedback stats stored as:
# { "State|Lane|Action": {"up": int, "down": int} }
if "step_feedback" not in st.session_state:
    st.session_state["step_feedback"] = {}


# -----------------------------
# Configuration
# -----------------------------
STATES = ["Balanced", "Elevated", "Overloaded", "Disconnected"]
DATA_PATH = "data/syncstate_ann_dataset.xlsx"

RAW_REQUIRED = [
    "avg_bpm",
    "session_minutes",
    "sleep_hours",
    "practice_load",
    "mood",
    "stress",
    "support",
    "wellbeing_state",
]

PROMPTS = {
    "Balanced": [
        "What’s working that you want to protect today?",
        "What’s one small choice that keeps you steady?",
    ],
    "Elevated": [
        "Where can you aim this energy gently—without overcommitting?",
        "What’s the smallest ‘next’ that still feels exciting?",
    ],
    "Overloaded": [
        "What’s the one thing you can release or postpone today?",
        "What would ‘enough’ look like for the next 2 hours?",
    ],
    "Disconnected": [
        "Do you feel more shut down or more scattered right now?",
        "What’s one grounding cue you can try for 60 seconds?",
    ],
    "Unsure": [
        "I might be off. Which feels closer: steady, elevated, overloaded, or disconnected?",
        "Quick check: does your body feel tense, or more numb/flat?",
    ],
}

NEXT_STEPS = {
    "Balanced": {
        "Protect": [
            "Name one thing you want to protect today (10 seconds).",
            "Choose one boundary: stop time / break time / screen-off time.",
        ],
        "Grow": [
            "Pick one small ‘next’ that feels light (2 minutes).",
            "Write one sentence: ‘Today counts even if…’",
        ],
    },
    "Elevated": {
        "Channel": [
            "Pick ONE priority for the next 30 minutes (not three).",
            "Set a 10-minute timer and start the smallest step.",
        ],
        "Soften": [
            "Do 3 slow exhales, then decide what can wait.",
            "Drink water and stretch your shoulders for 30 seconds.",
        ],
    },
    "Overloaded": {
        "Reduce": [
            "Write your top 3 tasks and postpone ONE.",
            "Choose ‘minimum viable’ for the next hour.",
        ],
        "Recover": [
            "60 seconds: unclench jaw, drop shoulders, slow breathing.",
            "Stand and walk for 90 seconds.",
        ],
    },
    "Disconnected": {
        "Reconnect": [
            "5-4-3-2-1 grounding: name 5 things you see…",
            "Touch something textured and describe it for 20 seconds.",
        ],
        "Support": [
            "Message one safe person: ‘Can I borrow 2 minutes?’",
            "Write one sentence about what you feel (no fixing).",
        ],
    },
    "Unsure": {
        "Clarify": [
            "Pick the closest state first, then choose one micro-step.",
            "What’s loudest right now: stress, tiredness, pressure, or numbness?",
        ],
    },
}


def pick_mode(conf: float) -> str:
    # Auto mode thresholds (real model output)
    if conf < 0.55:
        return "unsure"
    if conf < 0.70:
        return "leaning"
    return "suggest"


# -----------------------------
# Feature mapping (raw → 1–5)
# -----------------------------
def bpm_to_energy(bpm: float) -> int:
    if bpm < 70:
        return 1
    if bpm < 90:
        return 2
    if bpm < 110:
        return 3
    if bpm < 130:
        return 4
    return 5


def minutes_to_focus(m: float) -> int:
    if m < 10:
        return 1
    if m < 21:
        return 2
    if m < 36:
        return 3
    if m < 51:
        return 4
    return 5


def hours_to_sleep(h: float) -> int:
    if h < 4:
        return 1
    if h < 5:
        return 2
    if h < 6:
        return 3
    if h < 7:
        return 4
    return 5


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_excel(path)


@st.cache_resource
def train_model(df_raw: pd.DataFrame):
    missing = [c for c in RAW_REQUIRED if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    df = df_raw.dropna(subset=RAW_REQUIRED).copy()

    for col in ["avg_bpm", "session_minutes", "sleep_hours", "practice_load", "mood", "stress", "support"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=RAW_REQUIRED).copy()

    # Derived features (interpretable, 1–5)
    df["energy_bpm"] = df["avg_bpm"].apply(bpm_to_energy)
    df["energy"] = ((df["energy_bpm"] + df["mood"]) / 2).round().clip(1, 5).astype(int)

    df["focus"] = df["session_minutes"].apply(minutes_to_focus).astype(int)
    df["sleep"] = df["sleep_hours"].apply(hours_to_sleep).astype(int)
    df["tension"] = df["practice_load"].round().clip(1, 5).astype(int)
    df["stress_n"] = df["stress"].round().clip(1, 5).astype(int)

    df["state"] = df["wellbeing_state"].astype(str).str.strip()
    df = df[df["state"].isin(STATES)].copy()

    if len(df) < 40:
        raise ValueError(f"Not enough usable rows after cleaning ({len(df)}). Add more data.")

    FEATURES = ["energy", "stress_n", "focus", "tension", "sleep"]
    X = df[FEATURES].astype(float).values
    y = df["state"].map({s: i for i, s in enumerate(STATES)}).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    base = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(64, 32, 16),
                activation="relu",
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=400,
                random_state=42,
            )),
        ]
    )

    model = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=STATES, output_dict=False)
    cm = confusion_matrix(y_test, y_pred)

    return model, report, cm, df


# -----------------------------
# Next-step personalization helpers
# -----------------------------
def _fb_key(state: str, lane: str, action: str) -> str:
    return f"{state}|{lane}|{action}"


def record_feedback(state: str, lane: str, action: str, is_helpful: bool):
    key = _fb_key(state, lane, action)
    stats = st.session_state["step_feedback"].get(key, {"up": 0, "down": 0})
    if is_helpful:
        stats["up"] += 1
    else:
        stats["down"] += 1
    st.session_state["step_feedback"][key] = stats


def score_action(state: str, lane: str, action: str) -> int:
    stats = st.session_state["step_feedback"].get(_fb_key(state, lane, action), {"up": 0, "down": 0})
    return int(stats["up"]) - int(stats["down"])


def render_next_step(chosen_state: str, mode: str):
    st.markdown("### If you want, try one small next step")
    st.caption("Think of this as an experiment — you can adjust or ignore anything here.")

    library = NEXT_STEPS.get(chosen_state, NEXT_STEPS["Unsure"])
    lanes = list(library.keys())

    lane = st.selectbox("Pick a lane", lanes, index=0, key=f"lane_{mode}_{chosen_state}")
    actions = library[lane]

    actions_sorted = sorted(actions, key=lambda a: score_action(chosen_state, lane, a), reverse=True)
    action = st.radio("Pick one micro-action", actions_sorted, index=0, key=f"action_{mode}_{chosen_state}")

    c1, c2, c3 = st.columns(3)
    minutes = c1.selectbox("Time-box", [1, 2, 5, 10, 15], index=1, key=f"mins_{mode}_{chosen_state}")
    start = c2.button("Start now", key=f"start_{mode}_{chosen_state}")
    log = c3.button("Log this", key=f"log_{mode}_{chosen_state}")

    if start:
        st.info(f"Try this for {minutes} minute(s): **{action}**")

    if log:
        st.session_state["step_log"].append(
            {"state": chosen_state, "lane": lane, "action": action, "minutes": minutes}
        )
        st.success("Logged.")

    st.markdown("#### Was this helpful?")
    st.caption("This only affects suggestions during this session.")

    f1, f2, f3 = st.columns([1, 1, 2])
    if f1.button("👍 Helped", key=f"fb_up_{mode}_{chosen_state}_{lane}"):
        record_feedback(chosen_state, lane, action, is_helpful=True)
        st.success("Noted — I’ll surface options like that more (this session).")
        st.rerun()

    if f2.button("👎 Not for me", key=f"fb_down_{mode}_{chosen_state}_{lane}"):
        record_feedback(chosen_state, lane, action, is_helpful=False)
        st.info("Got it — I’ll de-emphasize that option (this session).")
        st.rerun()

    if st.session_state.get("step_log"):
        with st.expander("Your step log (this session)"):
            log_df = pd.DataFrame(st.session_state["step_log"])
            st.dataframe(log_df, use_container_width=True)
            csv = log_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download log as CSV",
                data=csv,
                file_name="syncstate_next_steps_log.csv",
                mime="text/csv",
            )


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="SYNCstate ANN", layout="centered")
st.title("SYNCstate ANN — Humble Reflection Demo")

if PUBLIC_DEMO:
    st.info(
        "Public demo build: internal datasets and training details are intentionally hidden. "
        "This demo showcases confidence-aware UX, not clinical inference."
    )

with st.expander("What this is (and isn’t)"):
    st.markdown(
        """
- **Not a diagnosis. Not medical advice.**
- Offers **reflection prompts** and **micro-actions**, not instructions.
- Uses **uncertainty** to avoid pretending it knows.
- If someone is unsafe, the correct response is **human help**, not AI output.
"""
    )

df_raw = load_data(DATA_PATH)
st.success("Dataset loaded.")

# Hide dataset preview + training internals in public demo mode
if not PUBLIC_DEMO:
    with st.expander("Dataset preview"):
        st.dataframe(df_raw.head(12), use_container_width=True)
        st.write("Columns:", list(df_raw.columns))

model, report, cm, df_clean = train_model(df_raw)

if not PUBLIC_DEMO:
    with st.expander("Training summary"):
        st.write("Rows used:", len(df_clean))
        st.write("State distribution:")
        st.write(df_clean["state"].value_counts())
        st.text(report)
        st.write("Confusion matrix:")
        st.write(pd.DataFrame(cm, index=STATES, columns=STATES))

st.divider()
st.subheader("How are you doing today?")
st.caption("Let Tee know how things feel right now. Move the slider from 1-5 with 5 being great.")

# Presets (no extra copy)
p1, p2, p3, p4 = st.columns(4)

def apply_preset(vals: dict, override: str):
    st.session_state["energy"] = vals["energy"]
    st.session_state["stress"] = vals["stress"]
    st.session_state["focus"] = vals["focus"]
    st.session_state["tension"] = vals["tension"]
    st.session_state["sleep"] = vals["sleep"]
    st.session_state["demo_override"] = override
    st.rerun()

if p1.button("Preset: Unsure"):
    apply_preset({"energy": 4, "stress": 2, "focus": 1, "tension": 4, "sleep": 5}, override="unsure")

if p2.button("Preset: Leaning"):
    apply_preset({"energy": 3, "stress": 4, "focus": 3, "tension": 3, "sleep": 4}, override="leaning")

if p3.button("Preset: Suggest"):
    apply_preset({"energy": 4, "stress": 5, "focus": 2, "tension": 5, "sleep": 1}, override="suggest")

if p4.button("Auto mode"):
    st.session_state["demo_override"] = "auto"
    st.rerun()

# Sliders
c1, c2, c3, c4, c5 = st.columns(5)
energy = c1.slider("Energy", 1, 5, 3, key="energy")
stress = c2.slider("Stress", 1, 5, 3, key="stress")
focus = c3.slider("Focus", 1, 5, 3, key="focus")
tension = c4.slider("Tension", 1, 5, 3, key="tension")
sleep = c5.slider("Sleep", 1, 5, 3, key="sleep")

unsafe_flag = st.checkbox("I’m not safe / I need urgent help right now")
run = st.button("Run SYNCstate")

if run:
    if unsafe_flag:
        st.error("You indicated you’re not safe. This demo can’t help with emergencies.")
        st.markdown("If you're in immediate danger, call your local emergency number.")
        st.markdown("In the U.S., you can call or text **988** (Suicide & Crisis Lifeline).")
        st.stop()

    X_new = np.array([[energy, stress, focus, tension, sleep]], dtype=float)
    proba = model.predict_proba(X_new)[0]
    pred_idx = int(np.argmax(proba))
    pred_state = STATES[pred_idx]
    conf = float(np.max(proba))

    dist = pd.DataFrame({"state": STATES, "probability": proba}).sort_values("probability", ascending=False)

    st.session_state["sync_result"] = {"pred_state": pred_state, "conf": conf, "dist": dist}
    st.session_state["last_inputs"] = {"energy": energy, "stress": stress, "focus": focus, "tension": tension, "sleep": sleep}
    st.rerun()

# Render results
result = st.session_state.get("sync_result")
if result is not None:
    pred_state = result["pred_state"]
    conf = result["conf"]
    dist = result["dist"]

    mode = pick_mode(conf)
    override = st.session_state.get("demo_override", "auto")
    if override != "auto":
        mode = override
        st.caption("Demo override is ON (interaction style forced).")

    st.markdown("### Result")
    st.write("Inputs used:", st.session_state.get("last_inputs"))
    st.write(f"**What the model leans toward (not a label):** {pred_state}")
    st.write(f"**Confidence (calibrated):** {conf:.2f}")

    if mode == "unsure":
        st.caption("This signal is shown for transparency, not as a final assessment.")

    chart_df = dist.set_index("state")[["probability"]]
    st.bar_chart(chart_df)
    st.caption("This shows likelihoods, not labels. You decide what fits.")

    st.markdown("### Humility-aware response")

    if mode == "unsure":
        st.info("I might be off here. Rather than label your state, I’ll pause and ask for your perspective.")
        st.write("**Prompt:** " + np.random.choice(PROMPTS["Unsure"]))
        choice = st.radio("Which feels closest right now?", STATES, index=0, key="unsure_choice")
        st.write("**Next prompt:** " + np.random.choice(PROMPTS[choice]))
        chosen_state = choice
        render_next_step(chosen_state, mode)

    elif mode == "leaning":
        st.warning("Leaning toward a state, but not certain. I’ll offer choices rather than a single answer.")
        top2 = dist.head(2)["state"].tolist()
        st.write(f"Top possibilities: **{top2[0]}** or **{top2[1]}**")
        choice = st.radio("Which feels closer?", top2, index=0, key="leaning_choice")
        st.write("You selected:", choice)
        st.write("**Prompt:** " + np.random.choice(PROMPTS[choice]))
        chosen_state = choice
        render_next_step(chosen_state, mode)

    else:
        st.success("Confident enough to suggest a reflection prompt (still not a judgment).")
        st.write("**Prompt:** " + np.random.choice(PROMPTS[pred_state]))
        chosen_state = pred_state
        render_next_step(chosen_state, mode)

    st.divider()
    if st.button("Start a new check-in"):
        st.session_state["sync_result"] = None
        st.session_state["last_inputs"] = None
        st.rerun()
