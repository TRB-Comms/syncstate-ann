import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix


# -----------------------------
# Session state init
# -----------------------------
if "sync_result" not in st.session_state:
    st.session_state.sync_result = None

if "last_inputs" not in st.session_state:
    st.session_state.last_inputs = None


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
        "Quick check: is your body tense or more numb/flat?",
    ],
}


# -----------------------------
# Humility thresholds
# -----------------------------
def pick_mode(conf: float) -> str:
    # Adjust these if you want a more/less humble model
    if conf < 0.60:
        return "unsure"
    if conf < 0.80:
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
    # Validate columns
    missing = [c for c in RAW_REQUIRED if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    df = df_raw.dropna(subset=RAW_REQUIRED).copy()

    # Cast numeric columns safely
    for col in ["avg_bpm", "session_minutes", "sleep_hours", "practice_load", "mood", "stress", "support"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=RAW_REQUIRED).copy()

    # Derived ANN features (normalized 1–5)
    df["energy_bpm"] = df["avg_bpm"].apply(bpm_to_energy)
    df["energy"] = ((df["energy_bpm"] + df["mood"]) / 2).round().clip(1, 5).astype(int)

    df["focus"] = df["session_minutes"].apply(minutes_to_focus).astype(int)
    df["sleep"] = df["sleep_hours"].apply(hours_to_sleep).astype(int)
    df["tension"] = df["practice_load"].round().clip(1, 5).astype(int)
    df["stress_n"] = df["stress"].round().clip(1, 5).astype(int)

    df["state"] = df["wellbeing_state"].astype(str).str.strip()
    df = df[df["state"].isin(STATES)].copy()

    if len(df) < 40:
        raise ValueError(
            f"Not enough usable rows after cleaning ({len(df)}). "
            "Add more data or reduce filtering."
        )

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

    # Calibrate probabilities so "confidence" is more meaningful
    model = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=STATES, output_dict=False)
    cm = confusion_matrix(y_test, y_pred)

    # Return cleaned df too for inspection
    return model, report, cm, df


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="SYNCstate ANN", layout="centered")
st.title("SYNCstate ANN — Humble Reflection Demo")
st.caption("ANN + calibrated uncertainty to support reflection without replacing human judgment.")

with st.expander("What this is (and isn’t)"):
    st.markdown(
        """
- **Not a diagnosis. Not medical advice.**
- Offers **reflection prompts**, not instructions.
- Uses **uncertainty** to avoid pretending it knows.
- If someone is unsafe, the correct response is **human help**, not AI output.
"""
    )

# Load + train
df_raw = load_data(DATA_PATH)
st.success(f"Loaded dataset: {DATA_PATH}")

with st.expander("Dataset preview"):
    st.dataframe(df_raw.head(15), use_container_width=True)
    st.write("Columns:", list(df_raw.columns))

model, report, cm, df_clean = train_model(df_raw)

with st.expander("Training summary"):
    st.write("Rows used:", len(df_clean))
    st.write("State distribution:")
    st.write(df_clean["state"].value_counts())
    st.text(report)
    st.write("Confusion matrix:")
    st.write(pd.DataFrame(cm, index=STATES, columns=STATES))

st.divider()
st.subheader("Try a new check-in")

c1, c2, c3, c4, c5 = st.columns(5)
energy = c1.slider("Energy", 1, 5, 3)
stress = c2.slider("Stress", 1, 5, 3)
focus = c3.slider("Focus", 1, 5, 3)
tension = c4.slider("Tension", 1, 5, 3)
sleep = c5.slider("Sleep", 1, 5, 3)

unsafe_flag = st.checkbox("I’m not safe / I need urgent help right now")

run = st.button("Run SYNCstate")

# When button is clicked, compute & store result in session_state
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

    dist = pd.DataFrame({"state": STATES, "probability": proba}).sort_values(
        "probability", ascending=False
    )

    st.session_state.sync_result = {
        "pred_state": pred_state,
        "conf": conf,
        "dist": dist,
    }
    st.session_state.last_inputs = {
        "energy": energy,
        "stress": stress,
        "focus": focus,
        "tension": tension,
        "sleep": sleep,
    }

    # Force a rerun so UI below uses stored result reliably
    st.rerun()

# Render results if we have them
result = st.session_state.sync_result
if result is not None:
    pred_state = result["pred_state"]
    conf = result["conf"]
    dist = result["dist"]

    st.markdown("### Result")
    st.write("Inputs used:", st.session_state.last_inputs)
    st.write(f"**Suggested state:** {pred_state}")
    st.write(f"**Confidence (calibrated):** {conf:.2f}")
    
    chart_df = dist.set_index("state")[["probability"]]
    st.bar_chart(chart_df)
    st.caption("Probabilities are shown to emphasize uncertainty rather than certainty.")
    st.markdown("### Humility-aware response")
    mode = pick_mode(conf)

    if mode == "unsure":
        st.info("I’m not confident enough to label this. Let’s reflect with a quick check instead.")
        st.write("**Prompt:** " + np.random.choice(PROMPTS["Unsure"]))
        choice = st.radio(
            "Which feels closest right now?",
            STATES,
            index=0,
            key="unsure_choice",
        )
        st.write("**Next prompt:** " + np.random.choice(PROMPTS[choice]))

    elif mode == "leaning":
        st.warning("Leaning toward a state, but not certain. I’ll offer choices rather than a single answer.")
        top2 = dist.head(2)["state"].tolist()
        st.write(f"Top possibilities: **{top2[0]}** or **{top2[1]}**")

        choice = st.radio(
            "Which feels closer?",
            top2,
            index=0,
            key="leaning_choice",
        )

        st.write("You selected:", choice)
        st.write("**Prompt:** " + np.random.choice(PROMPTS[choice]))

    else:
        st.success("Confident enough to suggest a reflection prompt (still not a judgment).")
        st.write("**Prompt:** " + np.random.choice(PROMPTS[pred_state]))

    st.divider()
    if st.button("Clear result"):
        st.session_state.sync_result = None
        st.session_state.last_inputs = None
        st.rerun()
