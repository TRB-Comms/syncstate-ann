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
# Configuration
# -----------------------------
STATES = ["Balanced", "Elevated", "Overloaded", "Disconnected"]
DATA_PATH = "data/syncstate_ann_dataset.xlsx"

# If your dataset uses different column names, edit these here:
RAW_COLS = {
    "avg_bpm": "avg_bpm",
    "session_minutes": "session_minutes",
    "sleep_hours": "sleep_hours",
    "practice_load": "practice_load",
    "mood": "mood",
    "stress": "stress",
    "support": "support",
    "wellbeing_state": "wellbeing_state",
}

PROMPTS = {
    "Balanced": [
        "What’s working that you want to protect today?",
        "What’s one small choice that keeps you steady?"
    ],
    "Elevated": [
        "Where can you aim this energy gently—without overcommitting?",
        "What’s the smallest ‘next’ that still feels exciting?"
    ],
    "Overloaded": [
        "What’s the one thing you can release or postpone today?",
        "What would ‘enough’ look like for the next 2 hours?"
    ],
    "Disconnected": [
        "Do you feel more shut down or more scattered right now?",
        "What’s one grounding cue you can try for 60 seconds?"
    ],
    "Unsure": [
        "I might be off. Which feels closer: steady, elevated, overloaded, or disconnected?",
        "Quick check: is your body tense or more numb/flat?"
    ]
}

def pick_mode(conf: float) -> str:
    if conf < 0.55:
        return "unsure"
    if conf < 0.70:
        return "leaning"
    return "suggest"

# -----------------------------
# Feature mapping (raw → 1–5)
# -----------------------------
def bpm_to_energy(bpm):
    if bpm < 70: return 1
    if bpm < 90: return 2
    if bpm < 110: return 3
    if bpm < 130: return 4
    return 5

def minutes_to_focus(m):
    if m < 10: return 1
    if m < 21: return 2
    if m < 36: return 3
    if m < 51: return 4
    return 5

def hours_to_sleep(h):
    if h < 4: return 1
    if h < 5: return 2
    if h < 6: return 3
    if h < 7: return 4
    return 5

@st.cache_data
def load_data():
    return pd.read_excel(DATA_PATH)

@st.cache_resource
def train_model(df: pd.DataFrame):
    # Rename columns if needed (RAW_COLS mapping)
    df = df.rename(columns={v: k for k, v in RAW_COLS.items()})

    # Basic cleaning
    need = ["avg_bpm","session_minutes","sleep_hours","practice_load","mood","stress","support","wellbeing_state"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    df = df.dropna(subset=need).copy()

    # Derive normalized ANN features
    df["energy_bpm"] = df["avg_bpm"].apply(bpm_to_energy)
    df["energy"] = ((df["energy_bpm"] + df["mood"]) / 2).round().clip(1,5)

    df["focus"] = df["session_minutes"].apply(minutes_to_focus)
    df["sleep"] = df["sleep_hours"].apply(hours_to_sleep)
    df["tension"] = df["practice_load"].clip(1,5)
    df["stress_n"] = df["stress"].clip(1,5)

    df["state"] = df["wellbeing_state"].astype(str).str.strip()
    df = df[df["state"].isin(STATES)]

    FEATURES = ["energy", "stress_n", "focus", "tension", "sleep"]
    X = df[FEATURES].astype(float).values
    y = df["state"].map({s:i for i, s in enumerate(STATES)}).values

    if len(df) < 30:
        raise ValueError("Not enough rows after cleaning. Need at least ~30+ rows.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    base = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            activation="relu",
            max_iter=300,
            random_state=42
        ))
    ])

    model = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=STATES, output_dict=False)
    cm = confusion_matrix(y_test, y_pred)

    return model, report, cm, df

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="SYNCstate ANN", layout="centered")
st.title("SYNCstate ANN — Humble Reflection Demo")
st.caption("ANN + calibrated uncertainty to support reflection without replacing human judgment.")

with st.expander("What this is (and isn’t)"):
    st.markdown("""
- **Not a diagnosis. Not medical advice.**
- Offers **reflection prompts**, not instructions.
- Uses **uncertainty** to avoid pretending it knows.
- If someone is unsafe, the correct response is **human help**, not AI output.
""")

# Load dataset
df = load_data()
st.success(f"Loaded dataset: {DATA_PATH}")
st.dataframe(df.head(10), use_container_width=True)

# Train model
model, report, cm, df_clean = train_model(df)

with st.expander("Training summary"):
    st.write("Rows used:", len(df_clean))
    st.write("State distribution:")
    st.write(df_clean["state"].value_counts())
    st.text(report)
    st.write("Confusion matrix:")
    st.write(pd.DataFrame(cm, index=STATES, columns=STATES))

st.divider()
st.subheader("Try a new check-in (demo inputs)")

c1, c2, c3, c4, c5 = st.columns(5)
energy = c1.slider("Energy", 1, 5, 3)
stress = c2.slider("Stress", 1, 5, 3)
focus = c3.slider("Focus", 1, 5, 3)
tension = c4.slider("Tension", 1, 5, 3)
sleep = c5.slider("Sleep", 1, 5, 3)

unsafe_flag = st.checkbox("I’m not safe / I need urgent help right now")

if st.button("Run SYNCstate"):
    if unsafe_flag:
        st.error("You indicated you’re not safe. This demo can’t help with emergencies.")
        st.markdown("If you're in immediate danger, call your local emergency number.")
        st.markdown("In the U.S., you can call or text **988** (Suicide & Crisis Lifeline).")
        st.stop()

    X_new = np.array([[energy, stress, focus, tension, sleep]], dtype=float)

    # Note: model expects FEATURES order: energy, stress_n, focus, tension, sleep
    # stress slider maps to stress_n
    X_new_model = np.array([[energy, stress, focus, tension, sleep]], dtype=float)

    proba = model.predict_proba(X_new_model)[0]
    pred_idx = int(np.argmax(proba))
    pred_state = STATES[pred_idx]
    conf = float(np.max(proba))

    dist = pd.DataFrame({"state": STATES, "probability": proba}).sort_values("probability", ascending=False)

    st.markdown("### Result")
    st.write(f"**Suggested state:** {pred_state}")
    st.write(f"**Confidence (calibrated):** {conf:.2f}")
    st.bar_chart(dist.set_index("state"))

    st.markdown("### Humility-aware response")
    mode = pick_mode(conf)

    if mode == "unsure":
        st.info("I’m not confident enough to label this. Let’s reflect with a quick check instead.")
        st.write("**Prompt:** " + np.random.choice(PROMPTS["Unsure"]))
        choice = st.radio("Which feels closest right now?", STATES, index=0)
        st.write("**Next prompt:** " + np.random.choice(PROMPTS[choice]))

    elif mode == "leaning":
        st.warning(
        "Leaning toward a state, but not certain. I’ll offer choices rather than a single answer."
    )
    top2 = dist.head(2)["state"].tolist()
    st.write(f"Top possibilities: **{top2[0]}** or **{top2[1]}**")

    choice = st.radio(
        "Which feels closer?",
        top2,
        index=0,
        key=f"choice_{pred_state}_{round(conf,2)}"
    )

        st.write("You selected:", choice)
        st.write("**Prompt:** " + np.random.choice(PROMPTS[choice]))

    else:
        st.success("Confident enough to suggest a reflection prompt (still not a judgment).")
        st.write("**Prompt:** " + np.random.choice(PROMPTS[pred_state]))
