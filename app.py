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

# -----------------------------
# Music-aware prompt library (copy-polished)
# -----------------------------
PROMPTS = {
    "Balanced": [
        "What part of your rhythm feels steady right now—and worth protecting?",
        "If today had a tempo, what pace keeps you grounded?",
        "What’s one thing you’re doing that helps you stay in time with yourself?",
        "What would help you keep this groove without overextending?",
    ],
    "Elevated": [
        "Your tempo feels higher—where can you aim that energy without speeding into burnout?",
        "What’s the smallest ‘next note’ you can play that still moves things forward?",
        "If you slowed down by 10%, what would you keep—and what would you drop?",
        "What helps you stay expressive without adding more tracks than you can mix?",
    ],
    "Overloaded": [
        "It may be time to simplify the arrangement—what can come out of the mix for now?",
        "For the next couple of hours, what would ‘good enough’ sound like—not perfect?",
        "If you reduced the tempo slightly, what would feel more manageable?",
        "What’s one boundary that gives you breathing room today?",
    ],
    "Disconnected": [
        "Do you feel muted (numb/flat) or scattered (out of sync)?",
        "What’s one small cue that helps you feel more present—right now?",
        "If your body could pick the next note, would it ask for rest, water, movement, or connection?",
        "What’s a gentle way to tune back in—without forcing it?",
    ],
    "Unsure": [
        "The signal is unclear—what feels closest: steady, elevated, overloaded, or disconnected?",
        "If you named the main signal: pressure, tiredness, stress, or numbness—what fits best?",
        "What would help most in this moment: clarity, relief, connection, or rest?",
        "If none of the labels fit, what word does fit right now?",
    ],
}

# -----------------------------
# B.E.A.Tsyncro-aligned next steps (Vibes / Restore / Support)
# -----------------------------
NEXT_STEPS = {
    "Balanced": {
        "Vibes": [
            "Pick one ‘keep it steady’ intention for today (10 seconds).",
            "Choose a tempo for the next hour: slow / medium / focused.",
        ],
        "Restore": [
            "Take 5 deep breaths—then unclench jaw and drop shoulders.",
            "Drink water. Small reset. No story needed.",
        ],
        "Support": [
            "Send a quick message: ‘I’m doing okay—just checking in.’",
            "Name one boundary that protects your groove today.",
        ],
    },
    "Elevated": {
        "Vibes": [
            "Choose ONE priority track for the next 30 minutes—ignore the rest.",
            "Set a 10-minute timer and start the smallest step (one bar at a time).",
        ],
        "Restore": [
            "Do 60 seconds of slow breathing (long exhale).",
            "Stand, stretch shoulders/neck, and relax your hands (30 seconds).",
        ],
        "Support": [
            "Say no to one thing that isn’t essential today.",
            "Ask someone: ‘Can you help me choose what matters most?’",
        ],
    },
    "Overloaded": {
        "Vibes": [
            "Simplify the arrangement: write 3 tasks and postpone ONE.",
            "Define ‘good enough’ for the next hour in one sentence.",
        ],
        "Restore": [
            "90 seconds: walk, shake out arms, drop shoulders, breathe out long.",
            "Eat something small or drink water—fuel counts.",
        ],
        "Support": [
            "Message one safe person: ‘I’m overloaded. Can I borrow 2 minutes?’",
            "Choose a boundary: no new requests for the next 2 hours.",
        ],
    },
    "Disconnected": {
        "Vibes": [
            "Name what’s true: muted, scattered, heavy, foggy, or flat.",
            "Pick one sensory anchor: light / sound / touch / temperature.",
        ],
        "Restore": [
            "5-4-3-2-1 grounding: 5 things you see…",
            "Cold water on hands or face (10 seconds) + slow exhale.",
        ],
        "Support": [
            "Reach out with one sentence: ‘I’m not myself today.’",
            "If talking feels hard: write one sentence about how you feel (no fixing).",
        ],
    },
    "Unsure": {
        "Vibes": [
            "Choose the closest state first—then pick one micro-step from it.",
            "What’s loudest: pressure, tiredness, stress, or numbness?",
        ],
        "Restore": [
            "Do one body check: shoulders, jaw, breath—release one.",
            "Drink water and take one slow minute.",
        ],
        "Support": [
            "Ask: ‘Can someone sit with me for a minute?’ (text counts).",
            "If you’re alone: name one thing that would feel supportive right now.",
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
    st.caption("Think of this like a tiny adjustment in tempo or tone. You’re in charge.")

    library = NEXT_STEPS.get(chosen_state, NEXT_STEPS["Unsure"])
    lanes = list(library.keys())

    lane = st.selectbox("What would help most right now?", lanes, index=0, key=f"lane_{mode}_{chosen_state}")
    actions = library[lane]

    actions_sorted = sorted(actions, key=lambda a: score_action(chosen_state, lane, a), reverse=True)
    action = st.radio("Pick one small step", actions_sorted, index=0, key=f"action_{mode}_{chosen_state}")

    c1, c2, c3 = st.columns(3)
    minutes = c1.selectbox("Time-box", [1, 2, 5, 10, 15], index=1, key=f"mins_{mode}_{chosen_state}")
    start = c2.button("Start", key=f"start_{mode}_{chosen_state}")
    log = c3.button("Save", key=f"log_{mode}_{chosen_state}")

    if start:
        st.info(f"For {minutes} minute(s), try: **{action}**")

    if log:
        st.session_state["step_log"].append(
            {"state": chosen_state, "lane": lane, "action": action, "minutes": minutes}
        )
        st.success("Saved.")

    st.markdown("#### Was this helpful?")
    st.caption("This only changes what shows up first during this session.")

    f1, f2, _ = st.columns([1, 1, 2])
    if f1.button("👍 Yes", key=f"fb_up_{mode}_{chosen_state}_{lane}"):
        record_feedback(chosen_state, lane, action, is_helpful=True)
        st.success("Thanks — I’ll surface options like that first.")
        st.rerun()

    if f2.button("👎 Not really", key=f"fb_down_{mode}_{chosen_state}_{lane}"):
        record_feedback(chosen_state, lane, action, is_helpful=False)
        st.info("Got it — I’ll move that option lower next time.")
        st.rerun()

    if st.session_state.get("step_log"):
        with st.expander("Your saved steps (this session)"):
            log_df = pd.DataFrame(st.session_state["step_log"])
            st.dataframe(log_df, use_container_width=True)
            csv = log_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download as CSV",
                data=csv,
                file_name="syncstate_next_steps_log.csv",
                mime="text/csv",
            )


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="SYNCstate", layout="centered")
st.title("SYNCstate")
st.caption("A gentle check-in to support reflection — not diagnosis or advice.")

if PUBLIC_DEMO:
    st.info(
        "This is a demonstration experience using representative, non-identifiable data. "
        "It shows how uncertainty-aware AI can support reflection — not make judgments."
    )

with st.expander("What this is — and what it isn’t"):
    st.markdown(
        """
**What this is**
- A brief self-check to help you notice patterns
- A pause for reflection when things feel fuzzy
- An example of AI that knows when *not* to be certain

**What this isn’t**
- Not a diagnosis or medical assessment
- Not a replacement for human support or judgment
- Not advice — you always choose what fits

If you’re feeling unsafe or in crisis, the right next step is human help.
"""
    )

df_raw = load_data(DATA_PATH)
st.success("Ready.")

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
st.subheader("How are you feeling today?")
st.caption("Use the sliders to let Tee know how you're feeling, 5 being very goo and 1, not good at all.")

# Presets
p1, p2, p3, p4 = st.columns(4)

def apply_preset(vals: dict, override: str):
    st.session_state["energy"] = vals["energy"]
    st.session_state["stress"] = vals["stress"]
    st.session_state["focus"] = vals["focus"]
    st.session_state["tension"] = vals["tension"]
    st.session_state["sleep"] = vals["sleep"]
    st.session_state["demo_override"] = override
    st.rerun()

if p1.button("Explore uncertainty"):
    apply_preset({"energy": 4, "stress": 2, "focus": 1, "tension": 4, "sleep": 5}, override="unsure")

if p2.button("Explore mixed signals"):
    apply_preset({"energy": 3, "stress": 4, "focus": 3, "tension": 3, "sleep": 4}, override="leaning")

if p3.button("Explore clearer signal"):
    apply_preset({"energy": 4, "stress": 5, "focus": 2, "tension": 5, "sleep": 1}, override="suggest")

if p4.button("Automatic"):
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
checkin = st.button("Check my vibe")

if checkin:
    if unsafe_flag:
        st.error("If you're in immediate danger, please contact emergency help right now.")
        st.markdown("In the U.S., you can call or text **988** (Suicide & Crisis Lifeline).")
        st.stop()

    X_new = np.array([[energy, stress, focus, tension, sleep]], dtype=float)
    proba = model.predict_proba(X_new)[0]
    pred_idx = int(np.argmax(proba))
    pred_state = STATES[pred_idx]
    conf = float(np.max(proba))

    dist = pd.DataFrame({"state": STATES, "probability": proba}).sort_values("probability", ascending=False)

    st.session_state["sync_result"] = {"pred_state": pred_state, "conf": conf, "dist": dist}
    st.session_state["last_inputs"] = {
        "energy": energy,
        "stress": stress,
        "focus": focus,
        "tension": tension,
        "sleep": sleep,
    }
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
        st.caption("Demo mode: showing a specific interaction style.")

    st.markdown("### What this check-in noticed")
    st.write("Inputs used:", st.session_state.get("last_inputs"))
    st.write(f"**Possible state this resembles:** {pred_state}")
    st.write(f"**How confident the system is:** {conf:.2f}")

    if mode == "unsure":
        st.caption("This is shown for transparency — not as a final assessment.")

    chart_df = dist.set_index("state")[["probability"]]
    st.bar_chart(chart_df)
    st.caption("These are likelihoods, not labels. You decide what fits — or if none do.")

    st.markdown("### A reflective response")

    if mode == "unsure":
        st.info("It’s not clear enough to name this confidently—so we’ll pause and check in with you instead.")
        st.caption("You don’t need to answer perfectly—just notice what comes up.")
        st.write("**Question:** " + np.random.choice(PROMPTS["Unsure"]))
        choice = st.radio("Which feels closest right now?", STATES, index=0, key="unsure_choice")
        st.write("**Follow-up question:** " + np.random.choice(PROMPTS[choice]))
        chosen_state = choice
        render_next_step(chosen_state, mode)

    elif mode == "leaning":
        st.warning("The signal is mixed—so instead of guessing, we’ll explore possibilities together.")
        top2 = dist.head(2)["state"].tolist()
        st.write(f"Two possibilities: **{top2[0]}** or **{top2[1]}**")
        choice = st.radio("Which feels closer?", top2, index=0, key="leaning_choice")
        st.caption("You don’t need to answer perfectly—just notice what comes up.")
        st.write("**Question:** " + np.random.choice(PROMPTS[choice]))
        chosen_state = choice
        render_next_step(chosen_state, mode)

    else:
        st.success("The signal is clearer here—so here’s a gentle question to help you tune in.")
        st.caption("You don’t need to answer perfectly—just notice what comes up.")
        st.write("**Question:** " + np.random.choice(PROMPTS[pred_state]))
        chosen_state = pred_state
        render_next_step(chosen_state, mode)

    st.divider()
    if st.button("Start a new check-in"):
        st.session_state["sync_result"] = None
        st.session_state["last_inputs"] = None
        st.rerun()
