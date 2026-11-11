"""
Purchase Intent Predictor - Streamlit app
Phase 8, Part A: Home page + Visitor Prediction page.
(The Analytics Dashboard and Model Information pages are added in Part B.)
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "model.joblib"
DATA_PATH = APP_DIR / "data" / "online_shoppers_clean.csv"

VER_PV, VER_NOPV = "with PageValues", "no PageValues"
SC_A, SC_B, SC_C = "A: cheap action (F2)", "B: balanced (F1)", "C: costly action (F0.5)"
STRATEGIES = {                      # label shown to the user -> scenario stored in model.joblib
    "Broad reach (cheap actions)": SC_A,
    "Balanced (recommended)": SC_B,
    "Selective (costly actions)": SC_C,
}

st.set_page_config(page_title="Purchase Intent Predictor", page_icon="🛒", layout="wide")

# ------------------------------------------------------------------ styling
st.markdown("""
<style>
.block-container {padding-top: 2.2rem; max-width: 1150px;}
h1, h2, h3 {letter-spacing: -0.01em;}
.hero {padding: 1.5rem 1.7rem; border-radius: 14px; border: 1px solid #e2e8f0;
       background: linear-gradient(135deg, #eef2ff 0%, #f8fafc 100%); margin-bottom: 1.3rem;}
.hero h1 {margin: 0 0 .35rem 0; font-size: 2.1rem; color: #0f172a;}
.hero p {margin: 0; color: #475569; font-size: 1.05rem;}
.step-num {display:inline-block; width: 1.8rem; height: 1.8rem; line-height: 1.8rem; text-align:center;
           border-radius: 50%; background:#4f46e5; color:white; font-weight:600; margin-bottom:.4rem;}
.badge {display:inline-block; padding:.45rem 1rem; border-radius: 999px; font-weight:700; font-size:1.15rem;}
.badge.good {background:#dcfce7; color:#166534;}
.badge.neutral {background:#f1f5f9; color:#334155;}
.tier {border-left: 5px solid #4f46e5; background:#f8fafc; padding:.8rem 1rem; border-radius: 8px; margin-bottom:.6rem;}
.tier.t0 {border-color:#16a34a;} .tier.t1 {border-color:#4f46e5;}
.tier.t2 {border-color:#f59e0b;} .tier.t3 {border-color:#94a3b8;}
.tier h4 {margin:0 0 .25rem 0; color:#0f172a;}
.scorewrap {position:relative; padding: 1.1rem 0 3rem 0;}
.scoretrack {position:relative; height:14px; background:#e2e8f0; border-radius:999px;}
.scorefill {position:absolute; left:0; top:0; height:14px; border-radius:999px;
            background: linear-gradient(90deg, #c7d2fe, #4f46e5);}
.scoredot {position:absolute; top:-5px; width:24px; height:24px; margin-left:-12px; border-radius:50%;
           background:#4f46e5; border:3px solid white; box-shadow:0 1px 4px rgba(0,0,0,.35);}
.tick {position:absolute; top:-4px; width:2px; height:22px; background:#64748b; margin-left:-1px;}
.ticklabel {position:absolute; top:26px; transform:translateX(-50%); font-size:.75rem;
            color:#475569; text-align:center; line-height:1.15; white-space:nowrap;}
.small-note {color:#64748b; font-size:.9rem;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ loading
@st.cache_resource(show_spinner="Loading the model...")
def _load_bundle():
    return joblib.load(MODEL_PATH)


def get_bundle():
    if not MODEL_PATH.exists():
        st.error("model.joblib was not found next to app.py. Put the file from Phase 7 in the same folder.")
        st.stop()
    try:
        return _load_bundle()
    except Exception as exc:                      # most often a scikit-learn version mismatch
        st.error("The model file could not be loaded. A different scikit-learn version than the one "
                 "used to save it (1.6.1) is the most common cause. Check requirements.txt.")
        st.code(str(exc))
        st.stop()


def test_row(b, version, scenario):
    """Test-set results (computed in Phase 5) for one model version and strategy."""
    for r in b["test_metrics"]:
        if r["features"] == version and r["scenario"] == scenario:
            return r
    return None


# ------------------------------------------------------------------ scoring
def score_visitor(b, visitor, page_value):
    """Return (model version used, purchase likelihood score) for one visitor."""
    use_pv = page_value is not None
    version = VER_PV if use_pv else VER_NOPV
    row = {c: visitor.get(c, np.nan) for c in b["feature_columns"]}
    row["PageValues"] = float(page_value) if use_pv else 0.0   # placeholder: required column, unused by no-PageValues model
    X_new = pd.DataFrame([row], columns=b["feature_columns"])
    score = float(b["models"][version].predict_proba(X_new)[0, 1])
    return version, score, X_new


TIERS = [
    ("t0", "Very high intent",
     "These visitors mostly buy on their own. Keep the path to checkout simple. A discount here is "
     "mostly lost margin, so don't offer one by default."),
    ("t1", "High intent",
     "A good candidate for product recommendations or a proactive chat assistant. Avoid blanket "
     "discounts. If you test one, keep a comparison group that gets no offer."),
    ("t2", "Moderate intent",
     "Worth a low-cost nudge, such as a chat offer or a saved-cart reminder. This is also a reasonable "
     "group on which to test incentives, because only a controlled test can show whether an offer "
     "changes the outcome."),
    ("t3", "Low intent",
     "No extra action. Spending on pop-ups or offers here is unlikely to be worthwhile."),
]


def intent_tier(score, cuts):
    if score >= cuts[SC_C]:
        return TIERS[0]
    if score >= cuts[SC_B]:
        return TIERS[1]
    if score >= cuts[SC_A]:
        return TIERS[2]
    return TIERS[3]


def score_bar_html(score, cuts):
    pos = lambda x: max(0.0, min(100.0, x * 100))
    ticks = ""
    for lab, key in (("A", SC_A), ("B", SC_B), ("C", SC_C)):
        ticks += (f'<div class="tick" style="left:{pos(cuts[key])}%"></div>'
                  f'<div class="ticklabel" style="left:{pos(cuts[key])}%">{lab}<br>{cuts[key]:.2f}</div>')
    return (f'<div class="scorewrap"><div class="scoretrack">'
            f'<div class="scorefill" style="width:{pos(score)}%"></div>'
            f'<div class="scoredot" style="left:{pos(score)}%"></div>{ticks}</div></div>')


# ------------------------------------------------------------------ page: Home
def page_home(b):
    st.markdown(
        '<div class="hero"><h1>Purchase Intent Predictor</h1>'
        '<p>Will this visitor buy? A machine learning model that scores online-shop sessions by how likely '
        'they are to end in a purchase.</p></div>', unsafe_allow_html=True)

    st.subheader("The business problem")
    st.write(
        "Online stores get thousands of visitors, but most never buy. Showing every visitor a discount or "
        "a chat pop-up wastes money and annoys people. A model that ranks visitors by purchase intent lets "
        "a business focus its effort: personalised recommendations, chat help, or promotions where they "
        "are most useful.")

    st.subheader("How the prediction system works")
    c1, c2, c3 = st.columns(3)
    steps = [
        ("1", "Describe a visit", "Enter what the session looked like: pages viewed, time spent, bounce "
                                  "and exit rates, month, traffic source and visitor type."),
        ("2", "The model scores it", "A Random Forest trained on 12,205 real shopping sessions returns a "
                                     "purchase likelihood score between 0 and 1."),
        ("3", "Choose an action", "Compare the score with a cutoff that matches how costly the action is, "
                                  "and read the suggested next step."),
    ]
    for col, (n, title, text) in zip((c1, c2, c3), steps):
        with col.container(border=True):
            st.markdown(f'<span class="step-num">{n}</span>', unsafe_allow_html=True)
            st.markdown(f"**{title}**")
            st.write(text)

    meta = b["meta"]
    n_test = meta["n_test"]
    test_buyers = round(meta["test_purchase_rate"] * n_test)
    st.subheader("How well does it work?")
    st.caption(f"Measured once on {n_test:,} visitors the model never saw ({test_buyers} of them bought), "
               f"using the 'Balanced' strategy.")
    colA, colB = st.columns(2)
    blocks = [(colA, VER_PV, "With PageValues",
               "Best accuracy. PageValues is a Google Analytics figure that may only be known late in a visit."),
              (colB, VER_NOPV, "Without PageValues",
               "The more cautious version for live use. It is used automatically when PageValues is unknown.")]
    for col, ver, title, blurb in blocks:
        r = test_row(b, ver, SC_B)
        with col.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(blurb)
            m1, m2 = st.columns(2)
            m3, m4 = st.columns(2)
            m1.metric("ROC-AUC", f"{b['test_auc'][ver]:.3f}",
                      help="Chance that a random buyer is ranked above a random non-buyer. 0.5 is a coin flip.")
            m2.metric("Precision", f"{r['precision']:.2f}",
                      help="Of the visitors flagged as likely buyers, the share who really bought.")
            m3.metric("Recall", f"{r['recall']:.2f}",
                      help="Of all real buyers, the share the model flagged.")
            m4.metric("F1-score", f"{r['F1']:.2f}",
                      help="A single score that balances precision and recall.")

    with st.expander("Read this before relying on the results", expanded=False):
        st.markdown(
            "- **One store, ten months.** The data covers sessions from February to December of a single "
            "shop (January and April are missing). Results may not carry over to another store or year.\n"
            "- **PageValues.** It drives most of the accuracy, and it is not clear when in a visit its value "
            "becomes available. Without it the model is useful but much weaker.\n"
            "- **Whole sessions, not live ones.** The model learned from finished sessions. It has not been "
            "tested on partial, mid-session data.\n"
            "- **Scores are not probabilities.** The model was trained to favour finding buyers, so a score "
            "of 0.70 does not mean a 70% chance of purchase. It ranks visitors.\n"
            "- **Prediction is not persuasion.** The model says who is likely to buy, not who would respond "
            "to an offer. Only a controlled test can show that.")
    st.caption("Data: UCI Online Shoppers Purchasing Intention Dataset (12,330 sessions; 12,205 after "
               "removing 125 exact duplicates).")


# ------------------------------------------------------------------ page: Visitor Prediction
def pick(options, preferred):
    return preferred if preferred in options else options[0]


def default_inputs(b):
    av = b["allowed_values"]
    return {
        "in_admin": 0, "in_admin_dur": 0.0, "in_info": 0, "in_info_dur": 0.0,
        "in_prod": 20, "in_prod_dur": 600.0, "in_bounce": 0.02, "in_exit": 0.04,
        "in_use_pv": False, "in_pv": 0.0, "in_special": 0.0,
        "in_month": pick(av["Month"], "May"), "in_weekend": False,
        "in_visitor": pick(av["VisitorType"], "Returning_Visitor"),
        "in_os": pick(av["OperatingSystems"], 2), "in_browser": pick(av["Browser"], 2),
        "in_region": pick(av["Region"], 1), "in_traffic": pick(av["TrafficType"], 2),
    }


PRESETS = {
    "Engaged shopper": {
        "in_admin": 2, "in_admin_dur": 60.0, "in_prod": 30, "in_prod_dur": 1100.0,
        "in_bounce": 0.01, "in_exit": 0.02, "in_use_pv": True, "in_pv": 25.0, "in_month": "Nov"},
    "Browsing, PageValues unknown": {
        "in_prod": 25, "in_prod_dur": 900.0, "in_bounce": 0.02, "in_exit": 0.04,
        "in_use_pv": False, "in_month": "May"},
    "Quick bounce": {
        "in_prod": 1, "in_prod_dur": 0.0, "in_bounce": 0.2, "in_exit": 0.2,
        "in_use_pv": True, "in_pv": 0.0, "in_month": "May"},
}


def apply_preset(name, defaults):
    st.session_state.update(defaults)
    for key, val in PRESETS[name].items():
        st.session_state[key] = val if (key not in ("in_month",) or val in
                                        st.session_state.get("_months", [val])) else defaults[key]
    st.session_state.pop("last", None)


NUMERIC_FIELDS = {          # widget key -> (model column, label for messages)
    "in_admin": ("Administrative", "Administrative pages"),
    "in_admin_dur": ("Administrative_Duration", "Administrative time"),
    "in_info": ("Informational", "Informational pages"),
    "in_info_dur": ("Informational_Duration", "Informational time"),
    "in_prod": ("ProductRelated", "Product pages"),
    "in_prod_dur": ("ProductRelated_Duration", "Product time"),
    "in_bounce": ("BounceRates", "Bounce rate"),
    "in_exit": ("ExitRates", "Exit rate"),
}


def validate_inputs(b):
    """Return (errors, warnings). Errors stop the prediction; warnings do not."""
    ss, errors, warns = st.session_state, [], []
    for key, (col, label) in NUMERIC_FIELDS.items():
        val = ss.get(key)
        if val is None or pd.isna(val):
            errors.append(f"**{label}** is empty. Enter a number (0 is fine).")
            continue
        lo, hi = b["numeric_ranges"][col]
        if val > hi:
            warns.append(f"**{label}** ({val:g}) is above anything seen in training (max {hi:g}). "
                         f"The prediction may be unreliable.")
    if ss.get("in_use_pv"):
        pv = ss.get("in_pv")
        if pv is None or pd.isna(pv):
            errors.append("**PageValues** is ticked as known but empty. Enter a number or untick the box.")
        elif pv > b["numeric_ranges"]["PageValues"][1]:
            warns.append(f"**PageValues** ({pv:g}) is above anything seen in training "
                         f"(max {b['numeric_ranges']['PageValues'][1]:g}).")
    if not errors:
        if ss["in_bounce"] > ss["in_exit"]:
            warns.append("The bounce rate is higher than the exit rate. Every bounce is also an exit, so "
                         "this is unusual. Please double-check both values.")
        for pages, dur, label in (("in_admin", "in_admin_dur", "administrative"),
                                  ("in_info", "in_info_dur", "informational"),
                                  ("in_prod", "in_prod_dur", "product")):
            if ss[pages] == 0 and ss[dur] > 0:
                warns.append(f"Time was entered for {label} pages, but the page count is 0. Please check.")
        if ss["in_admin"] + ss["in_info"] + ss["in_prod"] == 0:
            warns.append("No pages were entered at all, so this describes an empty session.")
    return errors, warns


def page_predict(b):
    av = b["allowed_values"]
    defaults = default_inputs(b)
    st.session_state["_months"] = av["Month"]
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

    st.title("Visitor Prediction")
    st.write("Describe one visitor session and get a purchase likelihood score. "
             "Leave PageValues unticked if you don't have it. The simpler model is then used.")

    st.markdown("**Quick examples**")
    pc = st.columns(len(PRESETS))
    for col, name in zip(pc, PRESETS):
        col.button(name, key=f"preset_{name}", on_click=apply_preset, args=(name, defaults))

    with st.form("visitor_form"):
        g1, g2, g3 = st.columns(3)
        with g1:
            st.markdown("##### Pages and time")
            st.number_input("Administrative pages", min_value=0, max_value=1000, step=1, key="in_admin",
                            help="Account-management pages visited (login, account, etc.).")
            st.number_input("Administrative time (seconds)", min_value=0.0, max_value=200000.0, step=10.0,
                            key="in_admin_dur")
            st.number_input("Informational pages", min_value=0, max_value=1000, step=1, key="in_info",
                            help="Pages such as 'about us' or 'contact'.")
            st.number_input("Informational time (seconds)", min_value=0.0, max_value=200000.0, step=10.0,
                            key="in_info_dur")
            st.number_input("Product pages", min_value=0, max_value=3000, step=1, key="in_prod")
            st.number_input("Product time (seconds)", min_value=0.0, max_value=500000.0, step=10.0,
                            key="in_prod_dur")
        with g2:
            st.markdown("##### Page metrics")
            st.number_input("Bounce rate (0 to 1)", min_value=0.0, max_value=1.0, step=0.005, format="%.4f",
                            key="in_bounce",
                            help="Average share of the visited pages' visits that left after one page. "
                                 "Typical values are 0 to 0.2.")
            st.number_input("Exit rate (0 to 1)", min_value=0.0, max_value=1.0, step=0.005, format="%.4f",
                            key="in_exit",
                            help="Average share of the visited pages' views that were the last in a session. "
                                 "Typical values are 0 to 0.2.")
            st.checkbox("I know this visit's PageValues", key="in_use_pv",
                        help="PageValues is a Google Analytics figure. If it is unknown, leave this unticked.")
            st.number_input("PageValues (used only if ticked)", min_value=0.0, max_value=5000.0, step=1.0,
                            key="in_pv",
                            help="In this model PageValues acts mostly like a switch: zero scores low, and "
                                 "any clearly positive value scores high.")
            st.selectbox("Closeness to a special day", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], key="in_special",
                         help="0 = not near a special day (e.g. Valentine's Day), 1 = right on it.")
        with g3:
            st.markdown("##### Visitor and context")
            st.selectbox("Month", av["Month"], key="in_month",
                         help="Only months present in the training data are offered.")
            st.checkbox("Weekend visit", key="in_weekend")
            st.selectbox("Visitor type", av["VisitorType"], key="in_visitor")
            st.selectbox("Operating system (code)", av["OperatingSystems"], key="in_os")
            st.selectbox("Browser (code)", av["Browser"], key="in_browser")
            st.selectbox("Region (code)", av["Region"], key="in_region")
            st.selectbox("Traffic type (code)", av["TrafficType"], key="in_traffic",
                         help="How the visitor arrived. The codes are anonymised in the dataset.")
        submitted = st.form_submit_button("Predict purchase intent", type="primary")

    if submitted:
        errors, warns = validate_inputs(b)
        if errors:
            st.session_state.pop("last", None)
            for e in errors:
                st.error(e)
        else:
            ss = st.session_state
            visitor = {
                "Administrative": int(ss["in_admin"]), "Administrative_Duration": float(ss["in_admin_dur"]),
                "Informational": int(ss["in_info"]), "Informational_Duration": float(ss["in_info_dur"]),
                "ProductRelated": int(ss["in_prod"]), "ProductRelated_Duration": float(ss["in_prod_dur"]),
                "BounceRates": float(ss["in_bounce"]), "ExitRates": float(ss["in_exit"]),
                "SpecialDay": float(ss["in_special"]), "Month": ss["in_month"],
                "OperatingSystems": int(ss["in_os"]), "Browser": int(ss["in_browser"]),
                "Region": int(ss["in_region"]), "TrafficType": int(ss["in_traffic"]),
                "VisitorType": ss["in_visitor"], "Weekend": int(bool(ss["in_weekend"])),
            }
            pv = float(ss["in_pv"]) if ss["in_use_pv"] else None
            try:
                version, score, X_new = score_visitor(b, visitor, pv)
                st.session_state["last"] = {"version": version, "score": score, "X": X_new,
                                            "warnings": warns, "pv_given": pv is not None}
            except Exception as exc:
                st.session_state.pop("last", None)
                st.error("Something went wrong while scoring this visitor. Please check the inputs.")
                st.code(str(exc))

    last = st.session_state.get("last")
    if not last:
        st.info("Fill in the form (or pick a quick example) and press **Predict purchase intent**.")
        return

    for w in last["warnings"]:
        st.warning(w)

    st.divider()
    st.subheader("Result")
    strategy_label = st.radio("Decision strategy", list(STRATEGIES), index=1, horizontal=True, key="strategy",
                              help="Broad = cheap actions such as a chat pop-up. Selective = costly actions "
                                   "such as a discount. Each strategy uses its own cutoff.")
    scenario = STRATEGIES[strategy_label]
    version, score = last["version"], last["score"]
    cuts = b["cutoffs"][version]
    cutoff = cuts[scenario]
    flagged = score >= cutoff

    left, right = st.columns(2)
    with left.container(border=True):
        st.markdown("##### Model output")
        if flagged:
            st.markdown('<span class="badge good">Likely to purchase</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge neutral">Unlikely to purchase</span>', unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m1.metric("Purchase likelihood score", f"{score:.2f}", help="Scale 0 to 1. This ranks visitors. "
                  "It is not a calibrated probability.")
        m2.metric("Cutoff for this strategy", f"{cutoff:.2f}")
        st.markdown(score_bar_html(score, cuts), unsafe_allow_html=True)
        st.caption(f"Model used: **{version}**. Marks A, B and C are the cutoffs of the three strategies.")
    with right.container(border=True):
        st.markdown("##### Suggested action (business rule, not a model output)")
        cls, title, text = intent_tier(score, cuts)
        st.markdown(f'<div class="tier {cls}"><h4>{title}</h4>{text}</div>', unsafe_allow_html=True)
        st.caption("These suggestions are rules of thumb written from the Phase 6 analysis. "
                   "They have not been A/B tested.")

    r = test_row(b, version, scenario)
    with st.container(border=True):
        st.markdown("##### How reliable is this?")
        if r is not None:
            st.write(
                f"On {b['meta']['n_test']:,} unseen visitors, the **{version}** model with the "
                f"**{strategy_label.split(' (')[0]}** strategy flagged {r['% flagged']:.0f}% of visitors. "
                f"Of every 100 flagged, about **{r['precision'] * 100:.0f}** actually bought, and it caught "
                f"about **{r['recall'] * 100:.0f}%** of all buyers.")
        if version == VER_NOPV:
            st.write("PageValues was not provided, so the simpler model was used. It ranks visitors "
                     f"clearly better than chance (test ROC-AUC {b['test_auc'][VER_NOPV]:.2f}) but is much "
                     f"less accurate than the full model ({b['test_auc'][VER_PV]:.2f}).")
        else:
            st.write("PageValues was provided, so the full model was used. Remember that PageValues may only "
                     "be known late in a real visit, which would limit live use.")
        with st.expander("Inputs sent to the model"):
            shown = last["X"].T.rename(columns={0: "value"})
            if version == VER_NOPV:
                shown.loc["PageValues", "value"] = "not provided (ignored by this model)"
            st.dataframe(shown.astype(str), use_container_width=True)


# ------------------------------------------------------------------ placeholders (Part B)
# def page_dashboard(b):
#     st.title("Analytics Dashboard")
#     st.info("This page is added in Phase 8, Part B.")


# def page_model(b):
#     st.title("Model Information")
#     st.info("This page is added in Phase 8, Part B.")

# ------------------------------------------------------------------ Part B: Analytics Dashboard

@st.cache_data
def load_dashboard_data():
    return pd.read_csv(DATA_PATH)


def page_dashboard(b):
    st.title("Analytics Dashboard")
    st.write("Explore the shopping sessions and understand the patterns behind purchase behaviour.")

    if not DATA_PATH.exists():
        st.error("The cleaned dataset was not found at data/online_shoppers_clean.csv.")
        return

    try:
        df = load_dashboard_data()
    except Exception as exc:
        st.error("Could not load the cleaned dataset.")
        st.code(str(exc))
        return

    if "Revenue" not in df.columns:
        st.error("The dataset must contain a Revenue column.")
        return

    # Convert Revenue to a consistent True/False format.
    revenue = df["Revenue"].astype(str).str.strip().str.lower()
    df["Revenue"] = revenue.isin(["true", "1", "yes"])

    total_sessions = len(df)
    total_buyers = int(df["Revenue"].sum())
    purchase_rate = (total_buyers / total_sessions * 100) if total_sessions else 0
    non_buyers = total_sessions - total_buyers

    st.subheader("Dataset overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total sessions", f"{total_sessions:,}")
    c2.metric("Purchases", f"{total_buyers:,}")
    c3.metric("Non-purchases", f"{non_buyers:,}")
    c4.metric("Purchase rate", f"{purchase_rate:.1f}%")

    st.divider()
    st.subheader("Purchase behaviour")

    left, right = st.columns(2)

    with left:
        st.markdown("**Purchase vs. non-purchase sessions**")
        outcome = pd.DataFrame({
            "Sessions": [non_buyers, total_buyers]
        }, index=["No purchase", "Purchase"])
        st.bar_chart(outcome)

    with right:
        st.markdown("**Purchase rate by visitor type**")
        if "VisitorType" in df.columns:
            visitor_rate = (
                df.groupby("VisitorType")["Revenue"]
                .mean()
                .mul(100)
                .sort_values(ascending=False)
                .to_frame("Purchase rate (%)")
            )
            st.bar_chart(visitor_rate)
        else:
            st.info("VisitorType is not available in the dataset.")

    st.divider()
    st.subheader("Purchase rate by month")

    if "Month" in df.columns:
        month_order = [
            "Feb", "Mar", "May", "June", "Jul", "Aug",
            "Sep", "Oct", "Nov", "Dec"
        ]
        monthly = (
            df.groupby("Month")["Revenue"]
            .mean()
            .mul(100)
            .reindex(month_order)
            .dropna()
            .to_frame("Purchase rate (%)")
        )
        st.bar_chart(monthly)
    else:
        st.info("Month is not available in the dataset.")

    st.divider()
    st.subheader("Session characteristics")

    left, right = st.columns(2)

    with left:
        st.markdown("**Average PageValues by purchase outcome**")
        if "PageValues" in df.columns:
            page_values = (
                df.groupby("Revenue")["PageValues"]
                .mean()
                .rename(index={False: "No purchase", True: "Purchase"})
                .to_frame("Average PageValues")
            )
            st.bar_chart(page_values)
        else:
            st.info("PageValues is not available in the dataset.")

    with right:
        st.markdown("**Purchase rate: weekday vs. weekend**")
        if "Weekend" in df.columns:
            weekend_rate = (
                df.groupby("Weekend")["Revenue"]
                .mean()
                .mul(100)
                .rename(index={False: "Weekday", True: "Weekend"})
                .to_frame("Purchase rate (%)")
            )
            st.bar_chart(weekend_rate)
        else:
            st.info("Weekend is not available in the dataset.")

    st.caption(
        "Source: UCI Online Shoppers Purchasing Intention Dataset. "
        "Charts describe the cleaned dataset; they do not represent live store activity."
    )


# ------------------------------------------------------------------ Part B: Model Information

def page_model(b):
    st.title("Model Information")
    st.write("Details about the trained models, evaluation results and decision strategies.")

    st.subheader("Model overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Model algorithm", "Random Forest")
    c2.metric("Input features", len(b["feature_columns"]))
    c3.metric("Test sessions", f'{b["meta"]["n_test"]:,}')

    st.write(
        "Two model versions are available. The first uses PageValues, "
        "while the second does not. The application automatically selects "
        "the appropriate version based on whether PageValues is provided."
    )

    st.divider()
    st.subheader("Model versions")

    for version in (VER_PV, VER_NOPV):
        model_obj = b["models"][version]

        # Display the final estimator name when the saved model is a Pipeline.
        estimator = model_obj
        if hasattr(model_obj, "named_steps") and model_obj.named_steps:
            estimator = list(model_obj.named_steps.values())[-1]

        with st.container(border=True):
            st.markdown(f"**{version}**")
            st.write(f"Estimator: `{type(estimator).__name__}`")
            st.metric("Test ROC-AUC", f'{b["test_auc"][version]:.3f}')

            if version == VER_PV:
                st.write(
                    "Uses PageValues. This version achieved stronger test "
                    "performance, but PageValues may only be available late in a session."
                )
            else:
                st.write(
                    "Does not use PageValues. This version is suitable when "
                    "that value is unavailable, although its test performance is lower."
                )

    st.divider()
    st.subheader("Test-set evaluation")

    metrics_df = pd.DataFrame(b["test_metrics"])

    if not metrics_df.empty:
        display_columns = [
            col for col in
            ["features", "scenario", "precision", "recall", "F1", "% flagged"]
            if col in metrics_df.columns
        ]

        display_metrics = metrics_df[display_columns].copy()
        display_metrics = display_metrics.rename(columns={
            "features": "Model version",
            "scenario": "Decision strategy",
            "precision": "Precision",
            "recall": "Recall",
            "F1": "F1-score",
            "% flagged": "Visitors flagged (%)"
        })

        st.dataframe(display_metrics, hide_index=True)
    else:
        st.info("Test metrics are not available in the saved model bundle.")

    st.caption(
        "These results were calculated on the held-out test set. "
        "The test set was not used to fit the models."
    )

    st.divider()
    st.subheader("Decision strategy cutoffs")

    selected_version = st.selectbox(
        "Choose model version",
        [VER_PV, VER_NOPV],
        key="model_info_version"
    )

    cutoff_rows = [
        {
            "Strategy": label,
            "Scenario": scenario,
            "Cutoff": b["cutoffs"][selected_version][scenario]
        }
        for label, scenario in STRATEGIES.items()
    ]

    st.dataframe(
        pd.DataFrame(cutoff_rows),
        hide_index=True
    )

    st.write(
        "A visitor is flagged when the model score is equal to or greater "
        "than the selected strategy's cutoff. Broad reach uses the more "
        "inclusive strategy, Balanced aims to balance precision and recall, "
        "and Selective uses the more cautious cutoff."
    )

    st.divider()
    st.subheader("Input features")

    st.write(
        f"The model receives {len(b['feature_columns'])} input features:"
    )
    st.write(", ".join(b["feature_columns"]))

    st.divider()
    st.subheader("Dataset information")

    st.write("Dataset: UCI Online Shoppers Purchasing Intention Dataset.")
    st.write("Original dataset: 12,330 online shopping sessions.")
    st.write(
        "The cleaned dataset contains 12,205 sessions after removing "
        "125 exact duplicates."
    )

    st.warning(
        "Limitations: the data comes from one online store and covers a "
        "limited period. The model was trained on completed sessions, "
        "and its scores are ranking scores rather than calibrated purchase probabilities."
    )


# ------------------------------------------------------------------ router
PAGES = {
    "Home": page_home,
    "Visitor Prediction": page_predict,
    "Analytics Dashboard": page_dashboard,
    "Model Information": page_model,
}

bundle = get_bundle()
with st.sidebar:
    st.markdown("### 🛒 Purchase Intent Predictor")
    choice = st.radio("Navigate", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.caption("Machine Learning final project. Real data: UCI Online Shoppers Purchasing Intention.")
PAGES[choice](bundle)