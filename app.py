# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from collections import Counter
st.write("App is running successfully 🚀")
st.set_page_config(layout="wide", page_title="Behavioural Bot Detection", page_icon="🤖")

st.title("Behavioural Fake Engagement Detector")
st.markdown("Simple dashboard: authenticity score, bot probability and anomaly explanations (unsupervised).")

# ---------------------
# Load saved artifacts
# ---------------------
@st.cache_resource
def load_artifacts():
    iso = joblib.load("isolation_forest_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_cols = joblib.load("feature_columns.pkl")
    df = pd.read_csv("final_output.csv")
    with open("anom_min_max.json", "r") as f:
        anom_range = json.load(f)
    return iso, scaler, feature_cols, df, anom_range

iso, scaler, feature_cols, df, anom_range = load_artifacts()

# ---------------------
# Sidebar: quick stats
# ---------------------
with st.sidebar:
    st.header("Dataset summary")
    st.write(f"Total accounts: {len(df)}")
    pct_high = (df['bot_probability'] > 0.7).mean() * 100
    st.write(f"% flagged high risk (p > 0.7): {pct_high:.1f}%")
    st.write("Average authenticity:", df["authenticity_score"].mean().round(3))

# --- Coordinated anomaly detection ---
suspicious_df = df[df["bot_probability"] >= 0.7]

if len(suspicious_df) > 0:

    # explode list of anomaly drivers
    all_drivers = suspicious_df["anomaly_explanation"].explode()

    driver_counts = all_drivers.value_counts(normalize=True)

    # if a feature appears in more than 40% of suspicious accounts
    coordinated_features = driver_counts[driver_counts > 0.4]

    if len(coordinated_features) > 0:
        st.markdown("## Coordinated Behaviour Detected")
        st.write("Repeated anomaly drivers across high-risk accounts:")
        st.write(list(coordinated_features.index))
    else:
        st.markdown("## No Strong Coordination Pattern Detected")

else:
    st.markdown("## No High-Risk Accounts Available for Coordination Analysis")

# ---------------------
# Top row: plots
# ---------------------
col1, col2 = st.columns([1.2, 1])
with col1:
    st.subheader("Bot Probability distribution")
    fig, ax = plt.subplots(figsize=(6,3))
    ax.hist(df["bot_probability"], bins=20)
    ax.set_xlabel("Bot Probability")
    ax.set_ylabel("Number of accounts")
    st.pyplot(fig)

with col2:
    st.subheader("Most common anomaly drivers")
    # Collect explanation lists (assumes anomaly_explanation is list-like string or Python list)
    # Ensure we convert strings back to lists if required
    explanations = df["anomaly_explanation"].apply(lambda x: eval(x) if isinstance(x, str) else x)
    all_feats = []
    for lst in explanations:
        if isinstance(lst, (list, tuple)):
            all_feats.extend(lst)
    feat_counts = Counter(all_feats)
    feat_df = pd.DataFrame.from_dict(feat_counts, orient="index", columns=["count"]).sort_values("count", ascending=False)
    fig2, ax2 = plt.subplots(figsize=(6,3))
    feat_df.plot(kind="bar", legend=False, ax=ax2)
    ax2.set_ylabel("Count in top-3 explanations")
    ax2.set_xlabel("")
    st.pyplot(fig2)

# ---------------------
# Middle: Top suspicious table
# ---------------------
st.subheader("Top suspicious accounts")
topk = st.slider("Show top N suspicious accounts", 5, 20, 10)
top = df.sort_values("bot_probability", ascending=False).head(topk)[["id","bot_probability","authenticity_score","anomaly_explanation"]]
st.dataframe(top.reset_index(drop=True), use_container_width=True)

# ---------------------
# Bottom: Live prediction (user input)
# ---------------------
st.subheader("Live account risk check (enter account-level features)")

with st.form("predict_form"):
    c1, c2, c3 = st.columns(3)
    followers = c1.number_input("followers", min_value=0, value=1000, step=1)
    following = c2.number_input("following", min_value=0, value=500, step=1)
    statuses_count = c3.number_input("statuses_count (total tweets)", min_value=0, value=200, step=1)
    tweet_count = st.number_input("tweet_count (tweets in sample)", min_value=0, value=50, step=1)
    avg_tweet_length = st.number_input("avg_tweet_length", min_value=0.0, value=120.0, step=1.0)
    domain_count = st.number_input("domain_count", min_value=0, value=2, step=1)
    verified = st.selectbox("verified", (0,1))
    submitted = st.form_submit_button("Compute risk")

def compute_features_single(followers, following, statuses_count, tweet_count, avg_tweet_length, domain_count, verified):
    ff_ratio = followers / (following + 1)
    followers_log = np.log1p(followers)
    following_log = np.log1p(following)
    statuses_log = np.log1p(statuses_count)
    ff_ratio_log = np.log1p(ff_ratio)
    activity_per_follower = statuses_count / (followers + 1)
    engagement_proxy = following / (followers + 1)
    # If your model used scaled/weighted linguistic features, you can let the user input them too (optional)
    # For now set neutral defaults:
    lexical_diversity = 0.4
    repetition_ratio = 0.01
    retweet_ratio = 0.05
    # If your feature_cols expect scaled versions (e.g., lexical_diversity_scaled), adapt them here
    feature_vector = {
        "followers_log": followers_log,
        "following_log": following_log,
        "statuses_log": statuses_log,
        "verified": verified,
        "tweet_count": tweet_count,
        "avg_tweet_length": avg_tweet_length,
        "domain_count": domain_count,
        "ff_ratio_log": ff_ratio_log,
        "activity_per_follower": activity_per_follower,
        "engagement_proxy": engagement_proxy,
        "lexical_diversity": lexical_diversity,
        "repetition_ratio": repetition_ratio,
        "retweet_ratio": retweet_ratio
    }
    return feature_vector

# --- Replace the previous live-predict block with this ---
if submitted:
    # compute the feature dict from user inputs
    feats = compute_features_single(followers, following, statuses_count,
                                    tweet_count, avg_tweet_length,
                                    domain_count, verified)

    # turn into single-row DataFrame (only contains the features we can compute)
    X_input_partial = pd.DataFrame([feats])

    # feature_cols is the full list used at training time (loaded from feature_columns.pkl)
    expected_cols = feature_cols  # already loaded at top of file

    # Compute training-set means to fill missing features (safe defaults)
    # df is the training dataframe loaded earlier
    training_means = df[expected_cols].mean()

    # Build a full input row that has all expected columns (in correct order)
    # Start from training means, then overwrite with values we have from the user's inputs
    X_input_full = pd.DataFrame([training_means.values], columns=expected_cols)
    # Overwrite the columns we computed from user input
    for col in X_input_partial.columns:
        if col in expected_cols:
            X_input_full.loc[0, col] = X_input_partial.loc[0, col]

    # Ensure numeric dtype and no NaNs
    X_input_full = X_input_full.astype(float).fillna(training_means)

    # Now scale and predict
    X_scaled = scaler.transform(X_input_full)    # scaler was trained on expected_cols order
    raw_score = iso.decision_function(X_scaled)  # bigger = normal
    anomaly_score = -raw_score

    # Normalize using stored min/max (from training df)
    anom_min = anom_range["anom_min"]
    anom_max = anom_range["anom_max"]
    bot_prob = (anomaly_score - anom_min) / (anom_max - anom_min)
    bot_prob = float(np.clip(bot_prob, 0.0, 1.0))
    authenticity = 1.0 - bot_prob

    st.metric("Bot probability", f"{bot_prob:.3f}")
    st.metric("Authenticity score", f"{authenticity:.3f}")
    # --- Risk classification ---
    if bot_prob < 0.4:
        risk_label = "Low Risk (Likely Organic)"
        risk_color = "green"
    elif bot_prob < 0.7:
        risk_label = "Medium Risk (Borderline Behaviour)"
        risk_color = "orange"
    else:
        risk_label = "High Risk (Likely Artificial Engagement)"
        risk_color = "red"

    st.markdown(f"### Risk Classification: :{risk_color}[{risk_label}]")

    # Explanation: compute deviation vs training means and show top 3
    row = X_input_full.iloc[0]
    deviations = (row - training_means).abs().sort_values(ascending=False)
    top_feats = list(deviations.head(3).index)
    st.write("Top anomaly drivers (reason):", top_feats)

st.markdown("---")
st.caption("Notes: This app uses account-level behavioral features to score the risk of artificial engagement. A single comment text alone is NOT sufficient to detect account-level behaviour.")