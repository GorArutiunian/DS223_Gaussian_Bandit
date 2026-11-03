import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm

from gaussian_bandit import PRIOR_MEAN, PRIOR_VARIANCE, run_gaussian_thompson

EXPERIMENTS = {
    "short": {
        "label": "Short run",
        "means": (1.0, 2.0, 3.0),
        "sigma": 1.0,
        "rounds": 100,
        "seed": 223,
    },
    "longer": {
        "label": "Longer run",
        "means": (1.0, 2.0, 3.0),
        "sigma": 1.0,
        "rounds": 1000,
        "seed": 223,
    },
    "noise": {
        "label": "Higher noise",
        "means": (1.0, 2.0, 3.0),
        "sigma": 3.0,
        "rounds": 1000,
        "seed": 223,
    },
}


def _set_inputs(config: dict) -> None:
    st.session_state["gb_mean_1"] = float(config["means"][0])
    st.session_state["gb_mean_2"] = float(config["means"][1])
    st.session_state["gb_mean_3"] = float(config["means"][2])
    st.session_state["gb_sigma"] = float(config["sigma"])
    st.session_state["gb_rounds"] = int(config["rounds"])
    st.session_state["gb_seed"] = int(config["seed"])


def _store_result(result) -> None:
    st.session_state["gb_result"] = {
        "true_means": [float(value) for value in result.true_means],
        "posterior_means": [float(value) for value in result.posterior_means],
        "posterior_stds": [float(value) for value in np.sqrt(result.posterior_variances)],
        "selection_counts": [int(value) for value in result.selection_counts],
        "rewards": [float(value) for value in result.rewards],
        "total_reward": float(result.total_reward),
        "recommendation": int(result.recommendation),
        "sigma": float(result.sigma),
        "seed": int(result.seed),
    }


def _results_frame(result: dict) -> pd.DataFrame:
    recommended = result["recommendation"]
    rows = []
    for index, true_mean in enumerate(result["true_means"]):
        rows.append(
            {
                "Alternative": f"Alternative {index + 1}",
                "True mean": true_mean,
                "Posterior mean": result["posterior_means"][index],
                "Posterior std": result["posterior_stds"][index],
                "Selections": result["selection_counts"][index],
                "Role": "Recommendation" if index == recommended else "",
            }
        )
    return pd.DataFrame(rows)


def _density_frame(result: dict) -> pd.DataFrame:
    means = np.asarray(result["posterior_means"], dtype=float)
    stds = np.asarray(result["posterior_stds"], dtype=float)
    low = float(np.min(means - 4.0 * stds))
    high = float(np.max(means + 4.0 * stds))
    if high <= low:
        high = low + 1.0
    grid = np.linspace(low, high, 250)
    frames = []
    for index, (mean, std) in enumerate(zip(means, stds)):
        role = "Recommendation" if index == result["recommendation"] else "Other"
        frames.append(
            pd.DataFrame(
                {
                    "mean": grid,
                    "density": norm.pdf(grid, loc=mean, scale=std),
                    "alternative": f"Alternative {index + 1}",
                    "series": role,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _render_results(result: dict) -> None:
    frame = _results_frame(result)
    recommended = result["recommendation"]
    recommended_name = f"Alternative {recommended + 1}"
    n_rounds = len(result["rewards"])
    selection_total = int(sum(result["selection_counts"]))

    st.subheader("Results")
    st.caption(
        f"Seed {result['seed']} · known σ = {result['sigma']:g} · "
        f"{n_rounds} rounds · prior Normal({PRIOR_MEAN:g}, {PRIOR_VARIANCE:g})"
    )
    st.success(
        f"Recommendation: {recommended_name}. "
        "It has the largest posterior mean."
    )

    total_col, rounds_col, count_col = st.columns(3)
    total_col.metric("Total observed reward", f"{result['total_reward']:.3f}")
    rounds_col.metric("Rounds", n_rounds)
    count_col.metric(f"Selections of {recommended_name}", result["selection_counts"][recommended])

    if selection_total == n_rounds:
        st.caption(f"Selection counts sum to {selection_total}, the number of rounds.")
    else:
        st.error(
            f"Selection counts sum to {selection_total}, which does not equal {n_rounds} rounds."
        )

    st.dataframe(
        frame,
        width="stretch",
        hide_index=True,
        column_config={
            "True mean": st.column_config.NumberColumn(format="%.4f"),
            "Posterior mean": st.column_config.NumberColumn(format="%.4f"),
            "Posterior std": st.column_config.NumberColumn(format="%.4f"),
            "Selections": st.column_config.NumberColumn(format="%d"),
        },
    )

    mean_chart = frame.melt(
        id_vars=["Alternative"],
        value_vars=["True mean", "Posterior mean"],
        var_name="Estimate",
        value_name="Value",
    )
    count_chart = frame.copy()
    count_chart["series"] = np.where(
        count_chart["Role"].eq("Recommendation"),
        "Recommendation",
        "Other",
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**True mean and posterior mean**")
        st.vega_lite_chart(
            mean_chart,
            {
                "mark": {"type": "bar", "cornerRadiusTopLeft": 3, "cornerRadiusTopRight": 3},
                "encoding": {
                    "x": {"field": "Alternative", "type": "nominal", "title": "Alternative"},
                    "y": {"field": "Value", "type": "quantitative", "title": "Mean"},
                    "xOffset": {"field": "Estimate", "type": "nominal"},
                    "color": {
                        "field": "Estimate",
                        "type": "nominal",
                        "scale": {"range": ["#94a3b8", "#d97706"]},
                        "legend": {"title": None, "orient": "top"},
                    },
                    "tooltip": [
                        {"field": "Alternative", "type": "nominal"},
                        {"field": "Estimate", "type": "nominal"},
                        {"field": "Value", "type": "quantitative", "format": ".4f"},
                    ],
                },
            },
            width="stretch",
        )
    with right:
        st.markdown("**Selections**")
        st.vega_lite_chart(
            count_chart,
            {
                "mark": {"type": "bar", "cornerRadiusTopLeft": 3, "cornerRadiusTopRight": 3},
                "encoding": {
                    "x": {"field": "Alternative", "type": "nominal", "title": "Alternative"},
                    "y": {"field": "Selections", "type": "quantitative", "title": "Selections"},
                    "color": {
                        "field": "series",
                        "type": "nominal",
                        "scale": {
                            "domain": ["Recommendation", "Other"],
                            "range": ["#d97706", "#94a3b8"],
                        },
                        "legend": {"title": None, "orient": "top"},
                    },
                    "tooltip": [
                        {"field": "Alternative", "type": "nominal"},
                        {"field": "Selections", "type": "quantitative"},
                    ],
                },
            },
            width="stretch",
        )

    st.markdown("**Posterior distributions**")
    st.vega_lite_chart(
        _density_frame(result),
        {
            "mark": {"type": "line", "strokeWidth": 3},
            "encoding": {
                "x": {"field": "mean", "type": "quantitative", "title": "Mean reward"},
                "y": {"field": "density", "type": "quantitative", "title": "Density"},
                "color": {
                    "field": "alternative",
                    "type": "nominal",
                    "legend": {"title": None, "orient": "top"},
                },
                "tooltip": [
                    {"field": "alternative", "type": "nominal"},
                    {"field": "mean", "type": "quantitative", "format": ".3f"},
                    {"field": "density", "type": "quantitative", "format": ".3f"},
                ],
            },
        },
        width="stretch",
    )

    st.markdown("**Cumulative reward**")
    history = pd.DataFrame({"reward": result["rewards"]})
    history["round"] = np.arange(1, n_rounds + 1)
    history["cumulative_reward"] = history["reward"].cumsum()
    st.line_chart(history.set_index("round")["cumulative_reward"])
    st.caption("Rewards stay continuous and can be negative, so the total can fall from one round to the next.")


if "gb_mean_1" not in st.session_state:
    _set_inputs(EXPERIMENTS["short"])

st.title("Gaussian Bandit")
st.caption(
    "Thompson sampling for continuous rewards. "
    "The noise standard deviation is known. The three means are not."
)

st.markdown(
    f"""
Each alternative has a prior **Normal({PRIOR_MEAN:g}, {PRIOR_VARIANCE:g})** on its unknown mean.
A round draws one candidate mean from each posterior and plays the largest draw.
The reward is one draw from Normal(true mean, σ²). Only the played alternative is updated:

`precision ← precision + 1/σ²`

`posterior mean ← (precision_before × mean + reward / σ²) / precision`

The final recommendation is the alternative with the largest posterior mean.
The same settings and seed reproduce the same run.
"""
)

st.markdown("**Load a course experiment**")
st.caption("All three use true means [1, 2, 3], the same prior, and seed 223.")
short_col, longer_col, noise_col = st.columns(3)
run_requested = False
if short_col.button("Short run · σ = 1 · 100 rounds", width="stretch"):
    _set_inputs(EXPERIMENTS["short"])
    run_requested = True
if longer_col.button("Longer run · σ = 1 · 1,000 rounds", width="stretch"):
    _set_inputs(EXPERIMENTS["longer"])
    run_requested = True
if noise_col.button("Higher noise · σ = 3 · 1,000 rounds", width="stretch"):
    _set_inputs(EXPERIMENTS["noise"])
    run_requested = True

input_cols = st.columns(3)
mean_1 = input_cols[0].number_input("True mean μ1", key="gb_mean_1", step=0.1, format="%.4f")
mean_2 = input_cols[1].number_input("True mean μ2", key="gb_mean_2", step=0.1, format="%.4f")
mean_3 = input_cols[2].number_input("True mean μ3", key="gb_mean_3", step=0.1, format="%.4f")

setting_cols = st.columns(3)
sigma = setting_cols[0].number_input("Known σ", key="gb_sigma", step=0.1, format="%.4f")
n_rounds = setting_cols[1].number_input("Rounds", key="gb_rounds", step=1)
seed = setting_cols[2].number_input("Random seed", key="gb_seed", step=1)

if st.button("Run", type="primary") or run_requested:
    means = [float(mean_1), float(mean_2), float(mean_3)]
    try:
        rounds_int = int(n_rounds)
        seed_int = int(seed)
        if float(sigma) <= 0 or not np.isfinite(float(sigma)):
            st.error("Known σ must be positive.")
        elif rounds_int < 1 or float(n_rounds) != rounds_int:
            st.error("Number of rounds must be a positive integer.")
        elif rounds_int > 100_000:
            st.error("Number of rounds must be 100000 or fewer.")
        elif seed_int < 0 or float(seed) != seed_int:
            st.error("Random seed must be a non-negative integer.")
        else:
            _store_result(
                run_gaussian_thompson(
                    true_means=means,
                    sigma=float(sigma),
                    n_rounds=rounds_int,
                    seed=seed_int,
                )
            )
    except ValueError as error:
        st.error(str(error))

result = st.session_state.get("gb_result")
if result is None:
    st.info("Choose an experiment or edit the inputs, then click Run.")
else:
    current = [float(mean_1), float(mean_2), float(mean_3), float(sigma), int(n_rounds), int(seed)]
    recorded = [
        *result["true_means"],
        result["sigma"],
        len(result["rewards"]),
        result["seed"],
    ]
    if current != recorded:
        st.warning("These inputs differ from the results below. Click Run to refresh them.")
    _render_results(result)
