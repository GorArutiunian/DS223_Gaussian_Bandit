# Gaussian Thompson sampling

Streamlit frontend for a carousel bandit, extended with Thompson sampling for continuous Gaussian rewards. The carousel pages come from the course app in [`app/front`](https://github.com/hovhannisyan91/DS223_Bayesian_AB_Testing/tree/main/app/front) of [DS223_Bayesian_AB_Testing](https://github.com/hovhannisyan91/DS223_Bayesian_AB_Testing), which is the app running at <https://carousel-mvp.streamlit.app/>. The assignment names [DS223_Group_Project](https://github.com/hovhannisyan91/DS223_Group_Project) as the course starting repository.

- GitHub: https://github.com/GorArutiunian/DS223_Gaussian_Bandit
- Deployed app: https://ds223gaussianbandit-jstmax5ztwbhfmk3tbqhv8.streamlit.app/

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/). From this directory:

```bash
uv sync --locked
uv run streamlit run main.py
```

The app opens on Create Carousels. Gaussian Bandit is the new page. Dependencies are locked in `uv.lock`. Streamlit Community Cloud should use that file, branch `main`, entrypoint `main.py`, and Python 3.12.

## Original carousel algorithm

The original pages learn click-through rates. Each alternative is a bandit with a Beta posterior. A carousel shows only `visible_count` of its alternatives each round.

**Alternative selection.** The prior is Beta(1, 1), a uniform distribution on the click probability. Each round, `sample_carousel` draws one probability from every alternative with `random.betavariate(alpha, beta)`, sorts those draws from high to low, and displays the top `visible_count`. That is Thompson sampling for a top-k carousel: an alternative is more likely to be shown when its posterior puts weight on a high click rate, and an uncertain alternative can still be drawn into the visible set.

**Reward recording.** The Interaction page records what happened to the alternatives that were actually shown. A click stores reward 1 for the clicked alternative and reward 0 for the other visible ones. "No click / Next round" stores reward 0 for every visible alternative. Each of those alternatives also gains one impression. Alternatives that were not displayed are left unchanged. The event is appended to that carousel's history with the round number, the displayed ids, the clicked id, and the 0/1 reward.

**Beta posterior updates.** A click adds 1 to α and to the click count. A non-click adds 1 to β. This is the conjugate Beta–Bernoulli update. The posterior mean α / (α + β) is the current estimate of the click rate. Analytics ranks alternatives by that mean and calls the largest one the champion. Ties go to more clicks, then more impressions, then the smaller alternative id.

**Session state.** `init_state` keeps `carousels` and `active_carousel_id` in `st.session_state`. Streamlit stores that dictionary for the browser session and every page imports the same helpers, so a carousel created on Create Carousels is the one Interaction updates and Analytics plots. The stored object holds the alternatives table, the round number, the current display, and the history. Nothing is written to a database. Refreshing or starting a new session clears it. The original code leaves API calls as TODOs.

## Gaussian Thompson sampling

The new page uses three alternatives and continuous rewards:

```text
R | A = i ~ Normal(μ_i, σ²)
```

σ > 0 is known and shared. The learner does not know μ_i. Every unknown mean starts at the same prior, Normal(0, 25): prior mean 0, prior variance 25, prior precision 1/25.

Each round:

1. Draw a candidate mean from each alternative's current Normal posterior.
2. Play the alternative with the largest draw. Ties keep the smallest index.
3. Simulate one reward from Normal(μ_selected, σ²). The true mean is used only here.
4. Update only the played alternative.

The precision update is

```text
precision ← precision + 1/σ²
posterior mean ← (precision_before × mean + reward / σ²) / precision
```

Known σ enters as the reward precision 1/σ². It is not estimated. A smaller σ adds more precision per observation, so the posterior concentrates faster. Rewards stay continuous and may be negative. The recommendation is the alternative with the largest posterior mean after the last round, not the alternative that was played most often.

The same true means, σ, round count, prior, and seed reproduce the same rewards and posteriors. The seed feeds NumPy's `Generator` and is not the global random state.

## Experiments

All three runs use true means [1, 2, 3], prior Normal(0, 25), and seed 223. Numbers below come from NumPy 2.4.4 in `uv.lock`. The Gaussian Bandit buttons run these settings.

| Experiment | σ | Rounds | Posterior means | Selections | Total reward | Recommendation |
| --- | ---: | ---: | --- | --- | ---: | --- |
| Short run | 1 | 100 | 0.3032, 1.5021, 3.0377 | 2, 4, 94 | 292.348 | Alternative 3 |
| Longer run | 1 | 1,000 | 0.3032, 1.5925, 2.9935 | 2, 5, 993 | 2981.341 | Alternative 3 |
| Higher noise | 3 | 1,000 | −0.1376, 1.4371, 2.9935 | 8, 27, 965 | 2928.023 | Alternative 3 |

Posterior standard deviations, in the same alternative order:

| Experiment | Posterior std |
| --- | --- |
| Short run | 0.7001, 0.4975, 0.1031 |
| Longer run | 0.7001, 0.4454, 0.0317 |
| Higher noise | 1.0376, 0.5735, 0.0966 |

Selection counts sum to the number of rounds in each run: 100, 1,000, and 1,000. Repeating a configuration with seed 223 reproduced the rewards, posterior means, and selection counts exactly.

**More observations.** Going from 100 rounds to 1,000 rounds tightens Alternative 3. Its posterior mean moves from 3.0377 to 2.9935, nearer the true mean 3, and its posterior standard deviation falls from 0.1031 to 0.0317. Almost all of the extra rounds are spent on that alternative: it is selected 94 times, then 993 times.

The unexpected part is Alternative 1. Its posterior mean is still 0.3032 after 1,000 rounds, and it is still selected only twice. Both of those plays happen early. Thompson sampling then commits to Alternative 3 and never returns, so more data does not repair the estimate of an arm it has stopped playing. Two observations, shrunk toward the prior mean of 0, leave Alternative 1 far below its true mean of 1. The procedure is trying to collect reward, not to estimate every mean equally well. The recommendation is still correct because Alternative 3's posterior mean is clearly the largest.

**Higher noise.** With σ = 3, each reward adds precision 1/9 instead of 1. The same 1,000 rounds explore more: Alternative 1 is played 8 times and Alternative 2 is played 27 times, compared with 2 and 5 when σ = 1. Alternative 3 is still played 965 times. Its posterior mean is 2.9935, essentially the same point estimate as in the low-noise long run, but its posterior standard deviation is 0.0966 rather than 0.0317, about three times wider, in line with the noise ratio. The recommendation remains Alternative 3.

Alternative 1's posterior mean becomes negative (−0.1376) even though the true mean is 1. That is what a prior at 0 and a few noisy observations can do when the algorithm does not keep sampling that arm. A negative estimate is allowed. The total reward is also noisier, but the decision rule only compares posterior means, and Alternative 3 still leads.

## Presentation outline

About four minutes:

1. The carousel draws a click probability from each Beta posterior and shows the largest draws. A click increments α. A miss increments β. Pages share one session-state dictionary.
2. Gaussian rewards replace the Beta model. σ is known, so the update adds 1/σ² to the precision of the played arm only. Selection uses posterior draws, never the true means.
3. Run the short experiment on the deployed page. Alternative 3 takes 94 of 100 plays and is the recommendation. Alternatives 1 and 2 stay close to the prior.
4. Run the longer experiment. Alternative 3's posterior tightens around 3. Alternative 1 does not move, because it is not played again.
5. Run the higher-noise experiment. The best arm is still identified, its posterior is wider, and the weaker arms get a few more plays. One of those posterior means is negative.

## Project layout

```text
main.py                       navigation for the four pages
bandit_utils.py               carousel state, Beta sampling, click updates
gaussian_bandit.py            Gaussian Thompson sampling
views/Create_Carousels.py     original carousel setup
views/Interaction.py          original click feedback
views/Analytics.py            original Beta analytics
views/Gaussian_Bandit.py      inputs, validation, and result charts
pyproject.toml
uv.lock
```

The page files live in `views/` rather than a folder named `pages`. With the locked Streamlit version, a `pages` folder replaces `st.navigation` and adds an extra page named `main`.
