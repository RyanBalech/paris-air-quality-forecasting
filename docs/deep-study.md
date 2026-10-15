# Broader forecasting study

## Questions

The first search changed only the number and size of LightGBM trees and a few
rolling summaries. It did not establish a meaningful audit improvement. This study
tests different assumptions about the problem:

1. Does additional weather information and a French holiday calendar help?
2. Are old pollutant lags useful once weather is available?
3. Does CatBoost provide a useful alternative to LightGBM?
4. Can separate weekly models exploit more recent observed pollution safely?
5. Do different pollutants benefit from different models or simple blends?

## Data findings

The training series contains 40,991 hourly rows. Median NO2 fell from 19.7 in
2022 to 14.3 in the available 2024 period; comparing incomplete years does not by
itself establish an annual trend. CO is missing for 98.8% of 2021. O3 occasionally
has negative measurements, and its absolute numerical range is much larger than
CO's. These features make naive interpolation, uniform preprocessing and an
aggregate score alone inadequate descriptions of model behavior.

The [competition page](https://www.kaggle.com/competitions/x-hec-ts-2025-predicting-air-quality-in-paris)
permits future weather and calendar information. It prohibits external observations
of test-period pollutant concentrations. It also states that missing hidden-test
labels were imputed using QOLMAT. Local validation here scores observed labels only;
its target distribution therefore cannot exactly reproduce the hidden evaluation.
The public leaderboard used the first week, while final evaluation used all three.
The page advertises late submission, but no late score has been obtained for this
repository and an official new competition placement is not claimed.

## Additional features and models

The additional [Open-Meteo ERA5](https://open-meteo.com/en/docs/historical-weather-api)
download contains 100-metre wind speed/direction, dew point and direct solar radiation.
Derived predictors include wind vector components, wind shear, dew-point depression,
and recent weather means. The `holidays` library supplies French public holidays,
adjacent holiday days and workday flags. Times retain the original dataset's
timezone assumption; the supplied timestamps do not specify one.

Boundary-layer height was also investigated. The API returned 4,368 missing hours
for January–June 2024 despite providing the surrounding years. It is excluded from
all predictors in this study. The exact response is cached, rather than interpolating
across six months or silently pretending the variable is complete.

The weather and calendar variants use 800 LightGBM iterations, 15 leaves and a
minimum of 120 observations per leaf. CatBoost uses 800 iterations, depth 6, learning
rate 0.05 and its [MAE objective](https://catboost.ai/docs/en/concepts/loss-functions-regression).
All use three years of training history and the same recency weighting. Separate
weekly models use minimum pollutant lags of 168, 336 and 504 hours for weeks one,
two and three respectively. Every lag and rolling statistic remains available at
the forecast origin. Tests poison future labels to check this invariant.

## Selection and evaluation

The original four development cutoffs are retained. Previously completed baseline
and regularized predictions are reused only when their data and source hashes match.
The study compares six base configurations, then selects independently per pollutant
from each single model and every equal-weight pair. This is 21 alternatives per
pollutant, so selection can still overfit four development windows.

The selected recipe is written before evaluation at three new historical cutoffs:
2023-10-01 22:00, 2024-01-01 22:00 and 2024-05-01 22:00. These forecast windows were
not scored in the earlier studies. They are historical audit windows, not a fully
untouched chronological test of the entire research process: earlier observations
can occur in later development training histories. Audit outcomes never choose the
recipe. Only the baseline and selected blend components run on these audit windows.

The report includes per-window, per-pollutant and per-week errors, observed label
counts, fit times, package versions and source/data hashes. Cache files preserve
completed fits. A three-window bootstrap is descriptive and does not establish
reliable confidence across all future conditions.

## Observed results

The selected blend uses an equal-weight weather-only LightGBM/CatBoost pair for
NO2, CO and O3; weather-only LightGBM for PM10; and an equal-weight CatBoost/weekly
LightGBM pair for PM2.5. This recipe was saved before the new audit was scored.

| Cutoff | Original baseline MAE | Selected blend MAE | Difference |
| --- | ---: | ---: | ---: |
| 2023-10-01 22:00 | 6.4683 | 6.0969 | -0.3714 |
| 2024-01-01 22:00 | 5.2316 | 5.1837 | -0.0480 |
| 2024-05-01 22:00 | 4.5186 | 4.3272 | -0.1915 |
| Mean | **5.4062** | **5.2026** | **-0.2036** |

The development means are 4.8848 and 4.5990 respectively. The audit improvement
is 3.8%. Four pollutants improve on the new audit set; PM10 regresses from 5.8449
to 5.8530 MAE. The paired fold bootstrap interval for the mean difference is
[-0.3714, -0.0480], but three correlated historical periods give limited
uncertainty information. The new audit windows differ from the earlier study's
audit set; their means must not be mixed into a single head-to-head comparison.

The same published [competition rules](https://www.kaggle.com/competitions/x-hec-ts-2025-predicting-air-quality-in-paris)
allow future weather and holidays, and exclude direct test-period pollutant data.
No external test-period pollutant measurements were used in this study. The closed
competition's late-submission control does not give this repository an official
historical placement or an independent hidden score.

## Reproduce

```bash
python -m pip install -e ".[dev,research]"
python -m paris_air_quality.advanced_weather
python -m paris_air_quality.deep_study
python -m paris_air_quality.deep_report
```

The selected submission is `outputs/submission_deep.csv`. To refit just the saved
recipe without repeating development and audit evaluation:

```bash
python -m paris_air_quality.predict_blend --report results/deep_study.json
```

Raw data, downloaded weather and prediction caches remain outside version control.
The original baseline and previous candidate submissions are preserved.

## Remaining hypotheses

This finite study does not establish an optimum. Remaining experiments include
weather timestamp alignment once the source timezone can be verified, consistent
boundary-layer data from another permitted source, regional weather/transport,
school calendars and historical traffic, and calibration to recent observed
pollution. Any further model selection needs additional evaluation periods or an
explicit acknowledgment that the existing audit windows have become development
data. Larger neural models would require their own fair comparison; model size is
not evidence of better forecasts.
