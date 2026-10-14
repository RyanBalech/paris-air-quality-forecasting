# Validation protocol

Every model predicts the same complete 504-hour window at each cutoff. It receives
pollutant observations only through that cutoff. Historical ERA5 weather is assumed
known for the forecast window under the competition's weather allowance.

## Development and audit

The original four windows are used to select among four fixed configurations:

| Configuration | Trees | Leaves | Minimum leaf observations | Extra summaries |
| --- | ---: | ---: | ---: | --- |
| baseline | 500 | 31 | 80 | No |
| longer_boosting | 1,200 | 31 | 80 | No |
| smoothed_history | 800 | 31 | 80 | Yes |
| regularized | 1,200 | 15 | 120 | Yes |

All configurations retain the same three-year training window, MAE objective,
learning rate, random seed and recency weights. Extra summaries include historical
pollution means over 24, 168 and 672 hours, a median across weekly pollution lags,
and 48/168-hour weather means. Pollution summaries end at least 504 hours before
the prediction timestamp.

The lowest mean development MAE determines the selected configuration. Selection is
written to disk before evaluation on the three reserved audit cutoffs:
2023-12-01 22:00, 2024-04-01 22:00 and 2024-07-01 22:00. Audit scores do not select
parameters. These windows do not overlap the development forecast windows. They
are historical audit windows, not an untouched final chronological test: observations
from earlier audit windows can occur in later development training histories.

Run the complete procedure with:

```bash
python -m paris_air_quality.improve
python -m paris_air_quality.diagnostics
```

Individual prediction files and scores are saved under `outputs/selection/`. The
cache key includes data, weather and model-source hashes and the complete search
configuration. Repeating the same run reuses completed validation fits. The selected
model is refitted on all training rows to create `outputs/submission_improved.csv`.
The original submission is preserved.

## Interpretation

The score averages the five pollutant MAEs without scaling them. Consequently,
an absolute improvement of one concentration unit in O3 affects the aggregate
equally to one unit in CO, although their numerical ranges differ considerably.
Per-pollutant and per-horizon-week errors help expose weaknesses hidden by the mean.

Missing actual labels are excluded, but every predicted value must be finite and
the prediction timestamps must exactly match the scored timestamps. Counts of
observed labels accompany every score. In particular, the July audit has only 206
observed PM10 hours out of 504.

The diagnostics include a paired bootstrap over audit windows. With only three
windows this interval is descriptive; it cannot establish reliable uncertainty
for performance across all future seasons or extreme pollution events. The
hidden Kaggle labels are unavailable, and no local score implies a leaderboard rank.
