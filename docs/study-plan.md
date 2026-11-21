# Rush Hour

## The question

Can a model forecast the next day's hourly bike rentals using only information available at midnight, and does it improve on simply reusing last week's pattern?

This is a demand forecasting study using real aggregate Capital Bikeshare observations from 2011 and 2012. It is distinct from the completed regression uncertainty and A/B testing projects.

## Information contract

At each midnight origin, forecast all 24 hours of that calendar day. The initial implementation assumes that the previous day's final hourly count is already available. Any operational deployment must confirm or relax this reporting latency assumption.

Only prior observed rental counts and calendar features are available. Exclude actual same-day weather, casual counts and registered counts. The latter two sum to the target. Historical lags are based on timestamps, not row positions. Future outcomes must have no influence on forecast features.

The source uses naive local clock labels without offset or DST disambiguation. Treat them as supplied labels, document this limitation, and do not fabricate a timezone conversion. Missing calendar hours are unobserved. They are not known zero rental hours.

## Evaluation design fixed before model comparison

1. Use expanding training histories and two model-selection folds: train through June 2012 to score July; train through July to score August.
2. Compare previous-day and previous-week seasonal baselines, a scaled and imputed Ridge model, and a small prespecified gradient boosting search. Fit imputers and scalers on training rows only.
3. Select using mean July and August MAE. Record all tried configurations and training windows.
4. Refit the selected configuration using observations before September 1.
5. Use September solely to construct empirical forecast error intervals. Freeze the model and interval rule before October.
6. Score the final October through December window once. At each daily origin, lagged outcomes from earlier test days are permitted because they are then observed. No model retraining on test labels.
7. Record every planned forecast and missing target. Primary errors are conditional on observed targets. Report missingness separately and discuss selection bias.

The data audit reports July through September as a combined validation window. Model selection and calibration are disjoint subsets of that window.

## Planned evidence

Report MAE, RMSE and WAPE on exactly the same observed hours for every model. Specify deterministic fallbacks for missing baseline lags using only earlier history. Never compare models after dropping different rows.

Include error by hour and weekday/weekend, daily error trajectories, predicted versus observed demand, interval coverage and width, and a feature ablation for calendar-only versus calendar-plus-history. Use a bootstrap that resamples entire days to compare model errors; discuss remaining dependence across days.

Connect forecasting to an explicitly hypothetical asymmetric under/overforecast cost curve. Counts alone do not reveal inventory, staffing constraints, station locations, lost demand or operating costs. Do not claim real savings or make station-level rebalancing recommendations.

## Milestone status

November 21: complete study executed with 2,000 daily bootstrap resamples. All 19 tests pass. Six figure pairs were rendered and inspected. Ridge alpha 1 was selected; final MAE is 59.4558 rentals per hour. Code, documentation, all figures and the full result table are published at https://github.com/waleedsid/rush-hour. Automated reproduction runs on Python 3.11 and 3.12; see the repository Actions page for current check results.

## Data observations established November 20, 2026

The pinned archive contains 17,379 hourly records across 17,544 calendar hours, leaving 165 absent hours. None of the observed rows has a zero total. The final test window has 2,168 observed labels out of 2,208 planned hours. The source archive README agrees with the actual 17,379 count, while the UCI webpage's instance field currently says 17,389.

The reader checks that cnt equals casual plus registered and rejects duplicate timestamps and invalid counts. Feature tests establish that adding or removing all future outcomes leaves forecasts unchanged. A separate real-data replay check passed for the October 1 origin.

## Reproduce this milestone

Run from this directory with Python 3.10 or later. Install dependencies with python -m pip install -e . first.

```sh
python -m unittest discover -s tests -v
python -m rush_hour audit
```

The CLI downloads and verifies the archive if needed, then writes reports/data-audit.json. Raw source data stays ignored. No external account, payment or private data is required.


Before fitting on November 21: fixed configurations are yesterday, last week, Ridge alpha 1/100, and absolute-error histogram boosting with 15/31 leaves, learning rate 0.05, 200 iterations, minimum 20 samples per leaf, L2=1, and no random early stopping. Baselines fall back to the other lag, previous day mean, previous week mean, then training median. Calendar ablation uses the best learned configuration without a second search. Intervals use September signed error 5th/95th percentiles. Cost sensitivity uses under/over ratios 1, 2, 4, 8 with offsets learned only in September.
