# Forecasting without tomorrow's information

## What the model knows

The forecast origin is midnight on a source calendar date. Each call produces all 24 hourly forecasts for that day, including hours whose eventual targets are absent from the source. Calendar encodings are derived from the timestamp. Rental lags and historical averages use only observations strictly before the origin.

The prediction for 23:00 therefore cannot use the same day's 22:00 outcome. This is a day ahead task, not 24 successive one hour ahead predictions. Previous test days become available history, but never become model training rows in the final evaluation.

The implementation assumes the previous day's final hourly total is available at midnight. Delayed reporting would require an earlier data cutoff. Source clock labels lack explicit timezone offsets; no daylight saving correction is invented.

## Inputs

Calendar features are sine and cosine encodings of hour, weekday and month, a weekend flag, and elapsed days since January 1, 2011. Historical features include exact clock lags of 24, 168 and 336 hours; means over the prior day and week; and counts of observed hours in those windows.

Absent lag values stay missing. Missingness counts describe what was actually observed. The targets casual and registered are excluded because they add up to the total count. Contemporaneous weather is also excluded because an observed weather reading is not a weather forecast available at midnight.

## Fixed comparisons

The prespecified search contains six configurations:

| Name | Definition |
| --- | --- |
| Yesterday | The same hour on the previous day |
| Last week | The same hour seven days earlier |
| Ridge, alpha 1 | Median imputation, missing indicators, scaling and Ridge |
| Ridge, alpha 100 | The same pipeline with stronger regularization |
| Boosting, 15 leaves | Histogram boosting with absolute error loss |
| Boosting, 31 leaves | The same learner with a larger leaf limit |

Boosting uses 200 iterations, learning rate 0.05, minimum 20 samples per leaf and L2 regularization 1. Random early stopping is disabled. Ridge transformations are fitted only on training rows. Negative point predictions are clipped to zero for every model.

If a seasonal baseline's preferred lag is missing, it uses the other lag, then the prior day mean, the prior week mean and finally the training target median. This fallback preserves common evaluation rows.

Model selection averages July and August MAE with equal weight for the two months. The July model uses only data before July 1; the August model uses only data before August 1. The selected configuration is refitted on observations before September 1, then frozen.

An ablation refits the best learned configuration with calendar features only. It has no separate hyperparameter search and is diagnostic rather than a newly selected test winner.

## Uncertainty intervals

September signed forecast errors supply the 5th and 95th percentiles. Those fixed offsets are added to subsequent point forecasts and clipped at zero. Their width therefore adapts only through clipping.

The nominal interval level is 90%. This is an empirical temporal calibration procedure, not a guarantee of 90% future coverage. Seasonal changes and serial dependence can invalidate an IID argument. Report actual October, November and December coverage and width, rather than asserting the target was achieved.

## Error measures and paired comparison

MAE and RMSE are in rentals per hour. WAPE is the sum of absolute errors divided by the sum of observed counts. It is not the average of hourly percentage errors. Bias is forecast minus actual, so positive values mean overforecasting.

Every method is evaluated on identical observed timestamps. Missing targets are excluded from scoring but retained in forecasts.csv. Conclusions are conditional on the observed hours; the data do not establish what demand occurred in missing hours.

The paired bootstrap resamples complete calendar days and recomputes the difference in hourly MAE, selected minus last week. It preserves dependence within each sampled day and unequal observed day sizes. It does not preserve dependence between adjacent days, so its percentile interval is descriptive and may understate uncertainty. It is not a claim about all future seasons.

## Hypothetical decision cost

Overforecasting has unit cost 1. Underforecasting is assigned costs 1, 2, 4 and 8. For each ratio, a forecast shift is chosen from the appropriate September residual quantile, separately for each compared method. The final loss is evaluated on observed test hours.

These are relative loss units, not dollars. The data do not provide available bikes, unmet requests, station locations, staffing decisions or supply constraints. Aggregate rental counts cannot by themselves establish optimal rebalancing actions or operating savings.

## Reproducibility and attribution

The source archive is pinned by SHA256 and downloaded from UCI. See [data attribution](../DATA_LICENSE.md). It includes the dataset DOI and the publication citation requested in the source README.

The chronological validation approach and boosting implementation can be compared with the official scikit-learn [lagged-feature example](https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html) and [histogram boosting reference](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html). This project independently implements a midnight availability contract, clock-based lags on an incomplete time grid, and a frozen selection/calibration/test procedure.
