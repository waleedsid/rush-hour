# Data provenance and attribution

Dataset: Bike Sharing, donated by Hadi Fanaee-T (2013), UCI Machine Learning Repository.

Dataset DOI: https://doi.org/10.24432/C5W894

Source page: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset

Download: https://archive.ics.uci.edu/static/public/275/bike%2Bsharing%2Bdataset.zip

Retrieved November 20, 2026. The source page states Creative Commons Attribution 4.0 International: https://creativecommons.org/licenses/by/4.0/

Pinned archive SHA256:
`b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401`

The archive's Readme.txt additionally requests this publication citation:

Fanaee-T, Hadi, and João Gama. Event labeling combining ensemble detectors and background knowledge. Progress in Artificial Intelligence (2013). https://doi.org/10.1007/s13748-013-0040-3

The source aggregates Capital Bikeshare system rental records and adds weather and calendar variables. This project uses only hourly total counts and timestamps for its initial forecasting features. Counts are not a measure of latent unmet demand or station availability.

Transformations: CSV type validation, timestamp construction, sorting, missing calendar-hour audit, prior-count lags and past-only summaries, and calendar encodings. The original archive remains unmodified and is not committed. Derivative reports must retain this attribution and identify their transformations. No endorsement by the dataset authors or Capital Bikeshare is implied.

The data license is separate from any eventual license for this project's original source code.

