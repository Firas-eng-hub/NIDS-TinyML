# Reproducibility and evaluation limits

- The main project uses 15 float features and five classes. The previous seven-feature, three-class synthetic-attack script is retained only as a local experiment and is excluded from publication, along with every notebook.
- Random row splits can share related traffic between training and test sets. Their scores do not establish generalization across capture days, networks, or attack campaigns. Consider grouped or temporal evaluation for research claims.
- The cleaner drops nonfinite values but currently allows negative durations and rates. Existing preprocessing artifacts include such values. Define and validate a domain-cleaning policy before retraining; do not silently reinterpret existing models.
- Main label mapping checks brute-force strings before web-attack strings. Web brute-force labels therefore map to BruteForce. Document any revised taxonomy before comparing new and old metrics.
- Scapy aggregation counts whole captured packet lengths and uses its own flow expiry rules. Equivalence to the dataset's CICFlowMeter extractor is unverified.
- New runs snapshot their configuration, split indices and raw CSV fingerprints. Legacy runs share mutable local split files and cannot be retroactively certified without their original inputs/configuration.
- Device report accuracy/F1 reflect valid accepted replies. Attempted, accepted and skipped rows are recorded so communication failures are visible. ROC/PR curves use PC probabilities and are labeled accordingly.
- The custom C export retains double branch thresholds and uses float leaf probabilities. Agreement must be tested, especially near thresholds; probabilities are not claimed to be bit-exact. Trees exceeding int16 node index capacity are rejected.
- STM32 fit, latency, UART reliability, and raw-feature preprocessing agreement must be measured with the target board and selected run. Host C tests do not replace hardware testing.
- The Flask server and in-memory dashboard are intended for local demonstrations. The frontend loads Chart.js from a CDN and needs network access for charts.

No datasets, split records, trained models, notebook files, or packet captures belong in the published repository. Generated outputs remain under ignored local directories.
