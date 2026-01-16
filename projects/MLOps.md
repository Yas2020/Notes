## MLOps Pipeline

### Motivation:
I wanted to learn how to take ML experiments from a Jupyter notebook into a production-ready ML system. (So Instead of focusing on extensive EDA, feature engineering, or building the best model,) I simulated a fraud detection task and concentrated on the end-to-end MLOps lifecycle.

### Pipeline Design
I used modular customized pipelines (ex. sklearn pipelines) to bundle preprocessing steps with the trained model to ensure 
- Portability: models + preprocessing can be saved or loaded easily
- Reproducibility: adopted DVC (Data Version Control) to track datasets, features, and ML artifacts.
  - Prevented redundant runs by skipping unchanged pipeline stages.
  - Allowed teammates (or future me) to reproduce exact experiments without conflicts or the need to have data locally at all.
- Used model registry to record training experiments (evaluation metrics, hyper-parameters), logged artifacts in remote and versioned them or promoted models to production stage
- Orchestrated these pipelines with Airflow DAGs for scheduling and flexible dependencies.


### Robustness at Inference:
- Trained an outlier detector on training data distributions to flag suspicious inputs at inference.
- Planned for drift detection (data distribution changes over time).
    - Common approaches: compare training vs live data distributions using KS test, Chi-square test, PSI (Population Stability Index), or embedding-based distance measures (KL divergence, Wasserstein distance).
    - Trigger alerts/retraining when drift exceeds thresholds.

### Monitoring & Observability:
- Model monitoring:
    - Logged evaluation metrics in MLflow for version comparison.
    - Implemented alerts for fraud detection rates, outlier counts, and prediction latency.
    - Added automatic rollback if thresholds were violated.
- System monitoring:
  - Collected CPU/memory/disk via Node Exporters → Prometheus → Grafana dashboards.
  - Used OpenTelemetry + Jaeger to trace requests across microservices for latency debugging.


### Questions:
- So what’s the impact of the project for business? 
  Ans: The project was designed for fraud detection, so by monitoring drift and outliers, the pipeline ensures we maintain fraud detection accuracy over time and avoid financial/business risk from model degradation.

- Why did you choose DVC over just git + S3? 
  Ans. See notes.
- How do you decide when to retrain a model?
    Ans: When drift is significant, performance on shadow validation degrades, or business KPIs drop. Could use scheduled retraining + event-triggered retraining.
- How do you ensure the pipeline is reproducible and collaborative?”
    Ans. Point to DVC + MLflow + Airflow orchestration
- What statistical tests can be used for drift detection?
    Ans: statistical tests (KS test, Chi-square, PSI, KL divergence)
- How would you scale this monitoring if the system was serving millions of requests?
  Ans:
  - Don’t test every request (too expensive). Instead:
    - Streaming random subsampling (e.g., 1–5% of traffic)
    - Reservoir sampling or sliding window aggregates for online stats
  - Use streaming frameworks (Kafka + Flink/Spark Structured Streaming) to compute drift metrics on the fly
  - Store aggregated distributions (histograms, sketches like HyperLogLog, Count-Min) rather than raw data
  - Only persist enough telemetry to detect anomalies—trade-off cost vs fidelity.
  - Batch telemetry + message queues (Kafka, Kinesis, Pulsar) → aggregate before metrics storage.
  - Use distributed monitoring (Prometheus federation, Grafana Loki/Tempo, OpenTelemetry with Jaeger in distributed mode).
  - Alerting with adaptive thresholds, not fixed ones.

- What was the biggest challenge?”
Ans.  Balancing experimentation speed vs production reliability.

------------

#### Drift detection - project extension
Demo-level “show awareness” drift detection or a production-grade scalable drift pipeline.

⏱ Option 1: Minimal / Demo (1–2 days)
- Collect recent predictions + inputs in a small store (e.g., write to parquet/CSV via Airflow task).
- Run a simple statistical test comparing current batch vs training set (Kolmogorov–Smirnov for continuous, Chi-square for categorical, or PSI).
- Output result → log to MLflow or push metric to Prometheus → alert if p-value < threshold.

👉 This would be “resume-ready” to claim drift detection as part of monitoring.

⏱ Option 2: Solid Prototype (1 week)
- Add message queue (Kafka/Kinesis) for streaming inference logs.
- Batch aggregation every hour/day.
- Run multiple tests (KS, PSI, KL divergence, population stability).
- Integrate into Grafana dashboards with red/amber/green drift alerts.

👉 This shows both detection and scaling approach, enough for interview talking points.

----------------