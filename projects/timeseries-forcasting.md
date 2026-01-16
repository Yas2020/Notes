With your resume, a manager will see potential but will want verification on core ML competence, problem-solving, and authenticity. We can break it into 4 key pillars to focus on:

🧠 ML Engineer Verification Checklist

- 1️⃣ **Core ML Foundation**
  - **Explain a model end-to-end**: Pick one (e.g. XGBoost, Logistic Regression, CNN) → walk through data prep → training → evaluation → deployment.
    - ✅ Goal: Show you understand the full ML lifecycle, not just using libraries.
  - **Feature engineering reasoning**: Be able to say how you selected, encoded, normalized, and handled missing features — especially in the forecasting or NLP projects.
  - **Evaluation clarity**: Mention precision/recall, ROC-AUC, BLEU, or other metrics, and explain why you chose them for each task.

- 2️⃣ **NLP & LLM Understanding**
  - **Tokenization → Embeddings → Model → Output**: Be able to explain each step simply — show that you understand how text becomes model input.
  - **Fine-tuning process**: Explain LoRA, reward modeling, and PPO alignment in your RLHF project in your own words — key signals: “frozen base”, “low-rank adapters”, “scalar reward”.
  - **RAG pipeline mental model**: Clearly describe retrieval → generation → post-processing. Mention one or two optimization tricks (e.g., using embeddings or caching retrieved chunks).
  
- 3️⃣ **Coding & Implementation**
  - **Python / PyTorch fluency**: Be ready to write or describe quick snippets — e.g. vectorizing a function, defining a simple model class, or loading a dataset dynamically.
  - **Data structures & logic thinking**: For LeetCode-type screens, demonstrate systematic thinking: explain your plan first, then write concise, readable code.

- 4️⃣ **MLOps & System Integration**
  - **MLOps pipeline logic**: Be able to verbally diagram your DVC + Airflow pipeline — what’s data ETL, preprocessing, training, registry, and how drift/monitoring works.
  - **CI/CD in ML context**: Explain why you used GitHub Actions + CodeBuild + CDK — and how it ensures reliable deployment (testing, rollback, GitOps loop).
  - **Observability**: Mention Prometheus/Grafana dashboards, drift alerts, latency monitoring — and say why these are critical for production ML.
  - **Containerization & orchestration**: Show you understand Docker + ECS/EKS basics, especially resource isolation and scaling for inference.
  
- 5️⃣ **Authenticity & Communication**
  - **Your contribution clarity**: For each project, clearly state:
→ “I focused on adapting this part / refactoring that / implementing these tests / adding CI/CD / learning these components.”
    - ✅ That defuses suspicion and earns respect.
  - **One deep story per project**: Pick 1–2 technically rich details you fully own — walk them through with confidence. (e.g., how you debugged an AWS CDK error, or why PPO reward model needed normalization.)
  - **Learning mindset statement**: Whenever you feel an area is weaker, say:
“I built on an existing repo to understand production-grade patterns — then customized and extended it to learn the internals hands-on.”
    - ✅ That transforms “cloned project” into “engineer who learns from real codebases”.

✅ Key Mindset for Interviews
- **Authenticity first**: Clearly state your role, contributions, and learning.
- **Depth over breadth**: One detailed, solid story beats 5 shallow ones.
- **Decision-driven explanation**: Every time you describe a project, answer “why did I do it this way?”
- **Confidence & calm**: The resume signals strong potential; your job is to verify it with clear examples and reasoning.


## Time Series Forecasting

Dataset was a real records from a retailer containng historical daily sales for about 21k items sales across 60 stores from Jan,2013-July,2015. Each record has features such as `shop_id`, `item_id`, `item_category_id`, `city`, `item_price`, `item_cnt_day`.  The goal was to build a monthly item-store sales forecasting system to help optimize inventory and reduce stockouts for a retail chain, using over 2.5 years of sales data from 60 stores and 21K items.

I performed EDA: removed outliers, filled/removed of missing data, treated categorical variables (label_encoding), dropping duplicates. 

Performed advanced feature engineering including:
- Aggregates: monthly sales and average price at item, category, shop, and city levels
- Lag features: created rolling lags for sales and price metrics
- External features: included number of holidays per month
- Memory optimization: downcast data types from int16 to int8 to scale to 12M+ rows

Next, I preprocessed data by building necessary features for every (`shop_id`, `item_id`) pairs indicating `year_of_pred`, `month_of_pred` and then concatnate them vertically to the main Pandas dataframe. First downsampled records (adding daily sales up a month- make data montly) pair, recasted data types in lower precsion point (from 16int to 8int) to save memory. With the new montly data, I created features such as, #holidays, monthly sales per `item_id` across all shops, average sales per `item_id_category` across shops and per shops, per city, created average price per month and calcualted at category, shop and city level. Simialr features for price change percentage Created lags for every new feature. The final dataset had shape (12299567, 105), that is 105 features per row. I trained a XGboost model on this massive dataset on GCP VertexAI. 

I was also interested in making forcasting models using classic SARIMA and Prophet. I reduced the size of data to only 5 stores and 10 categories. Then divided the data into subsets based on stores and categories and trained 50 Phophet models only based on item sales in month feature - ensuring the correct column names (ds for time and y for the value). Data on the month before prediction month was hold as validation set. Did similar with SARIMA but decided not to use SARIMA results. 

Created a simple ensamble model by applying a regularized linear model on top of the prediction of XGBoost plus meta features from Prophet on the validation set. 

```
meta_features = [
  xgb_pred,
  prophet_pred,
  prophet_trend,     # from Prophet component
  prophet_seasonal,  # if exposed
  item_popularity,   # high/low volume
  city_id / shop_id  # geo effects
]
```

For the final prediction, I applied the trained linear model on the predition of XGBoost and Prophet.

XGBoost captured interaction-heavy patterns, while Prophet captured trend/seasonality for sparse subsets. A linear ensemble on validation predictions improved overall stability and generalization.

### Summary

### Goal

Forecast next-month sales at the item-store level using 2.5 years of historical sales data (~12M records) to optimize inventory planning for a retail chain

#### Dataset
- Historical daily sales from Jan 2013 to Jul 2015
- 60 stores × 21,000 items → ~12.3M monthly item-store rows
- Features: shop_id, item_id, city, item_category, item_price, item_cnt_day

#### Approach
- Preprocessing & Feature Engineering
    - Aggregated daily → monthly sales  (fine because end user only cares about monthly totals)
    - Added 100+ features to embed both *univariate history* and *cross-sectional signals*.
      - lagged sales/price: Lag features (previous month’s sales, rolling means, etc.): classic and effective for capturing autocorrelation
      - Aggregated group features (avg. sales by category, shop, city, etc.): good way to inject cross-series information
      - Monthly averages by category/shop/city, # holidays, price change %, etc. Percentage change features help the model learn growth/decay trends, not ju vcxst raw scale.
      - Creating hierarchical levels (store/item → category → region) is especially useful when many time series are sparse.
    - Downcast data types (int16 → int8) to reduce memory
  effectively 

- Models
  - XGBoost: trained on entire dataset for tabular regression
    - Training a local model per series is impossible at this scale.
    - A single global model can leverage patterns across stores/items (e.g. similar items in different stores behave alike).
  - Gradient boosting is strong for heterogeneous tabular features. 
- Evaluation: rolling CV respecting time order
  - **Expanding window**: train on $[t_0, t_N]$, validate on $[t_{N+1}, t_{N+H}]$. Then train on $[t_0, t_{N+1}]$, validate on next window.
  - **Sliding window**: train on $[t_K, t_N]$, validate on $[t_{N+1}, t_{N+H}]$. Then train on $[t_{K+1}, t_{N+1}]$, validate on next window. 
  - In my case: 
    - Fold1 train: months 1-24 --> val: 25-27
    - Fold2 train: months 1-27 --> val: 28-30
  - Use these folds to tune hyper-parameters, estimate deployment performance
- Metrics to compute (per-series & aggregated):
  Per series (for each series i):
  - MAE_i (mean absolute error) — robust, interpretable.
  - RMSE_i — penalizes large errors.
  - RMSLE_i — if relative percent errors matter (sales scale varies).
  - MAPE_i (be careful for zeros) or use SMAPE.
  - MASE (Mean Absolute Scaled Error) — scale-free, good for comparison across series.

  Aggregate & reporting:
  - Median MAE across series (more robust than mean).
  - Weighted MAE (weights = revenue or avg sales) — reflects business impact.
  - Percentiles of error distribution (10th, 50th, 90th).
  - % of series with error < threshold (e.g., MAE < 10 units or MAPE < 20%).

  Compute both unweighted and revenue-weighted aggregates — long tail vs business impact.

- Residual analysis at scale (how to do it efficiently)
Residuals are crucial. Do it at two levels: **global diagnostics** and **sampled deep diagnostics**.

  Global / automated checks (fast, for all 120k):
  - Compute `residual = y_true - y_pred` for each (series, date) in validation folds.
  - Per series, compute:
    - mean(residual) → bias (systematic over/under prediction).
    - std(residual) → volatility of errors.
    - autocorrelation of residuals at lag 1 (`acf1`) — quick indicator: if large, model misses temporal dependencies.
    - Flag series that exceed thresholds (e.g., |mean| > 0.2*mean_sales OR acf1 > 0.3 OR MAE > business_threshold). These go to detailed review.

  Sampled detailed analysis (for flagged / representative series):
  - **Plot residuals over time** (look for seasonality patterns, changepoints).
  - **ACF/PACF** of residuals: if strong seasonal lags show up → model missing seasonality.
  - **Error vs predicted** scatter: heteroscedasticity / scale issues.
  - **Error distribution** (histograms / QQ plots) to check heavy tails.
  - **Error by feature bins** (e.g., error binned by product price, by item age, by store size) to find conditional bias.

- Results
  - Scalable pipeline run on GCP Vertex AI for model training and tuning

Practical pipeline: how to run this end-to-end (checklist)
- Implement rolling-origin splits and produce y_true/y_pred for each fold.
- For each series compute: MAE, RMSE, MASE, mean(residual), acf1. Save into series_metrics table.
- Compute global summaries: median MAE, revenue-weighted MAE, 90th percentile MAE.
- Flag series by business rules (e.g., top 5% revenue & MAE > threshold) and inspect these manually.
- For flagged series run ACF/PACF and residual plots; test Fourier/harmonic features and re-evaluate.
- If persistent seasonal signal, run aggregate Prophet and attempt dis-aggregate + ensemble as described previously.

### 🗣️ STAR Format Story (for Interviews)

#### Situation
I worked on a project to forecast monthly sales at the item-store level using a large retail dataset. The goal was to help optimize inventory and reduce stockouts for a retailer by providing accurate forecasts one month in advance.

#### Task
I needed to build a scalable, accurate forecasting model from 2.5 years of daily sales records (Jan 2013 – July 2015), covering over 21,000 items across 60 stores. The challenge was processing this data efficiently and capturing both trends and non-linear interactions.

#### Action
- Performed EDA: cleaned missing values, removed outliers, encoded categorical variables, and downcast data types to reduce memory footprint.
- Aggregated daily sales into monthly item-store records and engineered over 100 features (very typical) including lagged sales, average prices by category/shop/city, and holiday counts to capture multiple seasonality, series level metadata, external regressors (prices, promotions, macro indicators)
- Trained a high-performance XGBoost model on GCP Vertex AI using the full dataset (~12M records).
- Used regularization hyper-parameters lambda, alpha (L1/L2) to avoid overfitting due to having 100+ features 
- Monitored feature importance and pruned redundant features
- Used rolling-origin cross validation 
- Evaluate per series and aggregate metrics: examine error distribution (long tails vs top items)
- Use metrics: MAE (robust), MAPE(if scales), business KPIs (stokout reduction % if available)


#### Result
- Created a scalable, cloud-based forecasting pipeline
- Project demonstrated effective integration of tabular ML and time series methods, which could be productionized to support retail planning
- Model can be improved by  later adding external features (weather, promotions, macroeconomic indicators).



#### Other possible approaches:
- Hierarchical Forecasting: explicitly enforce consistency between item → category → store → region → total.
- Daily to Monthly via Forecast Reconciliation: Instead of aggregating first, predict daily and then sum up (sometimes better accuracy, but heavier compute). In this method, you have longer time series (daily vs monhtly) which is better for statical models such as Prophet or SARIMA
- **Ensemble**: Aggregate-then-Reconcile using Prophet: it is not practical to train 120k Prophet model (inefficient). To bring the number down, train Prophet models at higher levels then disaggregate as follows: 
  - Fit **Prophet** on aggregates (e.g., per category, per store, region, or total chain). This gives smooth forcasts for trend + seasonality
  - Forecast the aggregate at daily, then sum to monthly.
  - Disaggregate those aggregate forecasts back to item-store level using learned historical shares or a small model. 
    - Define:
  \[
\text{share}_{i,s}(t) = \frac{y_{i,s}(t)}{\sum_j y_{j,s}(t)}
  \]

      item $i$ in store $s$ at time $t$.
    - Forecast with Prophet at higher level (ex, store level).
    - Multiply Prophet 's forecast x historical shares to obtain item-store level Prophet forecast  
  - Blend disaggregated Prophet with your XGBoost per series. Some options are:
    - Weighted Average Ensemble of XGBoost prediction and Prophet prediction: 
    \[
      y_{i,s}(t) = wy^{X}_{i,s}(t) + (1-w)y^{P}_{i,s}(t)
    \]

      and choose $w$ as hyperparameter from cross validation (grid search 0.2-0.8)
    - Stacking (meta-learner):
      - Features:
        - XGBoost forcast
        - Prophet forcast
        - Maybe residual from validation sets
      - Train a ridge regression as meta model - this is more flexible than fixed weight
  - Benefit: you get clean seasonal/trend signals at higher levels (where Prophet shines) without N=120k models.
  -  Validation: backtest on rolling windows (last 3 month) and measure improvement of ensemble vs single model XGBoost
  -  Tip: Add Prophet if you see stable, strong seasonal /trend signals at aggregate or category level, or XGBoost underperforms on seasonal/holiday periods
- Add **Fourier seasonal features** (weekly/yearly) + holiday flags directly to XGBoost. That’s the cheapest “Prophet effect.” 
- Hybrid models: XGBoost and LightGBM stacking, or XGBoost and small LSTM features - experiment if time permits
- Statistical models like Prophet or SARIMA dont scale but they can be use with aggregation-disaggregation technique or used for the top-k revenue/volume/variance series (top 1-5%). For example, you can use them to generate residual series or seasonal (trend, seasonal amplitude) and feed them into XGBoost.

### Questions


🔹 High-level / Framing
1- What was the business problem and why did you choose to aggregate daily → monthly? 
 -  Ans. Sale Forecasting. It is a standard business problem helps optimizing inventory storage cost and possibly business loss due to overstocking over time. Business goal was to predict monthly sales, daily data was noisy because some items were returned (negative numbers for sales). It balances signal and noise—daily had too much volatility, while monthly gives actionable insight for planning.
   
2- Why did you decide to use a global model (XGBoost across all series) instead of building local models per series?
  - Ans:  A global model also helps transfer learning across series with similar dynamics, which individual models would miss. 
  
3- How do you define “success” for this forecasting problem? Which metric(s) are most meaningful for stakeholders?
  - Ans: MAPE, global weigthed error w.r.t. business value of items, storage volume decrease while meeting demand. 

🔹 Feature Engineering
4- You mentioned lag features and aggregated percentage features — can you walk me through one concrete example of a feature you engineered and why it helped?
- Ans. The most important is lag features out of the target value, in this case, item sales. Created 6-12 month lag of item sales for every prediction of on month in future. Similarly for other features. Used feature importance tools to remove weak lag features. Lag features helps a model to understand auto-correlations and seasonality in timeseries over time. This is why they are needed for forecasting. Because this series is multivarivte, we need to create lags for all features. Lag features effectively turn a time series problem into a supervised regression problem.

5- How did you handle seasonality (monthly/annual effects)?
- Ans. adding lag features, holiday features (counts in month), month number, year number. Month/year numbers help the model learn seasonality even without explicit sine/cos transforms.

6- How did you ensure your lag features didn’t leak information from the future?
- Ans. For every prediction at given time $t^*$, we only use data at time $t<t^*$ to build features.   
  
🔹 Modeling Choices
7- Why XGBoost? Did you try other baselines (ARIMA, Prophet, LightGBM, neural nets)? 
- Ans. XGBoost is a powerful gradient boosting model which is tree-based and can capture complicated interactions when features are rich. It's also scalable and fast which is suitable for this dataset. It can find patterns for every store-item pair separately in a single model. Statistical models can only  handle one timeseries at a time so are not easily scalable. Neural nets need more data per time series than is available in this data set. But i did try LSTM on daily dataset but the result was not better. LightGBM is also a reasonble choice but its result was not better than xgbbost on this dataset. In practice, XGBoost achieved the best accuracy-speed tradeoff on this dataset

8- What are the trade-offs between using a tree-based regressor like XGBoost vs. a time series model like SARIMA or Prophet? 
- Ans. Prophet and SARIMA are interpretable and clear output in terms of seasonality and trends but are not scalable. They can be great for quick baselines or explainability.
  
9- If you were to build an ensemble, how would you do it here?
- Ans. Use statistical model at high aggregate level and disaggregate using historical data and so their prediction is used as features in XGBoost. This way, we can leverage interpretability of statistical models and accuracy of boosting models. Alternatively,  stack them as meta-learners followed by a ridge regression for final prediction. 
  
🔹 Validation & Evaluation
10- You used last months for validation/test — what are the pros and cons of that vs. rolling-origin cross validation?
- Ans. Using CV for time series instead of train-test-split has the advantage the model will be trained and evaluated on all parts of the data. Much better choice than splitting. 
  
11- How did you evaluate performance per series? Did you check whether errors are systematically worse for certain store-item pairs?
- Ans. Yes, I computed evaluation metrics per series, residual errors per series, residual plots and qq plots per series to analyze residuals  
   
12- Can you describe how you’d perform residual analysis at scale across 120k series?
- Ans. at global and flagged/representative series: compute mean, std for residuals to compare. Compute ACF (lag1) to see if it passes a threshold (seasonality not fully captured). For flagged series, look at residual plots, error plots, acf, pacf to make sure seasonality and autocorrelation is captured and residuals are just white noise.

🔹 Scalability & Production
13- Training on 120k time series: how did you handle computational efficiency (memory, speed)?
- Ans. Recast data types (int16 --> int8), float32 --> float16, use generators instead of loops, use distributed training algorithms, distributed data storage, Pre-cache intermediate results to avoid recomputation.
  
14- How would you deploy this forecasting system so it runs monthly in production?
- Ans. Retrain the model every month and predict one one month in future . This needs to be done ahead of time for the business to implement it.  Evaluate model against shadow/holdout data available later. Adjust model if needed. Monitor errors so predictions stays within a threshold- rollback or retrain if metrics exceed thresholds

15- How would you set up drift detection / monitoring to flag if the forecasts stop being reliable?
- Ans.  z_scores, chi-squared for predictions to create a threshold . if outside that , review predictions. Store data per series to have a sample of data. any drift from that measure by KL-divergence should alert the admin about predictions. Thresholds should be empirically set based on historical error distribution + business tolerance.

🔹 Critical Thinking
16- What limitations does your current solution have?
If the forecast horizon changes (say weekly instead of monthly), how would your approach adapt? 
- Ans. Retrain model on weekly data

17- If new stores or items appear that weren’t in training, how would you handle them?
- Ans. There must be a layer of input type/format check at inference such as categorical values or numerical values such as outlier detection. Either map them to “other” bucket, retrain encoding, or flag for retraining.

18- How would you explain the predictions to a non-technical stakeholder (store manager, exec)?
- Ans. I would explain the how most important features contribute to the final predictions and why it makes sense. price, promotions, holidays, recency lags, seasonality markers (month, week), store ID 
