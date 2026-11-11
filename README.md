# Purchase Intent Predictor

An end-to-end Machine Learning project that predicts whether an online shopping session will result in a purchase. The project compares different feature configurations, evaluates model performance, analyzes business decision strategies, and provides an interactive Streamlit application for making predictions.

## Live Demo

[Open the Purchase Intent Predictor App](https://purchase-intent-predictor-zkz9qzq9khfcis8xpqvacd.streamlit.app/)

---

## 1. Project Overview

Online stores receive thousands of visitor sessions, but only a small percentage result in purchases. Identifying visitors with a higher likelihood of purchasing can help businesses improve marketing decisions and allocate resources more effectively.

This project builds a machine learning solution to predict purchase intent using visitor behavior and session-related information.

The project focuses on:

- Exploring and understanding online shopping session data.
- Cleaning and preprocessing the dataset.
- Comparing feature configurations with and without `PageValues`.
- Training and evaluating classification models.
- Understanding important behavioral and session-related features.
- Evaluating different business decision strategies.
- Analyzing prediction errors and model limitations.
- Deploying an interactive Streamlit application.

## 2. Dataset

The project uses the Online Shoppers Purchasing Intention dataset.

### Dataset Overview

| Property | Value |
|---|---:|
| Total sessions | 12,330 |
| Total columns | 18 |
| Missing values | 0 |
| Duplicate rows | 125 |
| Duplicate percentage | 1.01% |
| Target variable | Revenue |

The dataset contains information about visitor activity, browsing behavior, traffic sources, and whether a purchase was completed.

### Target Distribution

The target variable is `Revenue`.

| Revenue | Sessions | Percentage |
|---|---:|---:|
| False (No Purchase) | 10,422 | 84.53% |
| True (Purchase) | 1,908 | 15.47% |

The dataset is imbalanced, with a majority-to-minority ratio of approximately **5.46:1**.

This imbalance makes accuracy alone insufficient for evaluating the models. Therefore, metrics such as precision, recall, F1-score, ROC-AUC, and average precision are considered.

### Dataset Features

The dataset includes:

- **Administrative:** Number of administrative pages visited.
- **Administrative_Duration:** Time spent on administrative pages.
- **Informational:** Number of informational pages visited.
- **Informational_Duration:** Time spent on informational pages.
- **ProductRelated:** Number of product-related pages visited.
- **ProductRelated_Duration:** Time spent on product-related pages.
- **BounceRates:** Percentage of visitors who leave after viewing a single page.
- **ExitRates:** Exit rate of pages visited during the session.
- **PageValues:** Estimated value of pages visited before completing a transaction.
- **SpecialDay:** Closeness of the browsing session to a special day or event.
- **Month:** Month of the session.
- **OperatingSystems:** Visitor operating system.
- **Browser:** Visitor browser.
- **Region:** Geographic region.
- **TrafficType:** Traffic source category.
- **VisitorType:** New, returning, or other visitor type.
- **Weekend:** Whether the session occurred on a weekend.
- **Revenue:** Whether the session resulted in a purchase.

### Numerical Data Summary

The numerical analysis showed several important characteristics:

- `PageValues` has a median of zero, indicating that at least half of the sessions have no recorded page value.
- `ProductRelated` has a median of 18 visited pages.
- `ProductRelated_Duration` has a median of approximately 599 seconds.
- `Administrative` has a median of 1 page.
- `Informational` has a median of zero.
- `BounceRates` and `ExitRates` are generally low, with median values of approximately 0.0031 and 0.0252 respectively.

Several numerical features have highly dispersed distributions, suggesting that visitor behavior varies considerably between sessions.

---

## 3. Data Preprocessing

The preprocessing workflow was designed to prepare the data for machine learning while maintaining a consistent transformation process.

### Data Quality

- No missing values were found in the original dataset.
- 125 duplicate rows were identified, representing approximately 1.01% of the dataset.
- The target variable was examined to understand class imbalance.

### Train-Test Split

The processed dataset was divided into:

| Split | Rows | Features |
|---|---:|---:|
| Training | 9,764 | 65 |
| Testing | 2,441 | 65 |

Both preprocessing recipes produced the same training and testing dimensions.

### Feature Engineering and Encoding

Categorical variables were transformed into numerical features using preprocessing pipelines.

The resulting feature space contains 65 columns.

Infrequent categories were grouped for selected categorical variables:

| Feature | Original Categories | Grouped as Infrequent |
|---|---:|---|
| Month | 10 | None |
| VisitorType | 3 | None |
| OperatingSystems | 8 | 5, 6, 7 |
| Browser | 13 | 9, 11, 12 |
| Region | 9 | None |
| TrafficType | 19 | 14, 16, 17, 18, 19 |

### Feature Scaling

Standard scaling was applied to selected numerical features.

The training data showed approximately zero mean and unit standard deviation for the scaled columns.

The test data had similar statistics, indicating consistent scaling behavior.

### Missing Values After Preprocessing

No missing values remained in either the training or testing data.

---

## 4. Modeling Approach

The project investigates two feature configurations:

### Configuration A: With PageValues

This configuration includes the `PageValues` feature.

It represents a prediction scenario where page value information is available.

### Configuration B: Without PageValues

This configuration excludes `PageValues`.

It represents a scenario where the model must rely on other visitor behavior and session-related information.

Comparing these configurations helps evaluate how much predictive information `PageValues` contributes.

### Model Evaluation Metrics

The project uses several evaluation metrics:

| Metric | Purpose |
|---|---|
| ROC-AUC | Measures the model's ability to rank positive and negative sessions |
| Average Precision | Evaluates precision-recall performance |
| Precision | Measures how many predicted buyers actually purchase |
| Recall | Measures how many actual buyers are identified |
| F1-score | Balances precision and recall |

Because the dataset is imbalanced, the evaluation considers more than accuracy.

---

## 5. Model Comparison

The following table compares Random Forest feature configurations.

### Random Forest: Feature Configuration Comparison

| Features | Month | ROC-AUC | AUC Std | Average Precision |
|---|---|---:|---:|---:|
| With PageValues | With Month | 0.933 | 0.004 | 0.751 |
| With PageValues | Without Month | 0.913 | 0.005 | 0.726 |
| Without PageValues | With Month | 0.778 | 0.012 | 0.386 |
| Without PageValues | Without Month | 0.756 | 0.014 | 0.339 |

### Key Findings

- Including `PageValues` substantially improves model discrimination.
- Including `Month` also improves performance in both feature configurations.
- The strongest configuration in this comparison is Random Forest with both `PageValues` and `Month`.
- Removing `PageValues` leads to a considerable decrease in ROC-AUC and Average Precision.

These results highlight the predictive importance of page value information.

---

## 6. Feature Importance Analysis

Permutation importance was used to measure the decrease in test ROC-AUC when individual features were shuffled.

A larger decrease indicates that the model relies more heavily on that feature.

### With PageValues

| Feature | ROC-AUC Drop | Std |
|---|---:|---:|
| PageValues | 0.2973 | 0.0085 |
| Month | 0.0404 | 0.0038 |
| ExitRates | 0.0070 | 0.0011 |
| Administrative | 0.0043 | 0.0014 |
| ProductRelated | 0.0022 | 0.0011 |
| VisitorType | 0.0022 | 0.0005 |
| BounceRates | 0.0022 | 0.0010 |
| Administrative_Duration | 0.0017 | 0.0008 |
| TrafficType | 0.0016 | 0.0007 |
| ProductRelated_Duration | 0.0014 | 0.0013 |

### Without PageValues

| Feature | ROC-AUC Drop | Std |
|---|---:|---:|
| ExitRates | 0.0649 | 0.0086 |
| ProductRelated_Duration | 0.0599 | 0.0110 |
| Month | 0.0399 | 0.0057 |
| VisitorType | 0.0113 | 0.0019 |
| ProductRelated | 0.0093 | 0.0028 |
| Administrative | 0.0093 | 0.0025 |
| TrafficType | 0.0061 | 0.0014 |
| BounceRates | 0.0047 | 0.0040 |
| OperatingSystems | 0.0035 | 0.0012 |
| Administrative_Duration | 0.0023 | 0.0020 |

### Interpretation

When `PageValues` is available, it is by far the most influential feature.

When it is unavailable, the model relies more heavily on:

- ExitRates
- ProductRelated_Duration
- Month
- VisitorType

This demonstrates how the model adapts to a different feature availability scenario.

---

## 7. Logistic Regression Interpretation

Logistic Regression coefficients and odds ratios were examined to understand the direction and strength of learned associations.

An odds ratio greater than 1 indicates a positive association with the predicted purchase outcome, while an odds ratio below 1 indicates a negative association.

These are model associations, not proof of causal relationships.

### With PageValues

**Strongest positive associations**

| Feature | Coefficient | Odds Ratio |
|---|---:|---:|
| PageValues | 1.509 | 4.522 |
| Month_Nov | 0.901 | 2.461 |
| Month_Jul | 0.338 | 1.402 |
| TrafficType_8 | 0.336 | 1.400 |
| Month_Sep | 0.306 | 1.358 |
| TrafficType_5 | 0.198 | 1.219 |
| TrafficType_20 | 0.191 | 1.210 |
| ProductRelated | 0.171 | 1.187 |

**Strongest negative associations**

| Feature | Coefficient | Odds Ratio |
|---|---:|---:|
| Month_May | -0.726 | 0.484 |
| Month_Mar | -0.518 | 0.595 |
| TrafficType_13 | -0.410 | 0.663 |
| Month_Dec | -0.315 | 0.730 |
| ExitRates | -0.299 | 0.741 |
| TrafficType_1 | -0.259 | 0.772 |
| Month_Feb | -0.248 | 0.781 |
| TrafficType_3 | -0.244 | 0.783 |

### Without PageValues

**Strongest positive associations**

| Feature | Coefficient | Odds Ratio |
|---|---:|---:|
| ProductRelated_Duration | 0.566 | 1.762 |
| Month_Nov | 0.548 | 1.730 |
| Browser_13 | 0.348 | 1.416 |
| TrafficType_8 | 0.338 | 1.403 |
| TrafficType_20 | 0.332 | 1.393 |
| TrafficType_7 | 0.329 | 1.389 |
| Month_Oct | 0.292 | 1.338 |
| TrafficType_10 | 0.286 | 1.331 |

**Strongest negative associations**

| Feature | Coefficient | Odds Ratio |
|---|---:|---:|
| ExitRates | -1.057 | 0.347 |
| TrafficType_13 | -0.558 | 0.572 |
| Month_Feb | -0.453 | 0.636 |
| TrafficType_3 | -0.353 | 0.703 |
| Month_Mar | -0.350 | 0.705 |
| Browser_3 | -0.349 | 0.706 |
| OperatingSystems_3 | -0.268 | 0.765 |
| TrafficType_1 | -0.258 | 0.773 |

### Key Findings

- `PageValues` has a strong positive association with purchase intent.
- `ExitRates` has a strong negative association, particularly without `PageValues`.
- Product browsing duration becomes more important when `PageValues` is unavailable.
- Month and traffic source categories also contribute to the model's learned patterns.

---

## 8. Visitor Ranking Analysis

Visitors were ranked by their predicted purchase scores and divided into ten equal-sized groups.

This analysis evaluates how effectively the model concentrates actual buyers near the top of the ranking.

### With PageValues

| Visitor Group | Sessions | Buyers | Purchase Rate | Cumulative Buyers |
|---|---:|---:|---:|---:|
| Top 10% | 244 | 191 | 78.3% | 50.0% |
| 10–20% | 244 | 104 | 42.6% | 77.2% |
| 20–30% | 244 | 52 | 21.3% | 90.8% |
| 30–40% | 244 | 17 | 7.0% | 95.3% |
| 40–50% | 244 | 12 | 4.9% | 98.4% |
| 50–60% | 244 | 4 | 1.6% | 99.5% |
| 60–70% | 244 | 1 | 0.4% | 99.7% |
| 70–80% | 244 | 1 | 0.4% | 100.0% |
| 80–90% | 244 | 0 | 0.0% | 100.0% |
| 90–100% | 245 | 0 | 0.0% | 100.0% |

### Without PageValues

| Visitor Group | Sessions | Buyers | Purchase Rate | Cumulative Buyers |
|---|---:|---:|---:|---:|
| Top 10% | 244 | 116 | 47.5% | 30.4% |
| 10–20% | 244 | 66 | 27.0% | 47.6% |
| 20–30% | 244 | 53 | 21.7% | 61.5% |
| 30–40% | 244 | 54 | 22.1% | 75.7% |
| 40–50% | 244 | 40 | 16.4% | 86.1% |
| 50–60% | 244 | 31 | 12.7% | 94.2% |
| 60–70% | 244 | 12 | 4.9% | 97.4% |
| 70–80% | 244 | 5 | 2.0% | 98.7% |
| 80–90% | 244 | 4 | 1.6% | 99.7% |
| 90–100% | 245 | 1 | 0.4% | 100.0% |

### Interpretation

With `PageValues`, the top 10% of visitors contains 50% of all actual buyers.

Without `PageValues`, the top 10% contains approximately 30.4% of actual buyers.

The ranking analysis demonstrates the value of `PageValues` when prioritizing visitors for targeted business actions.

---

## 9. Business Decision Strategies

The project evaluates three decision strategies based on different F-beta scores.

| Strategy | F-beta | Business Interpretation |
|---|---:|---|
| A | F2 | Recall-oriented: identifies more potential buyers |
| B | F1 | Balanced: balances precision and recall |
| C | F0.5 | Precision-oriented: targets fewer, higher-confidence visitors |

The strategies represent different trade-offs between reaching more buyers and avoiding unnecessary actions.

### Scenario Results: With PageValues

Results are expressed per 1,000 visitors.

| Strategy | Triggers | Buyers per 100 Triggers | Wasted Triggers per 100 | All Buyers Reached |
|---|---:|---:|---:|---:|
| A (F2) | 283 | 49.6 | 50.4 | 89.5% |
| B (F1) | 171 | 66.0 | 34.0 | 72.3% |
| C (F0.5) | 109 | 77.5 | 22.5 | 54.2% |

### Scenario Results: Without PageValues

| Strategy | Triggers | Buyers per 100 Triggers | Wasted Triggers per 100 | All Buyers Reached |
|---|---:|---:|---:|---:|
| A (F2) | 504 | 26.9 | 73.1 | 86.6% |
| B (F1) | 274 | 33.3 | 66.7 | 58.4% |
| C (F0.5) | 94 | 47.8 | 52.2 | 28.8% |

### Business Interpretation

**Strategy A – Recall-oriented**

- Reaches the largest share of actual buyers.
- Generates more triggers.
- May be suitable when missing a potential buyer is expensive.

**Strategy B – Balanced**

- Provides a middle ground between buyer coverage and trigger efficiency.
- Can be used as a general-purpose decision strategy.

**Strategy C – Precision-oriented**

- Targets fewer visitors.
- Produces a higher percentage of actual buyers among triggered visitors.
- May be more suitable when each intervention is costly.

The results show that the best strategy depends on the business objective and the cost of each action.

---

## 10. Discount Sensitivity Analysis

A discount sensitivity analysis was conducted to explore how different discount sizes and customer win-back rates could affect business outcomes.

### Assumptions

The analysis uses hypothetical business assumptions rather than observed financial data:

- Discount size: 5%, 10%, or 20%.
- Win-back rate: 2%, 5%, or 10%.
- Order value: 1.
- Profit margin: 30%.
- Scenario: Strategy B (F1).

The analysis estimates the financial impact of offering discounts to flagged visitors who would not otherwise purchase.

### With PageValues

Scenario B identifies 66.0 actual buyers per 100 triggers.

| Discount | Win-back 2% | Win-back 5% | Win-back 10% |
|---|---:|---:|---:|
| 5% | -3.13 | -2.88 | -2.45 |
| 10% | -6.46 | -6.26 | -5.92 |
| 20% | -13.13 | -13.03 | -12.86 |

### Without PageValues

Scenario B identifies 33.3 actual buyers per 100 triggers.

| Discount | Win-back 2% | Win-back 5% | Win-back 10% |
|---|---:|---:|---:|
| 5% | -1.33 | -0.83 | 0.00 |
| 10% | -3.06 | -2.66 | -2.00 |
| 20% | -6.53 | -6.33 | -5.99 |

### Interpretation

Under the stated assumptions, most discount scenarios produce negative estimated outcomes.

This highlights an important business consideration: identifying a likely buyer does not automatically mean that offering a discount is profitable.

The discount strategy should account for whether the visitor needs an incentive to purchase and whether the additional profit can cover the discount cost.

These figures are illustrative estimates, not measured revenue or profit.

---

## 11. Visitor Type Analysis

Visitor types were examined under Strategy B (F1) to understand how prediction performance differs between new and returning visitors.

### With PageValues

| Visitor Type | Flagged Visitors | Actual Buyers | Precision |
|---|---:|---:|---:|
| New_Visitor | 82 | 74 | 0.902 |
| Other | 1 | 0 | 0.000 |
| Returning_Visitor | 335 | 202 | 0.603 |

### Without PageValues

| Visitor Type | Flagged Visitors | Actual Buyers | Precision |
|---|---:|---:|---:|
| New_Visitor | 168 | 71 | 0.423 |
| Other | 3 | 0 | 0.000 |
| Returning_Visitor | 499 | 152 | 0.305 |

### Interpretation

- With `PageValues`, new visitors have higher precision than returning visitors.
- Without `PageValues`, precision decreases for both groups.
- The `Other` category contains very few observations, so its results are not reliable.

Small groups, especially those with fewer than approximately 50 flagged visitors, should be interpreted cautiously.

---

## 12. Bootstrap Evaluation

Bootstrap evaluation was performed using 1,000 resamples of the test set under the Scenario B cutoff.

The results below show the reported performance estimates and their 95% bootstrap ranges.

### With PageValues

| Metric | Estimate | 95% Bootstrap Range |
|---|---:|---:|
| ROC-AUC | 0.935 | 0.922–0.947 |
| Precision | 0.661 | 0.616–0.705 |
| Recall | 0.723 | 0.676–0.767 |
| F1-score | 0.690 | 0.652–0.727 |

### Without PageValues

| Metric | Estimate | 95% Bootstrap Range |
|---|---:|---:|
| ROC-AUC | 0.786 | 0.765–0.809 |
| Precision | 0.332 | 0.296–0.368 |
| Recall | 0.584 | 0.536–0.633 |
| F1-score | 0.423 | 0.386–0.463 |

### Interpretation

The bootstrap results support the earlier finding that the model performs substantially better when `PageValues` is available.

The ranges also provide an indication of how much the evaluation metrics vary across bootstrap resamples of the test set.

---

## 13. Error Analysis

Error analysis was conducted under Scenario B to understand the characteristics of correctly and incorrectly classified sessions.

The four outcome groups are:

- **TP (True Positive):** Actual buyers correctly identified.
- **TN (True Negative):** Non-buyers correctly ignored.
- **FP (False Positive):** Non-buyers incorrectly flagged as potential buyers.
- **FN (False Negative):** Actual buyers missed by the model.

### 13.1 Error Counts

| Outcome | With PageValues | Without PageValues |
|---|---:|---:|
| True Negatives (TN) | 1,917 | 1,612 |
| True Positives (TP) | 276 | 223 |
| False Positives (FP) | 142 | 447 |
| False Negatives (FN) | 106 | 159 |

The configuration with `PageValues` produces fewer false positives and false negatives in this evaluation.

### 13.2 Median Feature Values: With PageValues

| Outcome | PageValues | ExitRates | BounceRates | ProductRelated | ProductRelated_Duration | Administrative |
|---|---:|---:|---:|---:|---:|---:|
| FN | 0.000 | 0.024 | 0.004 | 34.0 | 1429.365 | 3.0 |
| FP | 17.426 | 0.017 | 0.002 | 36.0 | 1250.709 | 4.0 |
| TN | 0.000 | 0.029 | 0.005 | 15.0 | 454.289 | 0.0 |
| TP | 27.970 | 0.013 | 0.000 | 25.5 | 940.525 | 2.0 |

### 13.3 Median Feature Values: Without PageValues

| Outcome | PageValues | ExitRates | BounceRates | ProductRelated | ProductRelated_Duration | Administrative |
|---|---:|---:|---:|---:|---:|---:|
| FN | 13.913 | 0.026 | 0.004 | 20.0 | 612.819 | 2.0 |
| FP | 0.000 | 0.013 | 0.000 | 34.0 | 1253.176 | 3.0 |
| TN | 0.000 | 0.035 | 0.008 | 12.0 | 337.175 | 0.0 |
| TP | 23.276 | 0.012 | 0.000 | 36.0 | 1405.327 | 2.0 |

### 13.4 New Visitor Distribution Across Error Groups

The following percentages represent the share of each outcome group that consists of new visitors.

| Outcome | With PageValues | Without PageValues |
|---|---:|---:|
| FN | 21.7% | 16.4% |
| FP | 5.6% | 21.7% |
| TN | 12.5% | 9.3% |
| TP | 26.8% | 31.8% |

This analysis shows that visitor type distribution differs across correct predictions and prediction errors.

### 13.5 Missed-Buyer Rate by Month

The missed-buyer rate is calculated among actual buyers for each month.

**With PageValues**

| Month | Missed Buyers | Actual Buyers | Missed-Buyer Rate |
|---|---:|---:|---:|
| Aug | 6 | 14 | 42.9% |
| Dec | 12 | 37 | 32.4% |
| Jul | 6 | 15 | 40.0% |
| June | 3 | 6 | 50.0% |
| Mar | 3 | 33 | 9.1% |
| May | 14 | 79 | 17.7% |
| Nov | 52 | 161 | 32.3% |
| Oct | 3 | 18 | 16.7% |
| Sep | 7 | 19 | 36.8% |

**Without PageValues**

| Month | Missed Buyers | Actual Buyers | Missed-Buyer Rate |
|---|---:|---:|---:|
| Aug | 9 | 14 | 64.3% |
| Dec | 14 | 37 | 37.8% |
| Jul | 10 | 15 | 66.7% |
| June | 5 | 6 | 83.3% |
| Mar | 14 | 33 | 42.4% |
| May | 39 | 79 | 49.4% |
| Nov | 48 | 161 | 29.8% |
| Oct | 12 | 18 | 66.7% |
| Sep | 8 | 19 | 42.1% |

### Error Analysis Findings

- The configuration with `PageValues` has fewer false positives and false negatives.
- Without `PageValues`, the model misses a larger share of actual buyers in several months.
- June has a high missed-buyer rate in both configurations, but its small number of actual buyers makes the percentage sensitive to individual observations.
- November has the largest number of missed buyers in the configuration with `PageValues`, partly reflecting its larger number of actual buyers.
- Error patterns suggest that performance can vary across visitor segments and months.

---

## 14. Streamlit Application

The project includes an interactive Streamlit application that makes the prediction workflow accessible through a user-friendly interface.

### Application Pages

**1. Home**

Provides an overview of the project and its purpose.

**2. Visitor Prediction**

Allows users to enter visitor session information and receive a purchase-intent prediction.

The application supports model selection based on whether `PageValues` is available.

**3. Analytics Dashboard**

Presents analytical insights and model-related results through visual summaries.

**4. Model Information**

Provides information about the available models and their evaluation.

### Prediction Considerations

The prediction score should not be interpreted as a calibrated probability.

The model provides a score that can help rank sessions and support business decisions, but it does not guarantee that a visitor will purchase.

---

## 15. Key Business Insights

The analysis produced several important findings:

1. **PageValues is highly informative.** Including it substantially improves ranking and classification performance.

2. **Visitor ranking can support targeted actions.** With PageValues, the top 10% of ranked visitors contains 50% of actual buyers.

3. **Business strategy matters.** Recall-oriented strategies reach more buyers, while precision-oriented strategies reduce the number of triggered visitors.

4. **Discounts are not automatically profitable.** The sensitivity analysis suggests that discount size and customer win-back rate must be considered together.

5. **Visitor segments behave differently.** New and returning visitors show different precision levels depending on feature availability.

6. **Prediction errors require attention.** Monthly and visitor-type error analysis can help identify areas where the model needs further investigation.

---

## 16. Limitations

- The dataset is imbalanced, with only 15.47% of sessions resulting in purchases.
- The data represents a specific online shopping dataset and may not generalize to every e-commerce business.
- `PageValues` has a major influence on predictive performance.
- The model's scores are not calibrated probabilities.
- Business scenario results depend on the selected decision thresholds and assumptions.
- The discount sensitivity analysis uses hypothetical financial assumptions rather than actual business transaction data.
- Some visitor categories and monthly groups have relatively small sample sizes.
- Historical patterns may not represent future visitor behavior.

---

## 17. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- Jupyter Notebook

---

## 18. Project Structure

The project contains the following main files and directories:

```text
purchase-intent-predictor/
│
├── data/
│   └── online_shoppers_clean.csv
│
├── streamlit/
│
├── app.py
├── model.joblib
├── purchase_intent_predictor.ipynb
├── requirements.txt
│
└── README.md
```

The structure above shows the main project files and directories. Additional files may be present depending on the deployment and development setup.

---

## 19. Installation and Setup

### Prerequisites

- Python installed on your machine.
- pip package manager.

### Step 1: Clone the Repository

```bash
git clone YOUR_REPOSITORY_URL
```

Replace `YOUR_REPOSITORY_URL` with the actual repository URL.

### Step 2: Navigate to the Project Directory

```bash
cd purchase-intent-predictor
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 4: Activate the Virtual Environment

**Windows:**

```bash
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser.

Make sure the required model and dataset files are available at the paths expected by the application.

---

## 20. Conclusion

The Purchase Intent Predictor demonstrates how machine learning can be used to analyze online shopping behavior and support purchase-intent prediction.

The comparison between feature configurations highlights the importance of `PageValues`, while ranking analysis, business strategy evaluation, and error analysis provide a broader understanding of how model predictions can be used in practical business scenarios.

The Streamlit application brings these results into an interactive environment, allowing users to explore the project and make predictions using visitor session information.

The project combines data preprocessing, machine learning, model evaluation, business-oriented analysis, and application deployment into one end-to-end workflow.
