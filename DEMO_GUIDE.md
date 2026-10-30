# Demo Guide — Classroom Presentation Flow

A 10-step walkthrough for presenting the project live. Everything runs from the
browser after `streamlit run app.py` — no terminal interaction needed.

1. **Introduce the problem and dataset.** On *Overview*: 7,043 customers of a
   fictional telecom provider, ~27% churned historically. Churn = customer left;
   `Churn = Yes` is the positive class. Emphasize this is a public sample.
2. **Show the churn distribution.** The bar chart shows class imbalance —
   explain why accuracy alone is misleading.
3. **Explore segments.** On *Data Explorer*, set the breakdown to *Contract*
   (month-to-month churn is highest) and *PaymentMethod*. Note the language:
   "higher observed churn rate," not "causes churn."
4. **Show data quality work.** On *Data Quality & Cleaning*: 11 whitespace
   `TotalCharges` entries, the zero-tenure correction to 0, duplicates retained
   and reported, raw file untouched. Outliers are inspected, not deleted.
5. **Compare the models.** On *Model Lab* → *Compare both*: both models share
   the same stratified 70/15/15 split; compare F1 and ROC-AUC side by side.
6. **Move the threshold down** (e.g., 0.30): more customers flagged, recall
   rises, precision falls — visible in the confusion matrix and the flagged count.
7. **Move the threshold up** (e.g., 0.70): fewer flags, precision rises, recall
   falls. Ask the class which error is costlier: a missed churner or a wasted
   retention offer.
8. **Explain why ROC-AUC never moved:** it is computed from the probability
   ranking across all thresholds, so the slider cannot change it.
9. **Show the fixed test evaluation.** The held-out test block (threshold 0.50)
   never responded to the slider — that is the honest, unbiased performance
   estimate. It was never used for tuning.
10. **Interpret and close.** Open the coefficient expander / *Insights &
    Limitations*: top positive and negative Logistic Regression coefficients are
    log-odds associations, not causal effects. End with the limitations list
    (risk estimates, not certainty; no intervention outcomes; fairness care for
    demographic features).
