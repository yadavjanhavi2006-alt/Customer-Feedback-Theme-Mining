# Customer Feedback Theme Mining

Run the complete pipeline from the repository root:

```powershell
python Notebook\pipeline.py
```

Or, from the `Notebook` directory:

```powershell
python pipeline.py
```

The pipeline loads `data/Reviews_3000.csv`, preprocesses review text, creates five
NMF themes, trains Naive Bayes, Logistic Regression, and SVM sentiment models, and
writes the results to `Output`.

Key generated files:

- `Output/output.csv`: consolidated, Power BI-ready enriched review data.
- `Output/model_performance.csv`: model accuracy, precision, recall, and F1 scores.
- `Output/topic_keywords.csv`: top keywords for each NMF theme.
- `Output/topic_evolution.csv`: monthly theme counts.
- `Output/dashboard.html`: self-contained interactive dashboard. Open this file in a browser.

Do not paste a project brief or natural-language requirements into PowerShell. Paste
only commands such as the examples above; PowerShell treats ordinary text as code and
will report errors such as "not recognized as the name of a cmdlet".
