# Spend Data Quality Workbench

A small analytics tool for the first step of any procurement analytics project: **making spend data trustworthy before anyone builds a dashboard on top of it.**

Spend data usually arrives with the same supplier spelled five different ways, duplicate records, missing category codes and the occasional amount that is off by a factor of 1,000. Dashboards built on that data understate supplier concentration and misstate category spend. This tool cleans up supplier names, measures what is wrong, and ranks the problems by how much spend they touch, so the team knows what to fix first.

**Live demo:** _[add Streamlit Cloud link after deploying]_

Built with Python, Streamlit, pandas, SciPy, RapidFuzz and Plotly.

---

## What it does

| Tab | Purpose |
|---|---|
| **Spend cube** | Cleaned spend by category, year and supplier, with filters and CSV export. |
| **Supplier matching** | Distinct spellings vs. resolved suppliers, top-10 concentration before and after cleaning, match quality (precision / recall / F1) and a ranked list of merges a human should review first. |
| **Data quality report** | Five quality dimensions, an issue table showing rows *and spend* affected, and a "fix first" list. |
| **Method** | The methodology in short form. |

You can run it on the built-in synthetic benchmark or upload your own CSV (map the supplier, amount, date and category columns in the sidebar; date and category are optional).

## Method

**1. Standardise.** Any table is mapped to an internal schema (supplier, amount, date, category code). Amounts are parsed robustly (currency symbols, decimal comma vs. decimal point), dates are parsed leniently, empty strings become missing values.

**2. Resolve suppliers.**
- Names are casefolded, stripped of diacritics and punctuation, and legal-form tokens (Oy, Ab, Ltd, GmbH, ...) are removed from either end.
- Similarity between distinct names is the RapidFuzz **token-sort ratio**, which ignores word order.
- Names are grouped with **average-linkage hierarchical clustering** (UPGMA). Cutting the tree at `1 - threshold/100` guarantees that every merge has an average internal similarity of at least the chosen threshold.
- The canonical name of a cluster is its most frequent original spelling.

**3. Check quality.** Rule- and statistics-based checks:
- Completeness (missing supplier, amount, date, category)
- Validity (malformed category codes, non-positive amounts)
- Uniqueness (duplicates are searched on the *resolved* supplier + date + amount + category, so re-typed names still match)
- Plausibility (outliers via the modified z-score of `log10(amount)`, computed within each CPV division, flagged above 3.5; Iglewicz & Hoaglin, 1993)
- Name consistency (share of rows whose spelling differs from the canonical one)

**4. Translate findings into impact.** Each issue is reported with the share of rows *and* the share of spend it affects. Severity and the "fix first" list are driven by the larger of the two, because a defect on 1% of rows can still touch a large share of spend.

## Evaluation on synthetic data

Real spend data has no answer key, so the demo generator simulates public-procurement awards with **hidden ground truth**: known supplier identities, injected duplicates, injected outliers and missing or invalid values. It includes deliberately hard cases such as distinct suppliers with near-identical names ("Nordic Freight Oy" vs. "Nordic Freight Services Oy"), and spend follows a heavy-tailed (Pareto) distribution. This makes the matching objectively measurable.

Supplier matching at the default threshold (88), pairwise metrics on distinct spellings:

| Data messiness | Distinct spellings | Suppliers after cleaning (true: 156) | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Low | 1,041 | 179 | 0.990 | 0.876 | 0.929 |
| Medium | 1,704 | 205 | 0.998 | 0.899 | 0.946 |
| High | 2,197 | 239 | 0.998 | 0.893 | 0.943 |

The threshold sweep in the app shows the trade-off explicitly. On the medium dataset, F1 peaks around a threshold of 86 (0.955); raising it to 96 gives perfect precision but recall falls to 0.41. The default of 88 leans towards precision on purpose: a false merge silently combines two suppliers' spend, while a missed merge is visible in the review list.

Duplicate detection finds far more duplicates after supplier consolidation than with exact-name matching (medium messiness: 339 vs. 73), which is the practical argument for resolving suppliers first.

Two caveats on the duplicate and outlier check shown in the app:
- The row-level score reads low for duplicates (precision around 0.5) partly by construction: the first record of a pair is kept and the later one flagged, which may be the original rather than the injected copy. Evaluated per duplicate pair, recall is about 93% / 75% / 56% for low / medium / high messiness. It drops as messiness rises because the category code is part of the duplicate key and gets blanked or corrupted independently on the copy.
- Outlier precision is lower by design: the log-normal amounts contain genuine extreme values that the injection did not create.

All numbers use a fixed random seed and are reproducible.

## Design decisions

- **Hierarchical clustering instead of "match to the closest name".** Greedy or nearest-neighbour matching is order-dependent and can chain unrelated names together. Average linkage gives a clean, explainable guarantee about every merge.
- **Robust statistics for outliers.** Spend amounts are heavy-tailed, so mean and standard deviation are themselves distorted by the errors they are meant to catch. Median/MAD on the log scale is not.
- **Impact over counts.** The issue table is sorted by share of spend affected, not by row count.
- **Human in the loop.** The tool does not silently rewrite data. It ranks merges by their weakest internal link so a reviewer checks the riskiest ones first.
- **Guardrails.** The similarity matrix is quadratic in distinct names, so the app refuses more than 4,000 unique names and tells the user to narrow the scope instead of running out of memory.

## Limitations

- Matching is lexical. Unrelated trading names of the same company will not merge, and similar names of different companies can.
- No use of business IDs (e.g. Finnish Y-tunnus) or addresses, which would be the strongest matching signal in real data.
- Outlier detection is per category; a wrong amount that is plausible for its category is not caught.
- In-memory and single-file by design; it is not built for multi-million-row datasets.
- The demo data is synthetic. Results on real data will differ, mainly because real name variation is less regular than simulated variation.

## Run locally

Requires Python 3.10+.

```bash
git clone <repo-url>
cd <repo-folder>
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

`requirements.txt`:

```
streamlit>=1.50
pandas>=2.0
numpy>=1.26
scipy>=1.11
rapidfuzz>=3.0
plotly>=5.18
```

## Using your own data

Upload a CSV with at least a supplier-name column and an amount column. Optional: an award/invoice date and a category code (8-digit CPV format is validated; other code schemes will be flagged as malformed, so leave the category column unmapped in that case). Delimiter and encoding (UTF-8 or Latin-1) are detected automatically. Data is processed in memory only and is not stored.

## Project structure

```
app.py             # data generation, supplier resolution, quality checks, UI
requirements.txt
README.md
```

## Author

Farid Lolo, MSc Industrial Engineering & Management (LUT University). [LinkedIn](https://linkedin.com/in/faridlolo)
