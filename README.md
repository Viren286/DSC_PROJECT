# Flight Fare Analytics: Booking Horizons and Fare Dynamics

**Group Prompters**, CSE558 Data Science midterm project (Monsoon 2026)

We study how Indian domestic air fares change as the departure date approaches. The aim is to find the booking "sweet spot" and to test whether late bookings are more expensive and more volatile.

| Member | Roll No. | Contribution |
|---|---|---|
| Viren Poswal | 2024623 | Dataset & matrix expansion |
| Sahitya Sharma | 2023464 | Data preprocessing & cleaning |
| Aadit Garg | 2024003 | EDA & question formulation |
| Madhav Raj Gaur | 2024323 | Hypothesis tests 1–3 |
| Shivam Harit | 2024528 | Hypothesis tests 4–5 & limitations |

## Dataset

[Kaggle: Flight Price Prediction](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction) (`Clean_Dataset.csv`). It has 300,153 flight listings across six metro cities over 50 days, with 12 columns: airline, flight code, source and destination city, departure and arrival time slot, stops, class, duration, days left before departure, and price in INR.

## Repository contents

| File | What it is |
|---|---|
| `main_v2.ipynb` | Full analysis: cleaning, encoding, the 3M-cell size check, EDA plots, hypothesis tests and test plots |
| `hypothesis_tests.py` | The five hypothesis tests, each with its alternative tests, df, p-values and 95% CIs. The notebook imports it |
| `Clean_Dataset.csv` | Raw dataset from Kaggle |
| `report/report.pdf` | Final report (5 pages, course LaTeX template) |
| `report/report.tex`, `report/figures/` | LaTeX source and the figures exported from the notebook |
| `requirements.txt` | Python dependencies |

## How to run

```bash
pip install -r requirements.txt
jupyter notebook main_v2.ipynb        # then Kernel → Restart & Run All
python hypothesis_tests.py Clean_Dataset.csv   # tests only, prints every statistic in the report
```

Run the notebook top to bottom exactly once. Re-running the outlier cell on its own trims already-cleaned data.

To rebuild the report: `cd report && tectonic report.tex` (or `pdflatex report.tex`).

## Pipeline

1. Drop the redundant `Unnamed: 0` index. There are 0 missing values and 0 duplicate rows.
2. Remove outliers with the IQR rule: 2,110 rows on `duration` (> 30.18 h) and 123 on `price` (> Rs. 99,128). This leaves **297,920 rows**.
3. One-hot encode all categoricals: 297,920 × 1,629 = 485,311,680 cells (≥ 3M). `flight` (1,561 levels, redundant with `airline`) is then dropped, leaving 68 modelling columns.

## Hypothesis tests (α = 0.05, N = 297,920)

| # | Question | Test | Result | 95% CI | Matters? |
|---|---|---|---|---|---|
| 1 | Do fares drop on the last day? | Welch t, one-sided | t = 14.96, df = 4,634, p = 1e-49 → reject H0 | Rs. 8,672 [7,535, 9,809] | Yes (> Rs. 1,000) |
| 2 | Are late (≤ 20 d) bookings dearer than early? | Two-sample Z, two-sided | Z = 46.09, p < 1e-300 → reject H0 | Rs. 3,924 [3,757, 4,091] | Yes (> Rs. 500) |
| 3 | Are late bookings more volatile? | F-test, one-sided | F = 0.977, p = 1.0 → fail to reject | ratio 0.977 [0.967, 0.988] | No pooled; yes within each class |
| 4 | Does fare fall with days left? | t-test on OLS slope | t = −49.77, df = 297,918 → reject H0 | −152 Rs/day [−158, −146] | Yes (> Rs. 100/day) |
| 5 | Is class mix tied to booking timing? | χ² independence | χ² = 46.89, df = 1, p = 7.5e-12 → reject H0 | +1.89 pp [1.34, 2.43] | No (< 5 pp) |

The alternatives compared for each test are Student t, Mann–Whitney U, permutation and bootstrap, Bartlett, Levene, Brown–Forsythe, HC1 robust SE, Spearman, Kendall, the G-test and Fisher's exact test. Results are in the report and the script output.

**Key finding:** fares fall by about Rs. 464 per extra day of advance booking inside 20 days, but by only about Rs. 25 per day beyond that. Booking about three weeks ahead captures nearly all the savings.
