"""Hypothesis tests for the Flight Fare Analytics midterm report (group Prompters).

Each test reports the chosen test, the alternatives we compared it against, the test
statistic, degrees of freedom, p-value, decision, and the estimate with a 95% CI.

Usage:
    python hypothesis_tests.py Clean_Dataset.csv      # standalone
    from hypothesis_tests import clean, run_all       # from the notebook
"""
import sys

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05
N_TESTS = 5
RNG = np.random.default_rng(42)


def clean(raw):
    """Same pipeline as the notebook: drop index, missing rows, duplicates, IQR outliers."""
    df = raw.drop(columns=[c for c in raw.columns if c.startswith("Unnamed")])
    df = df.dropna().drop_duplicates()
    mask = pd.Series(True, index=df.index)
    for col in ["duration", "days_left", "price"]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        mask &= df[col].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    return df[mask].reset_index(drop=True)


def decision(p):
    return "Reject H0" if p < ALPHA else "Fail to reject H0"


def welch(a, b):
    """Mean difference a-b with Welch SE, df and 95% CI."""
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    dof = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    diff = a.mean() - b.mean()
    h = stats.t.ppf(0.975, dof) * se
    return diff, se, dof, (diff - h, diff + h)


def cohen_d(a, b):
    sp = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp


def permutation_pvalue(a, b, n_perm=10_000):
    """One-sided permutation p-value for mean(a) - mean(b) > 0."""
    pooled = np.concatenate([a, b])
    observed = a.mean() - b.mean()
    count = 0
    for _ in range(n_perm):
        RNG.shuffle(pooled)
        count += pooled[: len(a)].mean() - pooled[len(a):].mean() >= observed
    return (count + 1) / (n_perm + 1)


def bootstrap_ci(a, b, n_boot=5_000):
    diffs = [RNG.choice(a, len(a)).mean() - RNG.choice(b, len(b)).mean() for _ in range(n_boot)]
    return np.percentile(diffs, [2.5, 97.5])


def header(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def test1_last_day_drop(df):
    header("TEST 1: Last-day price drop  |  Welch two-sample t-test, right-tailed")
    print("H0: mu(day 2) - mu(day 1) <= 0    H1: mu(day 2) - mu(day 1) > 0")
    print("alpha = 0.05, smallest meaningful difference = Rs 1,000")
    d1 = df.loc[df.days_left == 1, "price"].to_numpy(float)
    d2 = df.loc[df.days_left == 2, "price"].to_numpy(float)
    diff, se, dof, ci = welch(d2, d1)
    t, p = stats.ttest_ind(d2, d1, equal_var=False, alternative="greater")
    print(f"n(day 2) = {len(d2):,}, n(day 1) = {len(d1):,}; means = {d2.mean():,.0f} vs {d1.mean():,.0f}")
    print(f"SD day 2 = {d2.std(ddof=1):,.0f}, SD day 1 = {d1.std(ddof=1):,.0f}")
    print(f"Welch t = {t:.3f}, df = {dof:,.1f}, p = {p:.3e} -> {decision(p)}")
    print(f"Estimate = Rs {diff:,.0f}, 95% CI [{ci[0]:,.0f}, {ci[1]:,.0f}], Cohen's d = {cohen_d(d2, d1):.3f}")
    print("Alternatives:")
    ts, ps = stats.ttest_ind(d2, d1, equal_var=True, alternative="greater")
    print(f"  Student t (pooled var): t = {ts:.3f}, df = {len(d1) + len(d2) - 2:,}, p = {ps:.3e}")
    u, pu = stats.mannwhitneyu(d2, d1, alternative="greater")
    print(f"  Mann-Whitney U: U = {u:,.0f}, p = {pu:.3e}, P(day2 > day1) = {u / (len(d1) * len(d2)):.3f}")
    print(f"  Medians: day 2 = {np.median(d2):,.0f}, day 1 = {np.median(d1):,.0f}")
    print(f"  Permutation test (10,000 shuffles): p = {permutation_pvalue(d2, d1):.2e}")
    lo, hi = bootstrap_ci(d2, d1)
    print(f"  Bootstrap 95% CI for mean difference: [{lo:,.0f}, {hi:,.0f}]")
    print(f"  Levene (equal variances?): p = {stats.levene(d2, d1).pvalue:.2e}")


def test2_late_vs_early_mean(df, late_cutoff=20):
    header(f"TEST 2: Late (<= {late_cutoff} d) vs early (> 20 d) mean fare  |  two-sample Z, two-tailed")
    print("H0: mu(late) - mu(early) = 0    H1: mu(late) - mu(early) != 0")
    print("alpha = 0.05, smallest meaningful difference = Rs 500")
    late = df.loc[df.days_left <= late_cutoff, "price"].to_numpy(float)
    early = df.loc[df.days_left > 20, "price"].to_numpy(float)
    diff, se, dof, _ = welch(late, early)
    z = diff / se
    p = 2 * stats.norm.sf(abs(z))
    print(f"n(late) = {len(late):,}, n(early) = {len(early):,}; means = {late.mean():,.0f} vs {early.mean():,.0f}")
    print(f"Z = {z:.3f} (df not applicable; Welch df = {dof:,.0f}), p = {p:.3e} -> {decision(p)}")
    print(f"Estimate = Rs {diff:,.0f}, 95% CI [{diff - 1.96 * se:,.0f}, {diff + 1.96 * se:,.0f}], "
          f"Cohen's d = {cohen_d(late, early):.3f}, relative = {late.mean() / early.mean() - 1:+.1%}")
    print("Alternatives:")
    tw, pw = stats.ttest_ind(late, early, equal_var=False)
    print(f"  Welch t: t = {tw:.3f}, df = {dof:,.0f}, p = {pw:.3e}")
    u, pu = stats.mannwhitneyu(late, early)
    print(f"  Mann-Whitney U: p = {pu:.3e}, P(late > early) = {u / (len(late) * len(early)):.3f}; "
          f"medians {np.median(late):,.0f} vs {np.median(early):,.0f}")
    for cls in ["Economy", "Business"]:
        l = df.loc[(df.days_left <= late_cutoff) & (df["class"] == cls), "price"].to_numpy(float)
        e = df.loc[(df.days_left > 20) & (df["class"] == cls), "price"].to_numpy(float)
        d, s, _, _ = welch(l, e)
        print(f"  Within {cls}: diff = Rs {d:,.0f} [{d - 1.96 * s:,.0f}, {d + 1.96 * s:,.0f}], Z = {d / s:.2f}, "
              f"relative = {l.mean() / e.mean() - 1:+.1%}")
    l7 = df.loc[df.days_left <= 7, "price"].to_numpy(float)
    d, s, _, _ = welch(l7, early)
    print(f"  Sensitivity, late = last week (<= 7 d): diff = Rs {d:,.0f}, Z = {d / s:.2f}")


def test3_late_vs_early_variance(df, late_cutoff=20):
    header("TEST 3: Late vs early fare variance  |  F-test, right-tailed")
    print("H0: var(late) / var(early) <= 1    H1: var(late) / var(early) > 1")
    print("alpha = 0.05, smallest meaningful effect = SD ratio 1.10 (variance ratio 1.21)")
    late = df.loc[df.days_left <= late_cutoff, "price"].to_numpy(float)
    early = df.loc[df.days_left > 20, "price"].to_numpy(float)
    df1, df2 = len(late) - 1, len(early) - 1
    F = late.var(ddof=1) / early.var(ddof=1)
    p = stats.f.sf(F, df1, df2)
    ci = (F / stats.f.ppf(0.975, df1, df2), F / stats.f.ppf(0.025, df1, df2))
    print(f"SD late = {late.std(ddof=1):,.0f}, SD early = {early.std(ddof=1):,.0f}")
    print(f"F = {F:.4f}, df = ({df1:,}, {df2:,}), p = {p:.4f} -> {decision(p)}")
    print(f"Estimate = variance ratio {F:.4f}, 95% CI [{ci[0]:.4f}, {ci[1]:.4f}]")
    print("Alternatives:")
    print(f"  Bartlett: stat = {stats.bartlett(late, early).statistic:.2f}, p = {stats.bartlett(late, early).pvalue:.3e}")
    lev = stats.levene(late, early, center="mean")
    bf = stats.levene(late, early, center="median")
    print(f"  Levene (mean-centred): W = {lev.statistic:.2f}, df = (1, {len(late) + len(early) - 2:,}), p = {lev.pvalue:.3e}")
    print(f"  Brown-Forsythe (median-centred): W = {bf.statistic:.2f}, p = {bf.pvalue:.3e}")
    mad_l = np.mean(np.abs(late - np.median(late)))
    mad_e = np.mean(np.abs(early - np.median(early)))
    print(f"  Mean |x - median|: late = {mad_l:,.0f}, early = {mad_e:,.0f}")
    for cls in ["Economy", "Business"]:
        l = df.loc[(df.days_left <= late_cutoff) & (df["class"] == cls), "price"].to_numpy(float)
        e = df.loc[(df.days_left > 20) & (df["class"] == cls), "price"].to_numpy(float)
        Fc = l.var(ddof=1) / e.var(ddof=1)
        a, b = len(l) - 1, len(e) - 1
        print(f"  Within {cls}: SD {l.std(ddof=1):,.0f} vs {e.std(ddof=1):,.0f}, F = {Fc:.3f} "
              f"[{Fc / stats.f.ppf(0.975, a, b):.3f}, {Fc / stats.f.ppf(0.025, a, b):.3f}], "
              f"p = {stats.f.sf(Fc, a, b):.2e}, BF p = {stats.levene(l, e, center='median').pvalue:.2e}")


def ols(X, y):
    """OLS coefficients with classical and HC1 robust standard errors."""
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    se = np.sqrt(np.diag(XtX_inv) * (resid @ resid) / (n - k))
    meat = X.T @ (X * resid[:, None] ** 2)
    se_hc1 = np.sqrt(np.diag(XtX_inv @ meat @ XtX_inv) * n / (n - k))
    r2 = 1 - resid @ resid / np.sum((y - y.mean()) ** 2)
    return beta, se, se_hc1, r2, n - k


def test4_slope(df):
    header("TEST 4: Slope of price on days_left  |  t-test on OLS slope, two-tailed")
    print("H0: beta1 = 0    H1: beta1 != 0")
    print("alpha = 0.05, smallest meaningful slope = Rs 100 per day")
    x, y = df.days_left.to_numpy(float), df.price.to_numpy(float)
    X = np.column_stack([np.ones_like(x), x])
    beta, se, se_hc1, r2, dof = ols(X, y)
    t = beta[1] / se[1]
    p = 2 * stats.t.sf(abs(t), dof)
    h = stats.t.ppf(0.975, dof) * se[1]
    print(f"beta1 = {beta[1]:.2f} Rs/day, SE = {se[1]:.3f}, t = {t:.3f}, df = {dof:,}, p = {p:.3e} -> {decision(p)}")
    print(f"95% CI [{beta[1] - h:.2f}, {beta[1] + h:.2f}], R^2 = {r2:.4f}, intercept = {beta[0]:,.0f}")
    print("Alternatives:")
    print(f"  HC1 robust SE = {se_hc1[1]:.3f}, robust t = {beta[1] / se_hc1[1]:.2f}")
    rho = stats.spearmanr(x, y)
    print(f"  Spearman rho = {rho.statistic:.4f}, p = {rho.pvalue:.3e}")
    tau = stats.kendalltau(x, y)
    print(f"  Kendall tau-b = {tau.statistic:.4f}, p = {tau.pvalue:.3e}")
    business = (df["class"] == "Business").to_numpy(float)
    b2, s2, s2r, r2b, d2 = ols(np.column_stack([np.ones_like(x), x, business]), y)
    print(f"  Controlling for class: beta1 = {b2[1]:.2f}, SE = {s2[1]:.3f}, t = {b2[1] / s2[1]:.2f}, "
          f"df = {d2:,}, R^2 = {r2b:.4f}")
    eco = df[df["class"] == "Economy"]
    xe, ye = eco.days_left.to_numpy(float), eco.price.to_numpy(float)
    be, sen, _, r2e, de = ols(np.column_stack([np.ones_like(xe), xe]), ye)
    print(f"  Economy only: beta1 = {be[1]:.2f}, t = {be[1] / sen[1]:.2f}, df = {de:,}, R^2 = {r2e:.4f}")
    for lo, hi in [(1, 20), (21, 49)]:
        s = df[df.days_left.between(lo, hi)]
        r = stats.linregress(s.days_left, s.price)
        print(f"  Piecewise days {lo}-{hi}: slope = {r.slope:.1f} Rs/day "
              f"[{r.slope - 1.96 * r.stderr:.1f}, {r.slope + 1.96 * r.stderr:.1f}]")


def test5_class_vs_timing(df):
    header("TEST 5: Booking timing vs travel class  |  chi-square test of independence")
    print("H0: booking category (last-minute <= 7 d / standard) is independent of class    H1: associated")
    print("alpha = 0.05, smallest meaningful effect = 5 percentage points in Business share")
    cat = np.where(df.days_left <= 7, "Last_Minute", "Standard")
    tab = pd.crosstab(cat, df["class"])
    print(tab)
    chi2, p, dof, expected = stats.chi2_contingency(tab, correction=False)
    n = tab.to_numpy().sum()
    print(f"Min expected count = {expected.min():,.0f}")
    print(f"chi2 = {chi2:.3f}, df = {dof}, p = {p:.3e} -> {decision(p)}, Cramer's V = {np.sqrt(chi2 / n):.4f}")
    n1, n2 = tab.loc["Last_Minute"].sum(), tab.loc["Standard"].sum()
    p1, p2 = tab.loc["Last_Minute", "Business"] / n1, tab.loc["Standard", "Business"] / n2
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    d = p1 - p2
    print(f"Business share: last-minute {p1:.4f}, standard {p2:.4f}; "
          f"difference = {d * 100:.2f} pp, 95% CI [{(d - 1.96 * se) * 100:.2f}, {(d + 1.96 * se) * 100:.2f}] pp")
    print("Alternatives:")
    chi2y, py, _, _ = stats.chi2_contingency(tab, correction=True)
    print(f"  Yates-corrected chi2 = {chi2y:.3f}, p = {py:.3e}")
    g, pg, _, _ = stats.chi2_contingency(tab, lambda_="log-likelihood", correction=False)
    print(f"  G-test (likelihood ratio): G = {g:.3f}, p = {pg:.3e}")
    pp = (tab["Business"].sum()) / n
    z = d / np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    print(f"  Two-proportion Z: Z = {z:.3f} (Z^2 = {z * z:.3f}), p = {2 * stats.norm.sf(abs(z)):.3e}")
    odds, pf = stats.fisher_exact(tab.to_numpy())
    print(f"  Fisher exact: odds ratio = {odds:.4f}, p = {pf:.3e}")


def run_all(df):
    print(f"Rows tested: {len(df):,}; alpha = {ALPHA}; Bonferroni alpha = {ALPHA / N_TESTS}")
    test1_last_day_drop(df)
    test2_late_vs_early_mean(df)
    test3_late_vs_early_variance(df)
    test4_slope(df)
    test5_class_vs_timing(df)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "Clean_Dataset.csv"
    run_all(clean(pd.read_csv(path)))
