#![allow(non_snake_case)]
// Statistical hypothesis tests and distribution fitting
// Shapiro-Wilk, Anderson-Darling, one-sample Kolmogorov-Smirnov, MLE
// distribution fitting, local outlier factor, group-sequential and
// Bayesian A/B test analysis
// Shared special functions (normal CDF and quantile, log gamma, digamma,
// regularized incomplete gamma, Kolmogorov tail) live here and are reused
// by the causal and anomaly modules

use zyron_common::Xoshiro256pp;
use zyron_common::error::{Result, ZyronError};

// ===== Special functions =====

fn normalPdf(x: f64) -> f64 {
    (-(x * x) / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Standard normal CDF, Abramowitz and Stegun 26.2.17
/// Absolute error below 7.5e-8 over the whole line
pub fn normalCdf(x: f64) -> f64 {
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.2316419 * ax);
    let poly = t
        * (0.319381530
            + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    let tail = normalPdf(ax) * poly;
    if x >= 0.0 { 1.0 - tail } else { tail }
}

/// Inverse standard normal CDF, Acklam rational approximation
/// Relative error below 1.2e-9 for p in (0, 1). Inputs are clamped to a
/// representable open interval so the tails return large finite values
pub fn normalQuantile(p: f64) -> f64 {
    let p = p.clamp(1e-300, 1.0 - 1e-16);
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    const P_LOW: f64 = 0.02425;
    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= 1.0 - P_LOW {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// Natural log of the gamma function, Lanczos approximation with g = 7
pub fn lnGamma(x: f64) -> f64 {
    const COEFS: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    if x < 0.5 {
        // reflection formula keeps the approximation in its valid range
        return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - lnGamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut acc = COEFS[0];
    for (i, &c) in COEFS.iter().enumerate().skip(1) {
        acc += c / (x + i as f64);
    }
    let t = x + 7.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + acc.ln()
}

/// Digamma function via recurrence into the asymptotic range
pub fn digamma(x: f64) -> f64 {
    let mut x = x;
    let mut result = 0.0f64;
    while x < 6.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    result + x.ln() - 0.5 * inv - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 / 252.0))
}

/// Trigamma function via recurrence into the asymptotic range
pub fn trigamma(x: f64) -> f64 {
    let mut x = x;
    let mut result = 0.0f64;
    while x < 6.0 {
        result += 1.0 / (x * x);
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    result + inv * (1.0 + inv * (0.5 + inv * (1.0 / 6.0 - inv2 * (1.0 / 30.0 - inv2 / 42.0))))
}

/// Regularized upper incomplete gamma Q(s, x) for s > 0, x >= 0
/// Small x uses the lower series and complements, large x uses the
/// Lentz modified continued fraction
pub fn regularizedUpperGamma(s: f64, x: f64) -> f64 {
    if s <= 0.0 || x < 0.0 || !s.is_finite() || !x.is_finite() {
        return f64::NAN;
    }
    if x == 0.0 {
        return 1.0;
    }
    if x < s + 1.0 {
        let mut term = 1.0 / s;
        let mut sum = term;
        let mut a = s;
        for _ in 0..500 {
            a += 1.0;
            term *= x / a;
            sum += term;
            if term.abs() < sum.abs() * 1e-15 {
                break;
            }
        }
        let p = (sum.ln() + s * x.ln() - x - lnGamma(s)).exp();
        (1.0 - p).clamp(0.0, 1.0)
    } else {
        const FPMIN: f64 = 1e-300;
        let mut b = x + 1.0 - s;
        let mut c = 1.0 / FPMIN;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..500 {
            let an = -(i as f64) * (i as f64 - s);
            b += 2.0;
            d = an * d + b;
            if d.abs() < FPMIN {
                d = FPMIN;
            }
            c = b + an / c;
            if c.abs() < FPMIN {
                c = FPMIN;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-15 {
                break;
            }
        }
        ((s * x.ln() - x - lnGamma(s)).exp() * h).clamp(0.0, 1.0)
    }
}

/// Kolmogorov distribution upper tail Q(lambda) = 2 sum (-1)^(j-1) exp(-2 j^2 lambda^2)
pub fn kolmogorovQ(lambda: f64) -> f64 {
    if lambda <= 0.0 {
        return 1.0;
    }
    let mut sum = 0.0f64;
    let mut sign = 1.0f64;
    for j in 1..=100u32 {
        let jf = j as f64;
        let term = (-2.0 * jf * jf * lambda * lambda).exp();
        sum += sign * term;
        if term < 1e-12 {
            break;
        }
        sign = -sign;
    }
    (2.0 * sum).clamp(0.0, 1.0)
}

fn sortAscending(values: &mut [f64]) {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
}

fn finiteSorted(values: &[f64]) -> Vec<f64> {
    let mut v: Vec<f64> = values.iter().copied().filter(|x| x.is_finite()).collect();
    sortAscending(&mut v);
    v
}

// ===== Shapiro-Wilk =====

/// Shapiro-Wilk normality test, Royston 1995 AS R94 approximation
/// Valid for 3 to 5000 finite samples. Returns (W, p value)
pub fn shapiroWilk(values: &[f64]) -> Result<(f64, f64)> {
    let x = finiteSorted(values);
    let n = x.len();
    if !(3..=5000).contains(&n) {
        return Err(ZyronError::InvalidParameter {
            name: "values".to_string(),
            value: format!("shapiro_wilk needs 3 to 5000 finite samples, got {}", n),
        });
    }
    if x[n - 1] - x[0] <= 0.0 {
        return Err(ZyronError::InvalidParameter {
            name: "values".to_string(),
            value: "all samples are identical, W is undefined".to_string(),
        });
    }
    let nf = n as f64;
    // Blom scores for the expected normal order statistics
    let mut m = vec![0.0f64; n];
    for (i, slot) in m.iter_mut().enumerate() {
        *slot = normalQuantile(((i + 1) as f64 - 0.375) / (nf + 0.25));
    }
    let ssumm2: f64 = m.iter().map(|v| v * v).sum();
    let rsn = 1.0 / nf.sqrt();
    let poly = |c5: f64, c4: f64, c3: f64, c2: f64, c1: f64| {
        ((((c5 * rsn + c4) * rsn + c3) * rsn + c2) * rsn + c1) * rsn
    };
    let a_n = poly(-2.706056, 4.434685, -2.071190, -0.147981, 0.221157) + m[n - 1] / ssumm2.sqrt();
    let mut w_coef = vec![0.0f64; n];
    if n > 5 {
        let a_n1 =
            poly(-3.582633, 5.682633, -1.752461, -0.293762, 0.042981) + m[n - 2] / ssumm2.sqrt();
        let phi = (ssumm2 - 2.0 * m[n - 1] * m[n - 1] - 2.0 * m[n - 2] * m[n - 2])
            / (1.0 - 2.0 * a_n * a_n - 2.0 * a_n1 * a_n1);
        let sf = if phi > 0.0 { phi.sqrt() } else { f64::INFINITY };
        w_coef[n - 1] = a_n;
        w_coef[0] = -a_n;
        w_coef[n - 2] = a_n1;
        w_coef[1] = -a_n1;
        for i in 2..n - 2 {
            w_coef[i] = m[i] / sf;
        }
    } else {
        let phi = (ssumm2 - 2.0 * m[n - 1] * m[n - 1]) / (1.0 - 2.0 * a_n * a_n);
        let sf = if phi > 0.0 { phi.sqrt() } else { f64::INFINITY };
        w_coef[n - 1] = a_n;
        w_coef[0] = -a_n;
        for i in 1..n - 1 {
            w_coef[i] = m[i] / sf;
        }
    }
    let mean = x.iter().sum::<f64>() / nf;
    let num: f64 = w_coef.iter().zip(x.iter()).map(|(w, v)| w * v).sum();
    let den: f64 = x.iter().map(|v| (v - mean) * (v - mean)).sum();
    if den <= 0.0 {
        return Err(ZyronError::InvalidParameter {
            name: "values".to_string(),
            value: "zero variance sample".to_string(),
        });
    }
    let w = (num * num / den).clamp(0.0, 1.0);

    // p value per Royston 1995
    if n == 3 {
        let p = 6.0 / std::f64::consts::PI * (w.sqrt().asin() - 0.75f64.sqrt().asin());
        return Ok((w, p.clamp(0.0, 1.0)));
    }
    let one_minus_w = 1.0 - w;
    if one_minus_w <= 0.0 {
        return Ok((w, 1.0));
    }
    let z = if n <= 11 {
        let gamma = -2.273 + 0.459 * nf;
        let inner = gamma - one_minus_w.ln();
        if inner <= 0.0 {
            return Ok((w, 0.0));
        }
        let wln = -inner.ln();
        let mu = 0.5440 - 0.39978 * nf + 0.025054 * nf * nf - 0.0006714 * nf * nf * nf;
        let sigma = (1.3822 - 0.77857 * nf + 0.062767 * nf * nf - 0.0020322 * nf * nf * nf).exp();
        (wln - mu) / sigma
    } else {
        let lnn = nf.ln();
        let wln = one_minus_w.ln();
        let mu = -1.5861 - 0.31082 * lnn - 0.083751 * lnn * lnn + 0.0038915 * lnn * lnn * lnn;
        let sigma = (-0.4803 - 0.082676 * lnn + 0.0030302 * lnn * lnn).exp();
        (wln - mu) / sigma
    };
    Ok((w, (1.0 - normalCdf(z)).clamp(0.0, 1.0)))
}

// ===== Anderson-Darling =====

#[derive(Debug, Clone)]
pub struct AndersonDarlingResult {
    // Statistic already adjusted for estimated parameters
    pub a2: f64,
    pub criticalValues: [f64; 5],
    // Percent levels matching criticalValues element for element
    pub significanceLevels: [f64; 5],
}

/// Anderson-Darling goodness of fit with parameters estimated from the
/// sample (case 3). Supported distributions are normal, exp, logistic
pub fn andersonDarling(values: &[f64], dist: &str) -> Result<AndersonDarlingResult> {
    let x = finiteSorted(values);
    let n = x.len();
    if n < 8 {
        return Err(ZyronError::InvalidParameter {
            name: "values".to_string(),
            value: format!(
                "anderson_darling needs at least 8 finite samples, got {}",
                n
            ),
        });
    }
    let nf = n as f64;
    let mean = x.iter().sum::<f64>() / nf;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (nf - 1.0);
    let lower = dist.to_ascii_lowercase();
    let (cdf, factor, crit): (Box<dyn Fn(f64) -> f64>, f64, [f64; 5]) = match lower.as_str() {
        "normal" => {
            let sd = var.sqrt();
            if sd <= 0.0 {
                return Err(ZyronError::InvalidParameter {
                    name: "values".to_string(),
                    value: "zero variance sample".to_string(),
                });
            }
            (
                Box::new(move |v: f64| normalCdf((v - mean) / sd)),
                1.0 + 4.0 / nf - 25.0 / (nf * nf),
                [0.576, 0.656, 0.787, 0.918, 1.092],
            )
        }
        "exp" | "exponential" => {
            if x[0] < 0.0 || mean <= 0.0 {
                return Err(ZyronError::InvalidParameter {
                    name: "values".to_string(),
                    value: "exponential fit needs nonnegative samples with positive mean"
                        .to_string(),
                });
            }
            let lambda = 1.0 / mean;
            (
                Box::new(move |v: f64| 1.0 - (-lambda * v.max(0.0)).exp()),
                1.0 + 0.6 / nf,
                [0.922, 1.078, 1.341, 1.606, 1.957],
            )
        }
        "logistic" => {
            if var <= 0.0 {
                return Err(ZyronError::InvalidParameter {
                    name: "values".to_string(),
                    value: "zero variance sample".to_string(),
                });
            }
            let scale = (3.0 * var).sqrt() / std::f64::consts::PI;
            // 15 percent point interpolated between the published 25 and 10
            // percent points of the Stephens logistic table
            (
                Box::new(move |v: f64| 1.0 / (1.0 + (-(v - mean) / scale).exp())),
                1.0 + 0.25 / nf,
                [0.517, 0.563, 0.660, 0.769, 0.906],
            )
        }
        other => {
            return Err(ZyronError::InvalidParameter {
                name: "dist".to_string(),
                value: format!(
                    "'{}' is not supported, supported distributions are normal, exp, logistic",
                    other
                ),
            });
        }
    };
    let mut a2 = -nf;
    for i in 0..n {
        let f_lo = cdf(x[i]).clamp(1e-12, 1.0 - 1e-12);
        let f_hi = cdf(x[n - 1 - i]).clamp(1e-12, 1.0 - 1e-12);
        a2 -= (2.0 * i as f64 + 1.0) / nf * (f_lo.ln() + (1.0 - f_hi).ln());
    }
    Ok(AndersonDarlingResult {
        a2: a2 * factor,
        criticalValues: crit,
        significanceLevels: [15.0, 10.0, 5.0, 2.5, 1.0],
    })
}

// ===== One-sample Kolmogorov-Smirnov =====

/// One-sample KS test against a fully specified distribution
/// Supported dists with their params
///   normal (mean, std), exp (lambda), uniform (a, b), lognormal (mu, sigma)
/// Returns (D, p value) with the asymptotic Kolmogorov series and the
/// sqrt(n) small-sample correction
pub fn ks1Sample(values: &[f64], dist: &str, params: &[f64]) -> Result<(f64, f64)> {
    let x = finiteSorted(values);
    let n = x.len();
    if n == 0 {
        return Err(ZyronError::InvalidParameter {
            name: "values".to_string(),
            value: "empty sample".to_string(),
        });
    }
    let bad_params = |want: &str| ZyronError::InvalidParameter {
        name: "params".to_string(),
        value: format!("{} distribution expects {}", dist, want),
    };
    let lower = dist.to_ascii_lowercase();
    let cdf: Box<dyn Fn(f64) -> f64> = match lower.as_str() {
        "normal" => {
            let (mean, std) = match params {
                [m, s] if *s > 0.0 => (*m, *s),
                _ => return Err(bad_params("(mean, std) with std > 0")),
            };
            Box::new(move |v: f64| normalCdf((v - mean) / std))
        }
        "exp" | "exponential" => {
            let lambda = match params {
                [l] if *l > 0.0 => *l,
                _ => return Err(bad_params("(lambda) with lambda > 0")),
            };
            Box::new(move |v: f64| {
                if v <= 0.0 {
                    0.0
                } else {
                    1.0 - (-lambda * v).exp()
                }
            })
        }
        "uniform" => {
            let (a, b) = match params {
                [a, b] if *b > *a => (*a, *b),
                _ => return Err(bad_params("(a, b) with b > a")),
            };
            Box::new(move |v: f64| ((v - a) / (b - a)).clamp(0.0, 1.0))
        }
        "lognormal" => {
            let (mu, sigma) = match params {
                [m, s] if *s > 0.0 => (*m, *s),
                _ => return Err(bad_params("(mu, sigma) with sigma > 0")),
            };
            Box::new(move |v: f64| {
                if v <= 0.0 {
                    0.0
                } else {
                    normalCdf((v.ln() - mu) / sigma)
                }
            })
        }
        other => {
            return Err(ZyronError::InvalidParameter {
                name: "dist".to_string(),
                value: format!(
                    "'{}' is not supported, supported distributions are normal, exp, uniform, lognormal",
                    other
                ),
            });
        }
    };
    let nf = n as f64;
    let mut d = 0.0f64;
    for i in 0..n {
        let f = cdf(x[i]);
        let d_plus = (i + 1) as f64 / nf - f;
        let d_minus = f - i as f64 / nf;
        d = d.max(d_plus).max(d_minus);
    }
    let en = nf.sqrt();
    let lambda = (en + 0.12 + 0.11 / en) * d;
    Ok((d, kolmogorovQ(lambda)))
}

// ===== Distribution fitting =====

#[derive(Debug, Clone)]
pub struct DistributionFit {
    pub name: String,
    pub params: Vec<f64>,
    pub logLikelihood: f64,
    pub aic: f64,
}

const FIT_CANDIDATES: [&str; 5] = ["normal", "lognormal", "exp", "weibull", "gamma"];

/// Fits candidate distributions by maximum likelihood and returns them
/// best first by AIC. An empty candidate list fits every supported
/// distribution. Candidates the data cannot support (for example
/// lognormal on nonpositive samples) are skipped
pub fn fitDistribution(values: &[f64], candidates: &[String]) -> Result<Vec<DistributionFit>> {
    let x: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    let n = x.len();
    if n < 3 {
        return Err(ZyronError::InvalidParameter {
            name: "values".to_string(),
            value: format!(
                "fit_distribution needs at least 3 finite samples, got {}",
                n
            ),
        });
    }
    let wanted: Vec<String> = if candidates.is_empty() {
        FIT_CANDIDATES.iter().map(|s| s.to_string()).collect()
    } else {
        let mut out = Vec::with_capacity(candidates.len());
        for c in candidates {
            let lower = c.to_ascii_lowercase();
            let canonical = if lower == "exponential" {
                "exp"
            } else {
                lower.as_str()
            };
            if !FIT_CANDIDATES.contains(&canonical) {
                return Err(ZyronError::InvalidParameter {
                    name: "candidates".to_string(),
                    value: format!(
                        "'{}' is not supported, supported candidates are normal, lognormal, exp, weibull, gamma",
                        c
                    ),
                });
            }
            out.push(canonical.to_string());
        }
        out
    };
    let nf = n as f64;
    let mean = x.iter().sum::<f64>() / nf;
    let all_positive = x.iter().all(|v| *v > 0.0);
    let all_nonneg = x.iter().all(|v| *v >= 0.0);
    let mut fits: Vec<DistributionFit> = Vec::new();
    for name in &wanted {
        let fit = match name.as_str() {
            "normal" => fitNormal(&x, mean),
            "lognormal" if all_positive => fitLognormal(&x),
            "exp" if all_nonneg && mean > 0.0 => Some(fitExponential(&x, mean)),
            "weibull" if all_positive => fitWeibull(&x, mean),
            "gamma" if all_positive => fitGamma(&x, mean),
            _ => None,
        };
        if let Some(f) = fit {
            fits.push(f);
        }
    }
    if fits.is_empty() {
        return Err(ZyronError::InvalidParameter {
            name: "candidates".to_string(),
            value: "no candidate distribution is compatible with the sample".to_string(),
        });
    }
    fits.sort_by(|a, b| {
        a.aic
            .partial_cmp(&b.aic)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(fits)
}

fn fitNormal(x: &[f64], mean: f64) -> Option<DistributionFit> {
    let nf = x.len() as f64;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / nf;
    if var <= 0.0 {
        return None;
    }
    let ll = -0.5 * nf * ((2.0 * std::f64::consts::PI * var).ln() + 1.0);
    Some(DistributionFit {
        name: "normal".to_string(),
        params: vec![mean, var.sqrt()],
        logLikelihood: ll,
        aic: 4.0 - 2.0 * ll,
    })
}

fn fitLognormal(x: &[f64]) -> Option<DistributionFit> {
    let nf = x.len() as f64;
    let logs: Vec<f64> = x.iter().map(|v| v.ln()).collect();
    let mu = logs.iter().sum::<f64>() / nf;
    let var = logs.iter().map(|v| (v - mu) * (v - mu)).sum::<f64>() / nf;
    if var <= 0.0 {
        return None;
    }
    let sum_log: f64 = logs.iter().sum();
    let ll = -0.5 * nf * ((2.0 * std::f64::consts::PI * var).ln() + 1.0) - sum_log;
    Some(DistributionFit {
        name: "lognormal".to_string(),
        params: vec![mu, var.sqrt()],
        logLikelihood: ll,
        aic: 4.0 - 2.0 * ll,
    })
}

fn fitExponential(x: &[f64], mean: f64) -> DistributionFit {
    let nf = x.len() as f64;
    let lambda = 1.0 / mean;
    let ll = nf * lambda.ln() - lambda * x.iter().sum::<f64>();
    DistributionFit {
        name: "exp".to_string(),
        params: vec![lambda],
        logLikelihood: ll,
        aic: 2.0 - 2.0 * ll,
    }
}

fn fitWeibull(x: &[f64], mean: f64) -> Option<DistributionFit> {
    let nf = x.len() as f64;
    let mean_log = x.iter().map(|v| v.ln()).sum::<f64>() / nf;
    // Newton iteration on g(k) = sum(x^k ln x)/sum(x^k) - 1/k - meanLog
    let mut k = 1.0f64;
    if mean <= 0.0 {
        return None;
    }
    for _ in 0..100 {
        let mut a = 0.0f64;
        let mut b = 0.0f64;
        let mut a_prime = 0.0f64;
        for &v in x {
            let xk = v.powf(k);
            let lx = v.ln();
            a += xk * lx;
            b += xk;
            a_prime += xk * lx * lx;
        }
        if b <= 0.0 {
            return None;
        }
        let g = a / b - 1.0 / k - mean_log;
        let g_prime = (a_prime * b - a * a) / (b * b) + 1.0 / (k * k);
        if g_prime.abs() < 1e-300 {
            break;
        }
        let step = g / g_prime;
        let next = (k - step).max(1e-6);
        if (next - k).abs() < 1e-10 {
            k = next;
            break;
        }
        k = next;
        if !k.is_finite() {
            return None;
        }
    }
    let scale = (x.iter().map(|v| v.powf(k)).sum::<f64>() / nf).powf(1.0 / k);
    if !scale.is_finite() || scale <= 0.0 {
        return None;
    }
    let sum_log: f64 = x.iter().map(|v| v.ln()).sum();
    let sum_pow: f64 = x.iter().map(|v| (v / scale).powf(k)).sum();
    let ll = nf * k.ln() - nf * k * scale.ln() + (k - 1.0) * sum_log - sum_pow;
    if !ll.is_finite() {
        return None;
    }
    Some(DistributionFit {
        name: "weibull".to_string(),
        params: vec![k, scale],
        logLikelihood: ll,
        aic: 4.0 - 2.0 * ll,
    })
}

fn fitGamma(x: &[f64], mean: f64) -> Option<DistributionFit> {
    let nf = x.len() as f64;
    let mean_log = x.iter().map(|v| v.ln()).sum::<f64>() / nf;
    let s = mean.ln() - mean_log;
    if s <= 0.0 || !s.is_finite() {
        return None;
    }
    // Newton iteration on f(k) = ln(k) - digamma(k) - s
    let mut k = (3.0 - s + ((s - 3.0) * (s - 3.0) + 24.0 * s).sqrt()) / (12.0 * s);
    if !k.is_finite() || k <= 0.0 {
        k = 1.0;
    }
    for _ in 0..100 {
        let f = k.ln() - digamma(k) - s;
        let f_prime = 1.0 / k - trigamma(k);
        if f_prime.abs() < 1e-300 {
            break;
        }
        let next = (k - f / f_prime).max(1e-8);
        if (next - k).abs() < 1e-12 {
            k = next;
            break;
        }
        k = next;
    }
    let theta = mean / k;
    if !theta.is_finite() || theta <= 0.0 {
        return None;
    }
    let sum_log: f64 = x.iter().map(|v| v.ln()).sum();
    let sum_x: f64 = x.iter().sum();
    let ll = (k - 1.0) * sum_log - sum_x / theta - nf * k * theta.ln() - nf * lnGamma(k);
    if !ll.is_finite() {
        return None;
    }
    Some(DistributionFit {
        name: "gamma".to_string(),
        params: vec![k, theta],
        logLikelihood: ll,
        aic: 4.0 - 2.0 * ll,
    })
}

// ===== Local outlier factor =====

/// Classic LOF with k-distance and reachability distance
/// Returns one LOF score per input row, values near 1 are inliers
pub fn localOutlierFactor(rows: &[Vec<f64>], k: usize) -> Result<Vec<f64>> {
    let n = rows.len();
    if n < 2 {
        return Err(ZyronError::InvalidParameter {
            name: "rows".to_string(),
            value: format!("local_outlier_factor needs at least 2 rows, got {}", n),
        });
    }
    if k == 0 || k >= n {
        return Err(ZyronError::InvalidParameter {
            name: "k".to_string(),
            value: format!("k must be in 1..={}, got {}", n - 1, k),
        });
    }
    let p = rows[0].len();
    for r in rows {
        if r.len() != p {
            return Err(ZyronError::InvalidParameter {
                name: "rows".to_string(),
                value: "rows have inconsistent dimensions".to_string(),
            });
        }
        if r.iter().any(|v| !v.is_finite()) {
            return Err(ZyronError::InvalidParameter {
                name: "rows".to_string(),
                value: "rows contain non finite values".to_string(),
            });
        }
    }
    let dist = |a: &[f64], b: &[f64]| -> f64 {
        let mut s = 0.0f64;
        for j in 0..p {
            let d = a[j] - b[j];
            s += d * d;
        }
        s.sqrt()
    };
    // k nearest neighbors and k-distance per point
    let mut neighbors: Vec<Vec<(f64, usize)>> = Vec::with_capacity(n);
    let mut k_distance = vec![0.0f64; n];
    for i in 0..n {
        let mut ds: Vec<(f64, usize)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (dist(&rows[i], &rows[j]), j))
            .collect();
        ds.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        ds.truncate(k);
        k_distance[i] = ds.last().map(|(d, _)| *d).unwrap_or(0.0);
        neighbors.push(ds);
    }
    // local reachability density
    const LRD_CAP: f64 = 1e12;
    let mut lrd = vec![0.0f64; n];
    for i in 0..n {
        let mut reach_sum = 0.0f64;
        for &(d, j) in &neighbors[i] {
            reach_sum += d.max(k_distance[j]);
        }
        let count = neighbors[i].len() as f64;
        lrd[i] = if reach_sum > 0.0 {
            (count / reach_sum).min(LRD_CAP)
        } else {
            // duplicated points, density capped so ratios stay near 1
            LRD_CAP
        };
    }
    let mut out = vec![0.0f64; n];
    for i in 0..n {
        let mut ratio_sum = 0.0f64;
        for &(_, j) in &neighbors[i] {
            ratio_sum += lrd[j] / lrd[i];
        }
        out[i] = ratio_sum / neighbors[i].len() as f64;
    }
    Ok(out)
}

// ===== Group-sequential analysis =====

#[derive(Debug, Clone)]
pub struct SequentialLook {
    pub look: usize,
    pub z: f64,
    pub boundary: f64,
    pub crossed: bool,
}

/// Group-sequential two-proportion test over cumulative looks
/// Inputs are cumulative successes and trials per look for two variants
/// Boundaries come from the chosen alpha spending function using the
/// incremental spend per look (Bonferroni approximation to Lan-DeMets,
/// which ignores look correlation and is conservative)
pub fn sequentialAnalyze(
    successesA: &[f64],
    trialsA: &[f64],
    successesB: &[f64],
    trialsB: &[f64],
    alphaSpending: &str,
    alpha: f64,
) -> Result<Vec<SequentialLook>> {
    let k = successesA.len();
    if k == 0 || trialsA.len() != k || successesB.len() != k || trialsB.len() != k {
        return Err(ZyronError::InvalidParameter {
            name: "looks".to_string(),
            value: "look arrays must be nonempty and equal length".to_string(),
        });
    }
    if !(alpha > 0.0 && alpha < 1.0) {
        return Err(ZyronError::InvalidParameter {
            name: "alpha".to_string(),
            value: format!("alpha must be in (0, 1), got {}", alpha),
        });
    }
    let spending = alphaSpending.to_ascii_lowercase();
    if spending != "obrien_fleming" && spending != "pocock" {
        return Err(ZyronError::InvalidParameter {
            name: "alpha_spending".to_string(),
            value: format!(
                "'{}' is not supported, supported spending functions are obrien_fleming, pocock",
                alphaSpending
            ),
        });
    }
    for i in 0..k {
        let ok = successesA[i] >= 0.0
            && successesB[i] >= 0.0
            && trialsA[i] >= successesA[i]
            && trialsB[i] >= successesB[i];
        let cumulative = i == 0 || (trialsA[i] >= trialsA[i - 1] && trialsB[i] >= trialsB[i - 1]);
        if !ok || !cumulative {
            return Err(ZyronError::InvalidParameter {
                name: "looks".to_string(),
                value: format!(
                    "look {} must carry cumulative counts with successes <= trials",
                    i + 1
                ),
            });
        }
    }
    let total_info = trialsA[k - 1] + trialsB[k - 1];
    if total_info <= 0.0 {
        return Err(ZyronError::InvalidParameter {
            name: "looks".to_string(),
            value: "final look has zero trials".to_string(),
        });
    }
    let z_half = normalQuantile(1.0 - alpha / 2.0);
    let spend_at = |t: f64| -> f64 {
        match spending.as_str() {
            "obrien_fleming" => 2.0 * (1.0 - normalCdf(z_half / t.sqrt())),
            _ => alpha * (1.0 + (std::f64::consts::E - 1.0) * t).ln(),
        }
    };
    let mut out = Vec::with_capacity(k);
    let mut spent_prev = 0.0f64;
    for i in 0..k {
        let t = ((trialsA[i] + trialsB[i]) / total_info).clamp(1e-9, 1.0);
        let spent = spend_at(t).min(alpha);
        let delta = (spent - spent_prev).max(1e-10);
        spent_prev = spent;
        let boundary = normalQuantile(1.0 - delta / 2.0);
        // pooled two-proportion z at this look
        let na = trialsA[i];
        let nb = trialsB[i];
        let z = if na > 0.0 && nb > 0.0 {
            let pa = successesA[i] / na;
            let pb = successesB[i] / nb;
            let pooled = (successesA[i] + successesB[i]) / (na + nb);
            let se = (pooled * (1.0 - pooled) * (1.0 / na + 1.0 / nb)).sqrt();
            if se > 0.0 { (pb - pa) / se } else { 0.0 }
        } else {
            0.0
        };
        out.push(SequentialLook {
            look: i + 1,
            z,
            boundary,
            crossed: z.abs() > boundary,
        });
    }
    Ok(out)
}

// ===== Bayesian A/B analysis =====

#[derive(Debug, Clone)]
pub struct BayesianAbResult {
    pub probBBeatsA: f64,
    // expected drop in conversion when choosing A and B is truly better
    pub expectedLossA: f64,
    // expected drop in conversion when choosing B and A is truly better
    pub expectedLossB: f64,
    pub meanA: f64,
    pub meanB: f64,
}

const BAYESIAN_DRAWS: usize = 100_000;
const BAYESIAN_SEED: u64 = 0x5EED_AB01;

/// Beta posterior Monte Carlo comparison of two binomial variants
/// priorA and priorB are (alpha, beta) pairs of the Beta priors
pub fn bayesianAnalyze(
    successesA: u64,
    trialsA: u64,
    successesB: u64,
    trialsB: u64,
    priorA: (f64, f64),
    priorB: (f64, f64),
) -> Result<BayesianAbResult> {
    if successesA > trialsA || successesB > trialsB {
        return Err(ZyronError::InvalidParameter {
            name: "successes".to_string(),
            value: "successes cannot exceed trials".to_string(),
        });
    }
    for (label, (a, b)) in [("prior_a", priorA), ("prior_b", priorB)] {
        if !(a > 0.0 && b > 0.0 && a.is_finite() && b.is_finite()) {
            return Err(ZyronError::InvalidParameter {
                name: label.to_string(),
                value: format!("Beta prior parameters must be positive, got ({}, {})", a, b),
            });
        }
    }
    let alpha_a = priorA.0 + successesA as f64;
    let beta_a = priorA.1 + (trialsA - successesA) as f64;
    let alpha_b = priorB.0 + successesB as f64;
    let beta_b = priorB.1 + (trialsB - successesB) as f64;
    let mut rng = Xoshiro256pp::fromSeed(BAYESIAN_SEED);
    let mut wins_b = 0u64;
    let mut loss_a = 0.0f64;
    let mut loss_b = 0.0f64;
    for _ in 0..BAYESIAN_DRAWS {
        let pa = betaSample(&mut rng, alpha_a, beta_a);
        let pb = betaSample(&mut rng, alpha_b, beta_b);
        if pb > pa {
            wins_b += 1;
            loss_a += pb - pa;
        } else {
            loss_b += pa - pb;
        }
    }
    let draws = BAYESIAN_DRAWS as f64;
    Ok(BayesianAbResult {
        probBBeatsA: wins_b as f64 / draws,
        expectedLossA: loss_a / draws,
        expectedLossB: loss_b / draws,
        meanA: alpha_a / (alpha_a + beta_a),
        meanB: alpha_b / (alpha_b + beta_b),
    })
}

fn betaSample(rng: &mut Xoshiro256pp, alpha: f64, beta: f64) -> f64 {
    let x = gammaSample(rng, alpha);
    let y = gammaSample(rng, beta);
    let total = x + y;
    if total > 0.0 { x / total } else { 0.5 }
}

// Marsaglia-Tsang gamma sampler with the shape < 1 boost
fn gammaSample(rng: &mut Xoshiro256pp, shape: f64) -> f64 {
    if shape < 1.0 {
        let u = rng.nextF64().max(1e-300);
        return gammaSample(rng, shape + 1.0) * u.powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (3.0 * d.sqrt());
    loop {
        let x = rng.nextNormal();
        let v = 1.0 + c * x;
        if v <= 0.0 {
            continue;
        }
        let v3 = v * v * v;
        let u = rng.nextF64();
        if u < 1.0 - 0.0331 * x * x * x * x {
            return d * v3;
        }
        if u > 0.0 && u.ln() < 0.5 * x * x + d * (1.0 - v3 + v3.ln()) {
            return d * v3;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normalSample(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = Xoshiro256pp::fromSeed(seed);
        (0..n).map(|_| rng.nextNormal()).collect()
    }

    fn exponentialSample(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = Xoshiro256pp::fromSeed(seed);
        (0..n).map(|_| -rng.nextF64().max(1e-12).ln()).collect()
    }

    #[test]
    fn shapiroWilkAcceptsNormalRejectsExponential() {
        let normal = normalSample(120, 11);
        let (w, p) = shapiroWilk(&normal).expect("normal sample");
        assert!(w > 0.95, "w = {}", w);
        assert!(p > 0.05, "p = {}", p);
        let expo = exponentialSample(200, 12);
        let (w2, p2) = shapiroWilk(&expo).expect("exp sample");
        assert!(w2 < w);
        assert!(p2 < 0.01, "p = {}", p2);
    }

    #[test]
    fn shapiroWilkRejectsBadSizes() {
        assert!(shapiroWilk(&[1.0, 2.0]).is_err());
        let big = vec![0.0f64; 5001];
        assert!(shapiroWilk(&big).is_err());
    }

    #[test]
    fn andersonDarlingCriticalValuesSane() {
        let sample = normalSample(200, 21);
        let r = andersonDarling(&sample, "normal").expect("ad normal");
        assert!(r.a2 < r.criticalValues[2], "a2 = {}", r.a2);
        for i in 1..5 {
            assert!(r.criticalValues[i] > r.criticalValues[i - 1]);
            assert!(r.significanceLevels[i] < r.significanceLevels[i - 1]);
        }
        let expo = exponentialSample(200, 22);
        let r2 = andersonDarling(&expo, "normal").expect("ad on exp data");
        assert!(r2.a2 > r2.criticalValues[4], "a2 = {}", r2.a2);
    }

    #[test]
    fn ks1SampleDetectsMismatch() {
        let sample = normalSample(300, 31);
        let (_, p_match) = ks1Sample(&sample, "normal", &[0.0, 1.0]).expect("ks match");
        assert!(p_match > 0.05, "p = {}", p_match);
        let (d, p_off) = ks1Sample(&sample, "normal", &[2.0, 1.0]).expect("ks mismatch");
        assert!(d > 0.5, "d = {}", d);
        assert!(p_off < 0.001, "p = {}", p_off);
        let (_, p_exp) = ks1Sample(&sample, "exp", &[1.0]).expect("ks exp");
        assert!(p_exp < 0.001);
    }

    #[test]
    fn fitDistributionRecoversExponential() {
        let sample = exponentialSample(600, 41);
        let fits = fitDistribution(&sample, &[]).expect("fit");
        assert_eq!(fits[0].name, "exp", "best fit was {:?}", fits[0]);
        assert!(
            (fits[0].params[0] - 1.0).abs() < 0.15,
            "lambda = {}",
            fits[0].params[0]
        );
        for i in 1..fits.len() {
            assert!(fits[i].aic >= fits[i - 1].aic);
        }
    }

    #[test]
    fn fitDistributionRejectsUnknownCandidate() {
        let sample = exponentialSample(50, 42);
        let err = fitDistribution(&sample, &["cauchy".to_string()]);
        assert!(err.is_err());
    }

    #[test]
    fn lofFlagsPlantedOutlier() {
        let mut rows: Vec<Vec<f64>> = Vec::new();
        let mut rng = Xoshiro256pp::fromSeed(51);
        for _ in 0..80 {
            rows.push(vec![rng.nextNormal() * 0.2, rng.nextNormal() * 0.2]);
        }
        rows.push(vec![8.0, 8.0]);
        let lof = localOutlierFactor(&rows, 5).expect("lof");
        let outlier = lof[80];
        let max_inlier = lof[..80].iter().cloned().fold(f64::MIN, f64::max);
        assert!(outlier > 2.0, "outlier lof = {}", outlier);
        assert!(
            outlier > max_inlier,
            "outlier {} vs inlier max {}",
            outlier,
            max_inlier
        );
    }

    #[test]
    fn lofValidatesK() {
        let rows = vec![vec![0.0], vec![1.0], vec![2.0]];
        assert!(localOutlierFactor(&rows, 0).is_err());
        assert!(localOutlierFactor(&rows, 3).is_err());
    }

    #[test]
    fn sequentialBoundariesShrinkForObrienFleming() {
        let sa = [40.0, 90.0, 130.0];
        let ta = [400.0, 800.0, 1200.0];
        let sb = [60.0, 130.0, 200.0];
        let tb = [400.0, 800.0, 1200.0];
        let looks = sequentialAnalyze(&sa, &ta, &sb, &tb, "obrien_fleming", 0.05).expect("seq");
        assert_eq!(looks.len(), 3);
        assert!(looks[0].boundary > looks[1].boundary);
        assert!(looks[1].boundary > looks[2].boundary);
        assert!(looks[2].crossed, "final z = {}", looks[2].z);
        let pocock = sequentialAnalyze(&sa, &ta, &sb, &tb, "pocock", 0.05).expect("seq pocock");
        assert!(pocock[2].boundary > normalQuantile(0.975));
    }

    #[test]
    fn bayesianAnalyzeFavorsBetterVariant() {
        let r = bayesianAnalyze(100, 1000, 150, 1000, (1.0, 1.0), (1.0, 1.0)).expect("bayes");
        assert!(r.probBBeatsA > 0.95, "prob = {}", r.probBBeatsA);
        assert!(r.expectedLossA > r.expectedLossB);
        assert!((r.meanA - 0.1).abs() < 0.01);
        assert!((r.meanB - 0.15).abs() < 0.01);
    }

    #[test]
    fn specialFunctionsAgreeWithKnownValues() {
        assert!((normalCdf(0.0) - 0.5).abs() < 1e-9);
        assert!((normalCdf(1.959963985) - 0.975).abs() < 1e-6);
        assert!((normalQuantile(0.975) - 1.959963985).abs() < 1e-6);
        // lnGamma(5) = ln(24)
        assert!((lnGamma(5.0) - 24.0f64.ln()).abs() < 1e-10);
        // digamma(1) = -euler_mascheroni
        assert!((digamma(1.0) + 0.5772156649015329).abs() < 1e-8);
        // chi square with 1 dof at x = 3.841 gives p close to 0.05
        let p = regularizedUpperGamma(0.5, 3.841 / 2.0);
        assert!((p - 0.05).abs() < 0.001, "p = {}", p);
    }
}
