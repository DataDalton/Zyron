//! Financial functions: NPV, IRR, loan calculations, depreciation, bonds.
//!
//! Uses f64 for calculations. IRR converges via Newton-Raphson
//! (max 100 iterations, 1e-10 threshold).

use zyron_common::{Result, ZyronError};

const IRR_MAX_ITERATIONS: usize = 100;
const IRR_CONVERGENCE_THRESHOLD: f64 = 1e-10;

// ---------------------------------------------------------------------------
// NPV and IRR
// ---------------------------------------------------------------------------

/// Net Present Value: sum of cashflows[i] / (1+rate)^i.
pub fn npv(rate: f64, cashflows: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut discount = 1.0;
    let factor = 1.0 + rate;
    for &cf in cashflows {
        sum += cf / discount;
        discount *= factor;
    }
    sum
}

/// Internal Rate of Return via Newton-Raphson.
/// Returns the rate for which NPV = 0.
pub fn irr(cashflows: &[f64]) -> Result<f64> {
    if cashflows.len() < 2 {
        return Err(ZyronError::ExecutionError(
            "IRR requires at least 2 cashflows".into(),
        ));
    }

    // Check for sign change
    let has_positive = cashflows.iter().any(|&v| v > 0.0);
    let has_negative = cashflows.iter().any(|&v| v < 0.0);
    if !has_positive || !has_negative {
        return Err(ZyronError::ExecutionError(
            "IRR requires at least one positive and one negative cashflow".into(),
        ));
    }

    let mut rate = 0.1; // initial guess
    for _ in 0..IRR_MAX_ITERATIONS {
        let (f, df) = npv_and_derivative(rate, cashflows);
        if df.abs() < 1e-15 {
            return Err(ZyronError::ExecutionError(
                "IRR derivative near zero, no convergence".into(),
            ));
        }
        let new_rate = rate - f / df;
        if (new_rate - rate).abs() < IRR_CONVERGENCE_THRESHOLD {
            return Ok(new_rate);
        }
        rate = new_rate;
        // Guard against divergence
        if !rate.is_finite() {
            return Err(ZyronError::ExecutionError(
                "IRR diverged to infinity".into(),
            ));
        }
    }

    Err(ZyronError::ExecutionError(format!(
        "IRR did not converge in {} iterations",
        IRR_MAX_ITERATIONS
    )))
}

fn npv_and_derivative(rate: f64, cashflows: &[f64]) -> (f64, f64) {
    let mut npv_val = 0.0;
    let mut derivative = 0.0;
    let factor = 1.0 + rate;
    let mut discount = 1.0;
    for (i, &cf) in cashflows.iter().enumerate() {
        npv_val += cf / discount;
        if i > 0 {
            derivative -= (i as f64) * cf / (discount * factor);
        }
        discount *= factor;
    }
    (npv_val, derivative)
}

/// Extended NPV with irregular date spacing.
/// dates: slice of epoch day values.
pub fn xnpv(rate: f64, dates: &[i32], cashflows: &[f64]) -> f64 {
    if dates.len() != cashflows.len() || dates.is_empty() {
        return 0.0;
    }
    let base_date = dates[0] as f64;
    let mut sum = 0.0;
    for (i, &cf) in cashflows.iter().enumerate() {
        let years = (dates[i] as f64 - base_date) / 365.0;
        sum += cf / (1.0 + rate).powf(years);
    }
    sum
}

/// Extended IRR with irregular date spacing.
pub fn xirr(dates: &[i32], cashflows: &[f64]) -> Result<f64> {
    if dates.len() != cashflows.len() {
        return Err(ZyronError::ExecutionError(
            "dates and cashflows must have same length".into(),
        ));
    }
    if cashflows.len() < 2 {
        return Err(ZyronError::ExecutionError(
            "XIRR requires at least 2 cashflows".into(),
        ));
    }

    let has_positive = cashflows.iter().any(|&v| v > 0.0);
    let has_negative = cashflows.iter().any(|&v| v < 0.0);
    if !has_positive || !has_negative {
        return Err(ZyronError::ExecutionError(
            "XIRR requires sign change".into(),
        ));
    }

    let mut rate: f64 = 0.1;
    for _ in 0..IRR_MAX_ITERATIONS {
        let base_date = dates[0] as f64;
        let mut f: f64 = 0.0;
        let mut df: f64 = 0.0;
        for (i, &cf) in cashflows.iter().enumerate() {
            let years = (dates[i] as f64 - base_date) / 365.0;
            let factor = (1.0 + rate).powf(years);
            f += cf / factor;
            if years > 0.0 {
                df -= years * cf / (factor * (1.0 + rate));
            }
        }

        if df.abs() < 1e-15 {
            return Err(ZyronError::ExecutionError(
                "XIRR derivative near zero".into(),
            ));
        }
        let new_rate = rate - f / df;
        if (new_rate - rate).abs() < IRR_CONVERGENCE_THRESHOLD {
            return Ok(new_rate);
        }
        rate = new_rate;
        if !rate.is_finite() {
            return Err(ZyronError::ExecutionError("XIRR diverged".into()));
        }
    }

    Err(ZyronError::ExecutionError("XIRR did not converge".into()))
}

// ---------------------------------------------------------------------------
// Time value of money
// ---------------------------------------------------------------------------

/// Payment amount for a loan: returns the periodic payment needed
/// to pay off a loan of `pv` at interest rate `rate` per period over `nper` periods.
pub fn pmt(rate: f64, nper: f64, pv: f64) -> f64 {
    if rate.abs() < 1e-12 {
        -pv / nper
    } else {
        -pv * (rate * (1.0 + rate).powf(nper)) / ((1.0 + rate).powf(nper) - 1.0)
    }
}

/// Future Value of a series of periodic payments.
pub fn fv(rate: f64, nper: f64, pmt_val: f64, pv: f64) -> f64 {
    if rate.abs() < 1e-12 {
        -(pv + pmt_val * nper)
    } else {
        let factor = (1.0 + rate).powf(nper);
        -(pv * factor + pmt_val * (factor - 1.0) / rate)
    }
}

/// Present Value of a series of periodic payments.
pub fn pv(rate: f64, nper: f64, pmt_val: f64, fv_val: f64) -> f64 {
    if rate.abs() < 1e-12 {
        -(fv_val + pmt_val * nper)
    } else {
        let factor = (1.0 + rate).powf(nper);
        -(fv_val + pmt_val * (factor - 1.0) / rate) / factor
    }
}

// ---------------------------------------------------------------------------
// Depreciation
// ---------------------------------------------------------------------------

/// Straight-line depreciation per period.
pub fn depreciation_sl(cost: f64, salvage: f64, life: f64) -> f64 {
    if life <= 0.0 {
        return 0.0;
    }
    (cost - salvage) / life
}

/// Declining balance depreciation for a given period.
pub fn depreciation_db(cost: f64, salvage: f64, life: f64, period: f64) -> f64 {
    if life <= 0.0 || period < 1.0 {
        return 0.0;
    }
    // Rate = 1 - (salvage/cost)^(1/life)
    let rate = if cost > 0.0 && salvage > 0.0 {
        1.0 - (salvage / cost).powf(1.0 / life)
    } else {
        1.0 / life
    };

    // Compute book value at start of period
    let mut book = cost;
    for _ in 1..(period as u32) {
        book -= book * rate;
    }
    book * rate
}

/// Sum-of-years digits depreciation for a given period.
pub fn depreciation_syd(cost: f64, salvage: f64, life: f64, period: f64) -> f64 {
    if life <= 0.0 || period < 1.0 || period > life {
        return 0.0;
    }
    let sum_of_years = life * (life + 1.0) / 2.0;
    let remaining_life = life - period + 1.0;
    (cost - salvage) * remaining_life / sum_of_years
}

// ---------------------------------------------------------------------------
// Bond pricing
// ---------------------------------------------------------------------------

/// Bond price: present value of coupon payments plus face value.
pub fn bond_price(face: f64, coupon_rate: f64, yield_rate: f64, periods: u32) -> f64 {
    let coupon_payment = face * coupon_rate;
    let mut price = 0.0;
    for t in 1..=periods {
        price += coupon_payment / (1.0 + yield_rate).powi(t as i32);
    }
    price += face / (1.0 + yield_rate).powi(periods as i32);
    price
}

/// Bond yield: solve for yield given price using bisection.
pub fn bond_yield(face: f64, coupon_rate: f64, price: f64, periods: u32) -> Result<f64> {
    if face <= 0.0 || price <= 0.0 {
        return Err(ZyronError::ExecutionError(
            "Bond parameters must be positive".into(),
        ));
    }

    let mut low = 0.0;
    let mut high = 1.0;
    // Expand bracket if needed
    while bond_price(face, coupon_rate, high, periods) > price {
        high *= 2.0;
        if high > 100.0 {
            return Err(ZyronError::ExecutionError("Bond yield too high".into()));
        }
    }

    for _ in 0..100 {
        let mid = (low + high) / 2.0;
        let p = bond_price(face, coupon_rate, mid, periods);
        if (p - price).abs() < 1e-6 {
            return Ok(mid);
        }
        if p > price {
            low = mid;
        } else {
            high = mid;
        }
        if (high - low).abs() < 1e-10 {
            return Ok((low + high) / 2.0);
        }
    }
    Ok((low + high) / 2.0)
}

// ---------------------------------------------------------------------------
// Compound interest
// ---------------------------------------------------------------------------

/// Compound interest: principal * (1 + rate/n)^(n*t).
pub fn compound_interest(principal: f64, rate: f64, n: f64, t: f64) -> f64 {
    principal * (1.0 + rate / n).powf(n * t)
}

// ---------------------------------------------------------------------------
// Amortization
// ---------------------------------------------------------------------------

/// Amortization schedule: returns vector of (payment, principal_portion, interest_portion, remaining_balance).
pub fn amortization_schedule(principal: f64, rate: f64, periods: u32) -> Vec<(f64, f64, f64, f64)> {
    // pmt() returns a negative number (cashflow out); negate for positive payment amount.
    let payment = -pmt(rate, periods as f64, principal);
    let mut balance = principal;
    let mut schedule = Vec::with_capacity(periods as usize);

    for _ in 0..periods {
        let interest = balance * rate;
        let principal_paid = payment - interest;
        balance -= principal_paid;
        schedule.push((payment, principal_paid, interest, balance.max(0.0)));
    }
    schedule
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_npv_positive() {
        // NPV at 10% of [-100, 60, 60] = -100 + 60/1.1 + 60/1.21 = ~4.13
        let result = npv(0.1, &[-100.0, 60.0, 60.0]);
        assert!((result - 4.132).abs() < 0.01);
    }

    #[test]
    fn test_npv_zero_rate() {
        let result = npv(0.0, &[-100.0, 50.0, 50.0, 50.0]);
        assert_eq!(result, 50.0);
    }

    #[test]
    fn test_irr_basic() {
        // Cashflows: -100 at time 0, 110 at time 1 -> IRR = 10%
        let rate = irr(&[-100.0, 110.0]).unwrap();
        assert!((rate - 0.10).abs() < 1e-6);
    }

    #[test]
    fn test_irr_requires_sign_change() {
        assert!(irr(&[100.0, 200.0]).is_err());
        assert!(irr(&[-100.0, -200.0]).is_err());
    }

    #[test]
    fn test_irr_too_few() {
        assert!(irr(&[-100.0]).is_err());
    }

    #[test]
    fn test_xnpv() {
        let dates = vec![0i32, 365, 730];
        let cashflows = vec![-100.0, 60.0, 60.0];
        let result = xnpv(0.1, &dates, &cashflows);
        assert!((result - 4.132).abs() < 0.01);
    }

    #[test]
    fn test_xirr() {
        let dates = vec![0i32, 365];
        let cashflows = vec![-100.0, 110.0];
        let rate = xirr(&dates, &cashflows).unwrap();
        assert!((rate - 0.10).abs() < 0.01);
    }

    #[test]
    fn test_pmt_zero_rate() {
        // $1000 loan over 10 periods with 0% interest = -100/period
        let p = pmt(0.0, 10.0, 1000.0);
        assert!((p - (-100.0)).abs() < 1e-10);
    }

    #[test]
    fn test_pmt_interest() {
        // $100,000 loan at 0.5% monthly for 360 periods
        let p = pmt(0.005, 360.0, 100000.0);
        // Monthly payment should be approximately -599.55
        assert!((p - (-599.55)).abs() < 1.0);
    }

    #[test]
    fn test_fv_compound() {
        // $1000 invested at 5% for 10 periods (no payment)
        let v = fv(0.05, 10.0, 0.0, -1000.0);
        // Future value = 1000 * 1.05^10 ~= 1628.89
        assert!((v - 1628.89).abs() < 1.0);
    }

    #[test]
    fn test_pv_future() {
        // $1000 in 10 periods at 5% discount rate
        let v = pv(0.05, 10.0, 0.0, -1000.0);
        // Present value ~ 613.91
        assert!((v - 613.91).abs() < 1.0);
    }

    #[test]
    fn test_depreciation_sl() {
        // $10000 asset, $1000 salvage, 5 year life = 1800/year
        let d = depreciation_sl(10000.0, 1000.0, 5.0);
        assert_eq!(d, 1800.0);
    }

    #[test]
    fn test_depreciation_db_period_1() {
        let d = depreciation_db(10000.0, 1000.0, 5.0, 1.0);
        assert!(d > 0.0);
    }

    #[test]
    fn test_depreciation_syd() {
        // $10000 - $1000 = $9000 depreciable over 5 years, SYD = 15
        // Year 1: 5/15 * 9000 = 3000
        let d = depreciation_syd(10000.0, 1000.0, 5.0, 1.0);
        assert!((d - 3000.0).abs() < 0.01);
    }

    #[test]
    fn test_bond_price_par() {
        // When coupon rate == yield rate, bond trades at par (face value)
        let price = bond_price(1000.0, 0.05, 0.05, 10);
        assert!((price - 1000.0).abs() < 0.01);
    }

    #[test]
    fn test_bond_price_premium() {
        // When coupon rate > yield rate, bond is at premium
        let price = bond_price(1000.0, 0.06, 0.05, 10);
        assert!(price > 1000.0);
    }

    #[test]
    fn test_bond_price_discount() {
        // When coupon rate < yield rate, bond is at discount
        let price = bond_price(1000.0, 0.04, 0.05, 10);
        assert!(price < 1000.0);
    }

    #[test]
    fn test_bond_yield_at_par() {
        let y = bond_yield(1000.0, 0.05, 1000.0, 10).unwrap();
        assert!((y - 0.05).abs() < 0.01);
    }

    #[test]
    fn test_compound_interest() {
        // $1000 at 10% compounded annually for 5 years
        let result = compound_interest(1000.0, 0.10, 1.0, 5.0);
        // 1000 * 1.10^5 ~= 1610.51
        assert!((result - 1610.51).abs() < 0.01);
    }

    #[test]
    fn test_amortization_schedule() {
        let schedule = amortization_schedule(100000.0, 0.005, 360);
        assert_eq!(schedule.len(), 360);
        // Final balance should be very close to zero (allowing floating-point drift).
        let (_, _, _, final_balance) = schedule[359];
        assert!(
            final_balance.abs() < 5.0,
            "Final balance too large: {}",
            final_balance
        );
    }

    #[test]
    fn test_amortization_payment_consistent() {
        let schedule = amortization_schedule(10000.0, 0.01, 12);
        let first_payment = schedule[0].0;
        for (pmt_val, _, _, _) in &schedule {
            assert!((pmt_val - first_payment).abs() < 0.01);
        }
    }
}
