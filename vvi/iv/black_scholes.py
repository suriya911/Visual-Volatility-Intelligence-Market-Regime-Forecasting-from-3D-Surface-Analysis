from __future__ import annotations

import math

from scipy.optimize import brentq
from scipy.stats import norm


def bs_price(spot: float, strike: float, t: float, rate: float, sigma: float, option_type: str) -> float:
    if t <= 0 or sigma <= 0:
        return max(0.0, spot - strike) if option_type == "c" else max(0.0, strike - spot)

    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)

    if option_type == "c":
        return spot * norm.cdf(d1) - strike * math.exp(-rate * t) * norm.cdf(d2)
    return strike * math.exp(-rate * t) * norm.cdf(-d2) - spot * norm.cdf(-d1)


def implied_volatility(
    option_price: float,
    spot: float,
    strike: float,
    t: float,
    rate: float,
    option_type: str,
) -> float:
    if option_price <= 0 or spot <= 0 or strike <= 0 or t <= 0:
        return float("nan")

    def objective(sigma: float) -> float:
        return bs_price(spot, strike, t, rate, sigma, option_type) - option_price

    try:
        return float(brentq(objective, 1e-4, 5.0, maxiter=100))
    except ValueError:
        return float("nan")
