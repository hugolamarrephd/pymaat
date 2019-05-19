import numpy as np

# Price


def price_formatter(x):
    return 100.*x


def log_price_formatter(x):
    return 100.*np.exp(x)

# Variance


def yearly_var(x):
    return 100.*np.sqrt(np.maximum(x, 0.))


def monthly_var(x):
    return 100.*np.sqrt(12.*np.maximum(x, 0.))


def weekly_var(x):
    return 100.*np.sqrt(52.*np.maximum(x, 0.))


def daily_var(x):
    return 100.*np.sqrt(252.*np.maximum(x, 0.))


def variance_formatter(freq):
    if freq.lower() == 'daily':
        return daily_var
    elif freq.lower() == 'weekly':
        return weekly_var
    elif freq.lower() == 'monthly':
        return monthly_var
    elif freq.lower() == 'yearly':
        return yearly_var
    else:
        raise ValueError("Unexpected frequency")
