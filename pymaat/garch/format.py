import numpy as np

# Price
def base_price(x):
    return 100.*x

# Variance
def monthly_var(x):
    return 100.*np.sqrt(12.*x)

def weekly_var(x):
    return 100.*np.sqrt(52.*x)

def daily_var(x):
    return 100.*np.sqrt(252.*x)

def variance_formatter(freq):
    if freq.lower() == 'daily':
        return daily_var
    elif freq.lower() == 'weekly':
        return weekly_var
    elif freq.lower() == 'monthly':
        return monthly_var
