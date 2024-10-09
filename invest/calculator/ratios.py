import numpy as np

np.seterr(all="ignore")

def historic_earnings_growth_rate(eps_list, n):
    """
    Returns the Historic Earnings Growth Rate

    Parameters
    ----------
    eps_list : list
        Earnings per share for consecutive years
    n : int
        Number of years

    Returns
    -------
    float
    """
    if len(eps_list) < 2:
        return 0  # Not enough data to calculate growth rate
    
    growth_rates = []
    for year in range(len(eps_list) - 1):
        if eps_list[year] != 0:
            growth_rate = eps_list[year + 1] / eps_list[year] - 1
            growth_rates.append(growth_rate)
    
    if not growth_rates:
        return 0  # No valid growth rates calculated
    
    return np.mean(growth_rates)

def historic_earnings_cagr(eps_n, eps_prev_x, x):
    """
    Returns the Historic Earnings Compound Growth Rate

    Parameters
    ----------
    eps_n : float
        Earnings per share for year N (current year)
    x : int
        Number of years into the past
    eps_prev_x: float
        Earnings per share for year N-x

    Returns
    -------
    float
    """
    if eps_prev_x == 0 or x == 0:
        return 0  # Avoid division by zero
    
    cagr = ((eps_n / eps_prev_x) ** (1 / x)) - 1
    return 0 if np.isnan(cagr) else cagr

def historic_price_to_earnings_share(price_list, eps_list):
    """
    Returns the Historic Price to Earnings

    Parameters
    ----------
    price_list : numpy.ndarray
        Share prices over past years
    eps_list : numpy.ndarray
        Earnings per share over past years

    Returns
    -------
    float
    """
    if price_list.size == 0 or eps_list.size == 0:
        return 0  # Not enough data
    
    mean_eps = np.mean(eps_list)
    if mean_eps == 0:
        return 0  # Avoid division by zero
    
    return np.mean(price_list) / mean_eps

def forward_earnings(eps, historic_earnings_growth_rate_):
    """
    Returns the Forward Earnings

    Parameters
    ----------
    eps : float
        Earnings per share of current year
    historic_earnings_growth_rate_ : float
        Historic Earning Growth Rate

    Returns
    -------
    float
    """
    return eps * (1 + historic_earnings_growth_rate_)

def forward_earnings_cagr(forward_earnings_n, forward_earnings_prev_x, x):
    """
    Returns the Forward Earnings Compound Annual Growth Rate

    Parameters
    ----------
    forward_earnings_n : float
            Forward earnings for the current year
    forward_earnings_prev_x : float
        Forward Earnings for x years ago
    x: int
            Years into the past

    Returns
    -------
    float
    """
    if forward_earnings_prev_x == 0 or x == 0:
        return 0  # Avoid division by zero
    
    cagr = ((forward_earnings_n / forward_earnings_prev_x) ** (1 / x)) - 1
    return 0 if np.isnan(cagr) else cagr

def forward_price_to_earnings(share_price, forward_earnings_):
    """
    Returns the Forward Price to Earnings

    Parameters
    ----------
    share_price : float
        Current share price
    forward_earnings_ : float
        Forward Earnings for current year

    Returns
    -------
    float
    """
    if forward_earnings_ == 0:
        return 0  # Avoid division by zero
    
    return share_price / forward_earnings_

def pe_relative_sector(historic_price_to_earnings_share_, pe_sector_list):
    """
    Returns the Price to Earnings relative to the sector

    Parameters
    ----------
    historic_price_to_earnings_share_ : float
        Historic Price to Earnings of the share
    pe_sector_list : list
         Price to Earnings for the sector for past years

    Returns
    -------
    float
    """
    if not pe_sector_list:
        return 0  # Not enough data
    
    mean_pe_sector = np.mean(pe_sector_list)
    if mean_pe_sector == 0:
        return 0  # Avoid division by zero
    
    return historic_price_to_earnings_share_ / mean_pe_sector

def pe_relative_market(historic_price_to_earnings_share_, pe_market):
    """
    Returns the Price to Earnings relative to the market

    Parameters
    ----------
    historic_price_to_earnings_share_ : float
        Historic Price to Earnings of the share
    pe_market : list
        Price to Earnings for the market for past years

    Returns
    -------
    float
    """
    if not pe_market:
        return 0  # Not enough data
    
    mean_pe_market = np.mean(pe_market)
    if mean_pe_market == 0:
        return 0  # Avoid division by zero
    
    return historic_price_to_earnings_share_ / mean_pe_market

def cost_of_equity(market_return_rate, risk_free_return_rate, share_beta):
    """
    Returns the Cost of Equity

    Parameters
    ----------
    market_return_rate : float
         Market rate of return for the current year
    risk_free_return_rate : float
        Risk free rate of return for the current year
    share_beta: float
        Beta of share on last day of the year
    Returns
    -------
    float
    """
    equity_risk_premium = market_return_rate - risk_free_return_rate
    return risk_free_return_rate + share_beta * equity_risk_premium

def relative_debt_to_equity(debt_equity, debt_equity_industry):
    """
    Returns the Relative Debt to Equity

    Parameters
    ----------
    debt_equity : float
         Debt Equity of the share
    debt_equity_industry : float
        Debt Equity of the industry

    Returns
    -------
    float
    """
    if debt_equity_industry == 0:
        return 0  # Avoid division by zero
    
    return debt_equity / debt_equity_industry

def current_pe_market(current_share_pe, current_market_pe):
    """
    Returns the Price to Earnings relative to the market for the current year

    Parameters
    ----------
    current_share_pe : float
            PE of the current share
    current_market_pe : float
        PE of the industry

    Returns
    -------
    float
    """
    if current_market_pe == 0:
        return 0  # Avoid division by zero
    
    return current_share_pe / current_market_pe

def current_pe_sector(current_share_pe, current_sector_pe):
    """
    Returns the Price to Earnings relative to the market for the current year

    Parameters
    ----------
    current_share_pe : float
        PE of the current share
    current_sector_pe : float
        PE of the sector

    Returns
    -------
    float
    """
    if current_sector_pe == 0:
        return 0  # Avoid division by zero
    
    return current_share_pe / current_sector_pe