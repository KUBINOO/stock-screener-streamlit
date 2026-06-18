import yfinance as yf

def get_dcf_base_data(ticker):
    """
    It retrieves the basic data needed to calculate the DCF: FCF, debt, cash, and number of shares.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Free Cash Flow (if unavailable, we'll use operating cash flow)
        fcf = info.get('freeCashflow', 0)
        if not fcf:
            fcf = info.get('operatingCashflow', 0)
            
        total_debt = info.get('totalDebt', 0)
        total_cash = info.get('totalCash', 0)
        net_debt = total_debt - total_cash
        
        shares = info.get('sharesOutstanding', 0)
        current_price = info.get('currentPrice', 0)
        
        return {
            "fcf_ttm": fcf,
            "net_debt": net_debt,
            "shares_outstanding": shares,
            "current_price": current_price
        }
    
        # New dynamic parameters from yfinance
        beta = info.get('beta', 1.1)
        total_equity = info.get('totalStockholderEquity', 0)
        total_debt = info.get('totalDebt', 0)
        
        # Safe calculation of Debt-to-Equity ratio
        debt_to_equity = (total_debt / total_equity) if total_equity and total_equity > 0 else 0.5

        return {
            "fcf_ttm": fcf,
            "net_debt": net_debt,
            "shares_outstanding": shares,
            "current_price": current_price,
            "beta": beta,
            "debt_to_equity": debt_to_equity
        }
    
    except Exception:
        return None

def calculate_dcf(fcf_ttm, phase1_years, growth_rate, terminal_growth, wacc, shares_outstanding, net_debt):
    """
    Vypočítá vnitřní hodnotu akcie (Intrinsic Value) pomocí 2-Stage DCF modelu.
    """
    # Division by zero protection
    if wacc <= terminal_growth or shares_outstanding == 0:
        return 0, 0

    pv_fcf = 0
    current_fcf = fcf_ttm

    # Phase 1: Present value of future FCFs over the first X years
    for year in range(1, int(phase1_years) + 1):
        current_fcf *= (1 + growth_rate)
        pv_fcf += current_fcf / ((1 + wacc) ** year)

    # Phase 2: Present value of the terminal phase (to infinity)
    fcf_terminal = current_fcf * (1 + terminal_growth)
    terminal_value = fcf_terminal / (wacc - terminal_growth)
    pv_tv = terminal_value / ((1 + wacc) ** phase1_years)

    # Enterprise Value
    enterprise_value = pv_fcf + pv_tv
    
    # Equity Value
    equity_value = enterprise_value - net_debt

    # Intrinsic value per share
    intrinsic_value = equity_value / shares_outstanding

    return intrinsic_value, enterprise_value

def calculate_reverse_dcf(current_price, fcf_ttm, phase1_years, terminal_growth, wacc, shares_outstanding, net_debt):
    """
    Vypočítá implikovaný růst FCF (Reverse DCF), který ospravedlňuje aktuální tržní cenu akcie.
    Používá metodu binárního vyhledávání (Binary Search).
    """
    if current_price <= 0 or wacc <= terminal_growth or shares_outstanding == 0:
        return 0.0

    low_growth = -0.99  # Lower limit: -99%
    high_growth = 5.0   # Upper limit: +500 %
    tolerance = 0.01    # Accuracy to the cent
    
    implied_growth = 0.0

    # Binary Search for the Correct Growth Rate
    for _ in range(100): # Max 100 iterations to prevent an infinite loop
        mid_growth = (low_growth + high_growth) / 2
        
        # Let's call our existing DCF function with the tested growth rate
        calc_price, _ = calculate_dcf(
            fcf_ttm, phase1_years, mid_growth, 
            terminal_growth, wacc, shares_outstanding, net_debt
        )
        
        # If we found the correct growth (within the tolerance range)
        if abs(calc_price - current_price) < tolerance:
            implied_growth = mid_growth
            break
            
        # Adjusting the borders depending on whether we are below or above
        if calc_price < current_price:
            low_growth = mid_growth
        else:
            high_growth = mid_growth
            
    return implied_growth