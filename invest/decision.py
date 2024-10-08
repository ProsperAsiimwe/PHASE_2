import json
import pandas as pd
import pyAgrum as gum
import invest.evaluation.validation as validation
from invest.preprocessing.simulation import simulate
from invest.store import Store
import numpy as np

companies_jcsev = json.load(open('data/jcsev.json'))['names']
companies_jgind = json.load(open('data/jgind.json'))['names']
companies = companies_jcsev + companies_jgind
companies_dict = {"JCSEV": companies_jcsev, "JGIND": companies_jgind}


def prepare_data_for_learning(df):
    """
    Prepares data for CPT learning by extracting relevant features and converting them to a suitable format.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe containing all the financial data.
        
    Returns:
    --------
    pandas.DataFrame
        A DataFrame suitable for learning CPTs.
    """
    print("Available columns in the DataFrame:")
    print(df.columns.tolist())
    
    # Define mappings for column names
    column_mappings = {
        'current_PE_relative_share_market_to_historical': 'PERelative_ShareMarket',
        'current_PE_relative_share_sector_to_historical': 'PERelative_ShareSector',
        'forward_PE_current_to_historical': 'ForwardPE_CurrentVsHistory',
        'roe_vs_coe': 'ROEvsCOE',
        'relative_debt_to_equity': 'RelDE',
        'growth_cagr_vs_inflation': 'CAGRvsInflation',
        'systematic_risk': 'SystematicRisk',
        'Price': 'Price',
        'ShareBeta': 'ShareBeta',
        'Name': 'Name',
        'Date': 'Date'
    }
    
    # Select relevant columns that exist in the DataFrame
    existing_columns = [col for col in column_mappings.keys() if col in df.columns]
    learning_data = df[existing_columns].copy()
    
    # Rename columns to match the expected names
    learning_data.rename(columns={old: new for old, new in column_mappings.items() if old in existing_columns}, inplace=True)
    
    # Handle missing columns
    for expected_col in column_mappings.values():
        if expected_col not in learning_data.columns:
            print(f"Warning: Column '{expected_col}' is missing. Adding with NaN values.")
            learning_data[expected_col] = np.nan
    
    # Ensure 'Date' is in datetime format
    learning_data['Date'] = pd.to_datetime(learning_data['Date'])
    
    # Sort the DataFrame by 'Name' and 'Date'
    learning_data = learning_data.sort_values(['Name', 'Date'])
    
    # Calculate price change
    if 'Price' in learning_data.columns and 'Name' in learning_data.columns:
        learning_data['PriceChange'] = learning_data.groupby('Name')['Price'].pct_change()
        learning_data['PriceChange'] = pd.cut(learning_data['PriceChange'], bins=3, labels=['Negative', 'Stagnant', 'Positive'])
    else:
        print("Warning: 'Price' or 'Name' column not found. Setting PriceChange to NaN.")
        learning_data['PriceChange'] = np.nan
    
    # Convert categorical variables to discrete states
    for col in ['PERelative_ShareMarket', 'PERelative_ShareSector', 'ForwardPE_CurrentVsHistory']:
        if col in learning_data.columns and not learning_data[col].isna().all():
            learning_data[col] = pd.cut(learning_data[col], bins=3, labels=['Cheap', 'FairValue', 'Expensive'])
    
    for col in ['ROEvsCOE', 'RelDE', 'CAGRvsInflation']:
        if col in learning_data.columns and not learning_data[col].isna().all():
            learning_data[col] = pd.cut(learning_data[col], bins=3, labels=['Below', 'EqualTo', 'Above'])
    
    if 'ShareBeta' in learning_data.columns and not learning_data['ShareBeta'].isna().all():
        learning_data['SystematicRisk'] = pd.cut(learning_data['ShareBeta'], bins=3, labels=['lower', 'EqualTo', 'greater'])
    
    # Select only the columns we need for learning
    final_columns = ['PERelative_ShareMarket', 'PERelative_ShareSector', 'ForwardPE_CurrentVsHistory',
                     'ROEvsCOE', 'RelDE', 'CAGRvsInflation', 'SystematicRisk', 'PriceChange']
    learning_data = learning_data[final_columns]
    
    # print("Columns in the final learning data:")
    # print(learning_data.columns.tolist())
    
    print("Sample of the final learning data:")
    print(learning_data.head())
    
    # Convert all columns to string type, replacing NaN with 'NaN'
    learning_data = learning_data.astype(str).replace('nan', 'NaN')
    
    return learning_data

def investment_portfolio(df_, params, index_code, value_net, quality_net, invest_net, verbose=False):
    """
    Decides the shares for inclusion in an investment portfolio using INVEST
    Bayesian networks. Computes performance metrics for the IP and benchmark index.

    Parameters
    ----------
    df_ : pandas.DataFrame
        Fundamental and price data
    params : argparse.Namespace
        Command line arguments
    index_code: str,
        Johannesburg Stock Exchange sector index code
    value_net : ValueNetwork
        Instance of the Value Network
    quality_net : QualityNetwork
        Instance of the Quality Network
    invest_net : InvestmentRecommendationNetwork
        Instance of the Investment Recommendation Network
    verbose: bool, optional
        Print output to console

    Returns
    -------
    portfolio: dict
    """
    if params.noise:
        df = simulate(df_)
    else:
        df = df_

    prices_initial = {}
    prices_current = {}
    betas = {}
    investable_shares = {}

    print(f"Processing sector: {index_code}")
    print(f"Year range: {params.start} to {params.end}")

    for year in range(params.start, params.end):
        print(f"\nProcessing year {year}")
        store = Store(df, companies, companies_jcsev, companies_jgind,
                      params.margin_of_safety, params.beta, year, False)
        investable_shares[str(year)] = []
        prices_initial[str(year)] = []
        prices_current[str(year)] = []
        betas[str(year)] = []
        df_future_performance = pd.DataFrame()
        
        print(f"Number of companies being evaluated: {len(companies_dict[index_code])}")
        
        for company in companies_dict[index_code]:
            if store.get_acceptable_stock(company):
                # print(f"Company {company} is acceptable")
                if not df_future_performance.empty:
                    future_performance = df_future_performance[company][0]
                else:
                    future_performance = None
                if investment_decision(store, company, value_net, quality_net, invest_net, future_performance, 
                                       params.extension, params.ablation, params.network) == "Yes":
                    # print(f"Company {company} selected for investment")
                    mask = (df_['Date'] >= f"{year}-01-01") & (
                            df_['Date'] <= f"{year}-12-31") & (df_['Name'] == company)
                    df_year = df_[mask]

                    investable_shares[str(year)].append(company)
                    prices_initial[str(year)].append(df_year.iloc[0]['Price'])
                    prices_current[str(year)].append(df_year.iloc[params.holding_period]['Price'])
                    betas[str(year)].append(df_year.iloc[params.holding_period]["ShareBeta"])
            #     else:
            #         print(f"Company {company} not selected for investment")
            # else:
            #     print(f"Company {company} is not acceptable")

        print(f"Number of investable shares for year {year}: {len(investable_shares[str(year)])}")

    if verbose:
        print("\n{} {} - {}".format(index_code, params.start, params.end))
        print("-" * 50)
        print("\nInvestable Shares")
        for year in range(params.start, params.end):
            print(year, "IP." + index_code, len(investable_shares[str(year)]), investable_shares[str(year)])

    ip_ar, ip_cr, ip_aar, ip_treynor, ip_sharpe = validation.process_metrics(df_,
                                                                             prices_initial,
                                                                             prices_current,
                                                                             betas,
                                                                             params.start,
                                                                             params.end,
                                                                             index_code)
    benchmark_ar, benchmark_cr, benchmark_aar, benchmark_treynor, benchmark_sharpe = \
        validation.process_benchmark_metrics(params.start, params.end, index_code, params.holding_period)

    portfolio = {
        "ip": {
            "shares": investable_shares,
            "annualReturns": ip_ar,
            "compoundReturn": ip_cr,
            "averageAnnualReturn": ip_aar,
            "treynor": ip_treynor,
            "sharpe": ip_sharpe,
        },
        "benchmark": {
            "annualReturns": benchmark_ar,
            "compoundReturn": benchmark_cr,
            "averageAnnualReturn": benchmark_aar,
            "treynor": benchmark_treynor,
            "sharpe": benchmark_sharpe,
        }
    }
    return portfolio


def investment_decision(store, company, value_net, quality_net, invest_net, future_performance=None, 
                        extension=False, ablation=False, network='v'):
    # Prepare evidence for Value Network
    value_evidence = {
        'PERelative_ShareMarket': store.get_pe_relative_market(company),
        'PERelative_ShareSector': store.get_pe_relative_sector(company),
        'ForwardPE_CurrentVsHistory': store.get_forward_pe(company),
    }
    
    if future_performance is not None:
        value_evidence['FutureSharePerformance'] = future_performance
    
    # print(f"Value evidence for {company}: {value_evidence}")
    
    # Make Value decision
    value_decision = value_net.make_decision(value_evidence)
    # print(f"Value decision for {company}: {value_decision}")

    # Prepare evidence for Quality Network
    quality_evidence = {
        'ROEvsCOE': store.get_roe_vs_coe(company),
        'RelDE': store.get_relative_debt_equity(company),
        'CAGRvsInflation': store.get_cagr_vs_inflation(company),
    }
    if extension:
        quality_evidence['SystematicRisk'] = store.get_systematic_risk(company)

    # print(f"Quality evidence for {company}: {quality_evidence}")

    # Make Quality decision
    quality_decision = quality_net.make_decision(quality_evidence)
    # print(f"Quality decision for {company}: {quality_decision}")

    if ablation and network == 'v':
        if value_decision in ["Cheap", "FairValue"]:
            return "Yes"
        else:
            return "No"
    if ablation and network == 'q':
        if quality_decision in ["High", "Medium"]:
            return "Yes"
        else:
            return "No"
    
    final_decision = invest_net.make_decision(value_decision, quality_decision)
    # print(f"Final investment decision for {company}: {final_decision}")
    return final_decision