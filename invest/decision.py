import json
import pandas as pd
import pyAgrum as gum
import invest.evaluation.validation as validation
from invest.preprocessing.simulation import simulate
from invest.store import Store

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
    pyAgrum.database
        A pyAgrum database suitable for learning CPTs.
    """
    # Select relevant columns for learning
    relevant_columns = [
        'PERelative_ShareMarket', 'PERelative_ShareSector', 'ForwardPE_CurrentVsHistory',
        'ROEvsCOE', 'RelDE', 'CAGRvsInflation', 'SystematicRisk',
        'Price', 'ShareBeta'
    ]
    
    learning_data = df[relevant_columns].copy()
    
    # Discretize continuous variables
    learning_data['PriceChange'] = df.groupby('Name')['Price'].pct_change()
    learning_data['PriceChange'] = pd.cut(learning_data['PriceChange'], bins=3, labels=['Negative', 'Stagnant', 'Positive'])
    
    # Convert categorical variables to discrete states
    for col in ['PERelative_ShareMarket', 'PERelative_ShareSector', 'ForwardPE_CurrentVsHistory']:
        learning_data[col] = pd.cut(learning_data[col], bins=3, labels=['Cheap', 'FairValue', 'Expensive'])
    
    for col in ['ROEvsCOE', 'RelDE', 'CAGRvsInflation']:
        learning_data[col] = pd.cut(learning_data[col], bins=3, labels=['Below', 'EqualTo', 'Above'])
    
    learning_data['SystematicRisk'] = pd.cut(learning_data['ShareBeta'], bins=3, labels=['lower', 'EqualTo', 'greater'])
    
    # Drop rows with NaN values
    learning_data = learning_data.dropna()
    
    # Convert to pyAgrum database
    gum_db = gum.DatabaseGenerator(learning_data)
    return gum_db

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

    for year in range(params.start, params.end):
        store = Store(df, companies, companies_jcsev, companies_jgind,
                      params.margin_of_safety, params.beta, year, False)
        investable_shares[str(year)] = []
        prices_initial[str(year)] = []
        prices_current[str(year)] = []
        betas[str(year)] = []
        df_future_performance = pd.DataFrame()
        for company in companies_dict[index_code]:
            if store.get_acceptable_stock(company):
                if not df_future_performance.empty:
                    future_performance = df_future_performance[company][0]
                else:
                    future_performance = None
                if investment_decision(store, company, value_net, quality_net, invest_net, future_performance, 
                                       params.extension, params.ablation, params.network) == "Yes":
                    mask = (df_['Date'] >= str(year) + '-01-01') & (
                            df_['Date'] <= str(year) + '-12-31') & (df_['Name'] == company)
                    df_year = df_[mask]

                    investable_shares[str(year)].append(company)
                    prices_initial[str(year)].append(df_year.iloc[0]['Price'])
                    prices_current[str(year)].append(df_year.iloc[params.holding_period]['Price'])
                    betas[str(year)].append(df_year.iloc[params.holding_period]["ShareBeta"])

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
    
    # print("Value evidence before normalization:", value_evidence)  # Debugging output
    
    # Make Value decision
    value_decision = value_net.make_decision(value_evidence)
    # print("Value decision:", value_decision)  # Debugging output

    # Prepare evidence for Quality Network
    quality_evidence = {
        'ROEvsCOE': store.get_roe_vs_coe(company),
        'RelDE': store.get_relative_debt_equity(company),
        'CAGRvsInflation': store.get_cagr_vs_inflation(company),
    }
    if extension:
        quality_evidence['SystematicRisk'] = store.get_systematic_risk(company)

    # print("Quality evidence before normalization:", quality_evidence)  # Debugging output

    # Make Quality decision
    try:
        quality_decision = quality_net.make_decision(quality_evidence)
        # print("Quality decision:", quality_decision)  # Debugging output
    except Exception as e:
        print(f"Error in quality decision: {str(e)}")
        print("Quality network structure:")
        for node in quality_net.model.nodes():
            var = quality_net.model.variable(node)
            print(f"  {var.name()}: {[var.label(i) for i in range(var.domainSize())]}")
        raise

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
    # print("Final decision:", final_decision)  # Debugging output
    return final_decision