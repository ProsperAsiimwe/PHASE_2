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


def prepare_data_for_learning(df, value_net, quality_net, invest_net):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    try:
        max_year = df['Date'].dt.year.max()
        store = Store(df, companies, companies_jcsev, companies_jgind, 1.4, 0.6, max_year, False)
        store.process()
        
        learning_data = store.df_shares.copy()
        
        if learning_data.empty:
            print("Warning: Store did not produce any data. Check the Store class implementation.")
            return pd.DataFrame()
        
        # Get all variables from all networks
        all_variables = {}
        for network in [value_net, quality_net, invest_net]:
            for node in network.model.nodes():
                var = network.model.variable(node)
                all_variables[var.name()] = {
                    'domain_size': var.domainSize(),
                    'labels': [var.label(i) for i in range(var.domainSize())]
                }
        
        # Keep only the variables that are present in both the data and the networks
        common_variables = set(learning_data.columns) & set(all_variables.keys())
        learning_data = learning_data[list(common_variables)]
        
        # Ensure the data matches the domain size of the variables
        for var in common_variables:
            info = all_variables[var]
            learning_data[var] = pd.Categorical(learning_data[var], categories=info['labels'], ordered=True)
        
        if learning_data.empty or learning_data.isnull().all().all():
            print("Warning: No valid data for learning. Check data preprocessing.")
            return pd.DataFrame()
        
        print("Final learning data shape:", learning_data.shape)
        print("Final learning data columns:", learning_data.columns)
        print("Final learning data sample:")
        print(learning_data.head())
        print("\nVariable information:")
        for var in learning_data.columns:
            print(f"{var}: {learning_data[var].dtype}, Unique values: {learning_data[var].unique()}")
        
        return learning_data
    
    except Exception as e:
        print(f"Error in prepare_data_for_learning: {str(e)}")
        return pd.DataFrame()

def investment_portfolio(df_, params, index_code, value_net, quality_net, invest_net, verbose=False):
    print(f"Processing sector: {index_code}")
    print(f"Date range in data: {df_['Date'].min()} to {df_['Date'].max()}")
    print(f"Total rows in data: {len(df_)}")
    
    if params.noise:
        df = simulate(df_)
    else:
        df = df_

    prices_initial = {}
    prices_current = {}
    betas = {}
    investable_shares = {}

    for year in range(params.start, params.end):
        print(f"\nProcessing year {year}")
        year_data = df[(df['Date'] >= pd.Timestamp(f"{year}-01-01")) & (df['Date'] <= pd.Timestamp(f"{year}-12-31"))]
        print(f"Data for year {year}: {len(year_data)} rows")
        
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
                print(f"Company {company} is acceptable")
                if not df_future_performance.empty:
                    future_performance = df_future_performance[company][0]
                else:
                    future_performance = None
                if investment_decision(store, company, value_net, quality_net, invest_net, future_performance, 
                                       params.extension, params.ablation, params.network) == "Yes":
                    print(f"Company {company} selected for investment")
                    mask = (df_['Date'] >= f"{year}-01-01") & (
                            df_['Date'] <= f"{year}-12-31") & (df_['Name'] == company)
                    df_year = df_[mask]

                    if not df_year.empty:
                        investable_shares[str(year)].append(company)
                        prices_initial[str(year)].append(df_year.iloc[0]['Price'])
                        prices_current[str(year)].append(df_year.iloc[params.holding_period]['Price'])
                        betas[str(year)].append(df_year.iloc[params.holding_period]["ShareBeta"])
                    else:
                        print(f"Warning: No data found for {company} in year {year}")
                else:
                    print(f"Company {company} not selected for investment")
            else:
                print(f"Company {company} is not acceptable")

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
    
    print(f"Value evidence for {company}: {value_evidence}")
    
    # Make Value decision
    value_decision = value_net.make_decision(value_evidence)
    print(f"Value decision for {company}: {value_decision}")

    # Prepare evidence for Quality Network
    quality_evidence = {
        'ROEvsCOE': store.get_roe_vs_coe(company),
        'RelDE': store.get_relative_debt_equity(company),
        'CAGRvsInflation': store.get_cagr_vs_inflation(company),
    }
    if extension:
        quality_evidence['SystematicRisk'] = store.get_systematic_risk(company)

    print(f"Quality evidence for {company}: {quality_evidence}")

    # Make Quality decision
    quality_decision = quality_net.make_decision(quality_evidence)
    print(f"Quality decision for {company}: {quality_decision}")

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
    print(f"Investment decision for {company}:")
    print(f"Value evidence: {value_evidence}")
    print(f"Value decision: {value_decision}")
    print(f"Quality evidence: {quality_evidence}")
    print(f"Quality decision: {quality_decision}")
    print(f"Final decision: {final_decision}")
    return final_decision