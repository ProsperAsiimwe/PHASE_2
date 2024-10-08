import argparse
import time
import art
import os
import numpy as np
import pandas as pd
import pyAgrum as gum
from invest.decision import investment_portfolio, prepare_data_for_learning
from invest.preprocessing.dataloader import load_data
from invest.networks.value_evaluation import ValueNetwork
from invest.networks.quality_evaluation import QualityNetwork
from invest.networks.invest_recommendation import InvestmentRecommendationNetwork
from invest.cpt_learning_algorithms import learn_cpt_mdl, learn_cpt_bic, learn_cpt_mle

VERSION = 1.2

def walk_forward_validation(df, start_year, end_year, learning_method, args):
    results = {
        "JGIND": {"CR": [], "AAR": [], "TR": [], "SR": []},
        "JCSEV": {"CR": [], "AAR": [], "TR": [], "SR": []}
    }
    
    for train_end in range(start_year, end_year):
        train_df = df[df['Date'] < f"{train_end}-01-01"]
        test_df = df[(df['Date'] >= f"{train_end}-01-01") & (df['Date'] < f"{train_end+1}-01-01")]
        
        # Initialize networks
        value_net = ValueNetwork()
        quality_net = QualityNetwork(extension=args.extension)
        invest_net = InvestmentRecommendationNetwork()
        
        if learning_method != "original":
            learning_data = prepare_data_for_learning(train_df)
            
            # Check if learning_data is empty or all NaN
            if learning_data.empty or learning_data.isnull().all().all():
                print(f"Warning: No valid data for learning in year {train_end}. Using original network structures.")
            else:
                learner = gum.BNLearner(learning_data)
                
                # Use EM algorithm to handle missing values
                learner.useEM(1e-4)  # Using a default epsilon value of 1e-4
                
                if learning_method == "mdl":
                    learner.useScoreLog2Likelihood()
                elif learning_method == "bic":
                    learner.useScoreBIC()
                # For MLE, we don't need to set any specific scoring method
                
                # Learn CPTs for each network
                try:
                    value_cpt = learner.learnParameters(value_net.model.dag())
                    quality_cpt = learner.learnParameters(quality_net.model.dag())
                    invest_cpt = learner.learnParameters(invest_net.model.dag())
                    
                    value_net.update_cpts(value_cpt)
                    quality_net.update_cpts(quality_cpt)
                    invest_net.update_cpts(invest_cpt)
                except Exception as e:
                    print(f"Error learning parameters: {e}")
                    print("Using original network structures without learning")
        
        # Run investment portfolio for both sectors
        for sector in ["JGIND", "JCSEV"]:
            portfolio = investment_portfolio(test_df, args, sector, value_net, quality_net, invest_net, False)
            results[sector]["CR"].append(portfolio["ip"]["compoundReturn"])
            results[sector]["AAR"].append(portfolio["ip"]["averageAnnualReturn"])
            results[sector]["TR"].append(portfolio["ip"]["treynor"])
            results[sector]["SR"].append(portfolio["ip"]["sharpe"])
    
    return results

def get_learning_function(method):
    if method == "mdl":
        return learn_cpt_mdl
    elif method == "bic":
        return learn_cpt_bic
    elif method == "mle":
        return learn_cpt_mle
    else:
        raise ValueError(f"Unknown CPT learning method: {method}")

def run_experiments(df, args):
    methods = ["original", "mdl", "bic", "mle"]
    results = {method: {} for method in methods}
    
    for method in methods:
        print(f"\nRunning experiment for {method.upper()} method")
        try:
            results[method] = walk_forward_validation(df, args.start, args.end, method, args)
            print(f"Experiment for {method.upper()} completed successfully")
        except Exception as e:
            print(f"Error occurred during {method.upper()} experiment: {e}")
            results[method] = None
        
        # Print intermediate results
        if results[method] is not None:
            for sector in ["JGIND", "JCSEV"]:
                print(f"\nIntermediate results for {sector} using {method.upper()} method:")
                print(f"CR: {results[method][sector]['CR']}")
                print(f"AAR: {results[method][sector]['AAR']}")
                print(f"TR: {results[method][sector]['TR']}")
                print(f"SR: {results[method][sector]['SR']}")
    
    return results

def summarize_results(results):
    summary = {}
    for method, sector_results in results.items():
        summary[method] = {}
        for sector in ["JGIND", "JCSEV"]:
            summary[method][sector] = {
                "CR": np.mean(sector_results[sector]["CR"]),
                "AAR": np.mean(sector_results[sector]["AAR"]),
                "TR": np.mean(sector_results[sector]["TR"]),
                "SR": np.mean(sector_results[sector]["SR"])
            }
    return summary

def print_results_table(summary):
    for sector in ["JGIND", "JCSEV"]:
        print(f"\n{sector} Sector Results:")
        print("Method\t\tCR\t\tAAR\t\tTR\t\tSR")
        print("-" * 60)
        for method, results in summary.items():
            cr = results[sector]["CR"] * 100
            aar = results[sector]["AAR"] * 100
            tr = results[sector]["TR"]
            sr = results[sector]["SR"]
            print(f"{method.upper()}\t\t{cr:.2f}%\t\t{aar:.2f}%\t\t{tr:.2f}\t\t{sr:.2f}")

def main():
    start = time.time()
    df = load_data()
    
    results = run_experiments(df, args)
    summary = summarize_results(results)
    print_results_table(summary)
    
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\nTotal Experiment Time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Intelligent system for automated share evaluation',
                                     epilog=f'Version {VERSION}')
    parser.add_argument("--start", type=int, default=2015)
    parser.add_argument("--end", type=int, default=2018)
    parser.add_argument("--margin_of_safety", type=float, default=1.4)
    parser.add_argument("--beta", type=float, default=0.6)
    parser.add_argument("--extension", type=str2bool, default=False)
    parser.add_argument("--noise", type=str2bool, default=False)
    parser.add_argument("--ablation", type=str2bool, default=False)
    parser.add_argument("--network", type=str, default='v')
    parser.add_argument("--gnn", type=str2bool, default=False)
    parser.add_argument("--holding_period", type=int, default=-1)
    parser.add_argument("--horizon", type=int, default=10)
    args = parser.parse_args()

    print(art.text2art("INVEST"))
    print("Insaaf Dhansay & Kialan Pillay")
    print("© University of Cape Town 2021")
    print(f"Version {VERSION}")
    print("=" * 50)

    main()