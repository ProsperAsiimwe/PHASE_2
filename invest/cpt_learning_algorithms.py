import pyAgrum as gum
import numpy as np
import pandas as pd

def learn_cpt_generic(data, bn, score_method):
    # Create a new BN with the same structure as the original
    learned_bn = gum.BayesNet()
    
    # Add nodes
    for node in bn.nodes():
        var = bn.variable(node)
        new_var = gum.LabelizedVariable(var.name(), "", var.domainSize())
        for i in range(var.domainSize()):
            new_var.changeLabel(i, var.label(i))
        learned_bn.add(new_var)
    
    # Add arcs
    for arc in bn.arcs():
        learned_bn.addArc(learned_bn.idFromName(bn.variable(arc[0]).name()),
                          learned_bn.idFromName(bn.variable(arc[1]).name()))

    # Prepare data
    new_data = pd.DataFrame()
    for node in learned_bn.nodes():
        var_name = learned_bn.variable(node).name()
        if var_name in data.columns:
            new_data[var_name] = data[var_name].astype(int)
        else:
            print(f"Warning: Variable {var_name} not found in data. Adding with default values.")
            new_data[var_name] = 0

    # Create a learner
    learner = gum.BNLearner(new_data)

    # Set the scoring method
    if score_method == 'MDL':
        learner.useScoreLog2Likelihood()
    elif score_method == 'BIC':
        learner.useScoreBIC()
    elif score_method == 'MLE':
        pass  # MLE is the default for parameter learning
    else:
        print(f"Unknown score method: {score_method}. Using default MLE.")

    # Use EM algorithm for parameter learning
    learner.useEM(epsilon=1e-4)

    try:
        # Learn parameters
        learned_params = learner.learnParameters(learned_bn)
        return learned_params
    except Exception as e:
        print(f"Error during parameter learning: {str(e)}")
        return None


def learn_cpt_mdl(data, bn):
    return learn_cpt_generic(data, bn, 'MDL')

def learn_cpt_bic(data, bn):
    return learn_cpt_generic(data, bn, 'BIC')

def learn_cpt_mle(data, bn):
    return learn_cpt_generic(data, bn, 'MLE')