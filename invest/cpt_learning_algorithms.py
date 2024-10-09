import pyAgrum as gum

def learn_cpt_mdl(data, bn):
    common_variables = [var for var in bn.names() if var in data.columns and bn.isChanceNode(bn.idFromName(var))]
    
    if not common_variables:
        print("No common variables found between the data and the Bayesian Network.")
        return None

    new_bn = gum.BayesNet()
    for var in common_variables:
        orig_var = bn.variable(bn.idFromName(var))
        new_var = gum.LabelizedVariable(var, "", orig_var.domainSize())
        for i in range(orig_var.domainSize()):
            new_var.changeLabel(i, orig_var.label(i))
        new_bn.add(new_var)
    
    for arc in bn.arcs():
        if bn.variable(arc[0]).name() in common_variables and bn.variable(arc[1]).name() in common_variables:
            new_bn.addArc(new_bn.idFromName(bn.variable(arc[0]).name()), new_bn.idFromName(bn.variable(arc[1]).name()))
    
    new_data = data[common_variables].copy()
    
    # Ensure data categories match the original network
    for var in common_variables:
        orig_var = bn.variable(bn.idFromName(var))
        valid_labels = [orig_var.label(i) for i in range(orig_var.domainSize())]
        new_data[var] = new_data[var].apply(lambda x: x if x in valid_labels else np.nan)
    
    new_data = new_data.dropna()
    
    if new_data.empty:
        print("No valid data remaining after aligning categories with the original network.")
        return None

    learner = gum.BNLearner(new_data)
    learner.useScoreLog2Likelihood()
    learner.useScoreBDeu()
    learner.useScoreBIC()
    learner.useEM(epsilon=1e-4)
    
    try:
        learned_bn = learner.learnParameters(new_bn)
        return learned_bn
    except Exception as e:
        print(f"Error during parameter learning: {str(e)}")
        return None

def learn_cpt_bic(data, bn):
    common_variables = [var for var in bn.names() if var in data.columns and bn.isChanceNode(bn.idFromName(var))]
    
    if not common_variables:
        print("No common variables found between the data and the Bayesian Network.")
        return None

    new_bn = gum.BayesNet()
    for var in common_variables:
        orig_var = bn.variable(bn.idFromName(var))
        new_var = gum.LabelizedVariable(var, "", orig_var.domainSize())
        for i in range(orig_var.domainSize()):
            new_var.changeLabel(i, orig_var.label(i))
        new_bn.add(new_var)
    
    for arc in bn.arcs():
        if bn.variable(arc[0]).name() in common_variables and bn.variable(arc[1]).name() in common_variables:
            new_bn.addArc(new_bn.idFromName(bn.variable(arc[0]).name()), new_bn.idFromName(bn.variable(arc[1]).name()))
    
    new_data = data[common_variables].copy()
    
    # Ensure data categories match the original network
    for var in common_variables:
        orig_var = bn.variable(bn.idFromName(var))
        valid_labels = [orig_var.label(i) for i in range(orig_var.domainSize())]
        new_data[var] = new_data[var].apply(lambda x: x if x in valid_labels else np.nan)
    
    new_data = new_data.dropna()
    
    if new_data.empty:
        print("No valid data remaining after aligning categories with the original network.")
        return None

    learner = gum.BNLearner(new_data)
    learner.useScoreBIC()
    learner.useEM(epsilon=1e-4)
    
    try:
        learned_bn = learner.learnParameters(new_bn)
        return learned_bn
    except Exception as e:
        print(f"Error during parameter learning: {str(e)}")
        return None

def learn_cpt_mle(data, bn):
    common_variables = [var for var in bn.names() if var in data.columns and bn.isChanceNode(bn.idFromName(var))]
    
    if not common_variables:
        print("No common variables found between the data and the Bayesian Network.")
        return None

    new_bn = gum.BayesNet()
    for var in common_variables:
        orig_var = bn.variable(bn.idFromName(var))
        new_var = gum.LabelizedVariable(var, "", orig_var.domainSize())
        for i in range(orig_var.domainSize()):
            new_var.changeLabel(i, orig_var.label(i))
        new_bn.add(new_var)
    
    for arc in bn.arcs():
        if bn.variable(arc[0]).name() in common_variables and bn.variable(arc[1]).name() in common_variables:
            new_bn.addArc(new_bn.idFromName(bn.variable(arc[0]).name()), new_bn.idFromName(bn.variable(arc[1]).name()))
    
    new_data = data[common_variables].copy()
    
    # Ensure data categories match the original network
    for var in common_variables:
        orig_var = bn.variable(bn.idFromName(var))
        valid_labels = [orig_var.label(i) for i in range(orig_var.domainSize())]
        new_data[var] = new_data[var].apply(lambda x: x if x in valid_labels else np.nan)
    
    new_data = new_data.dropna()
    
    if new_data.empty:
        print("No valid data remaining after aligning categories with the original network.")
        return None


    learner = gum.BNLearner(new_data)
    learner.useEM(epsilon=1e-4)
    
    try:
        learned_bn = learner.learnParameters(new_bn)
        return learned_bn
    except Exception as e:
        print(f"Error during parameter learning: {str(e)}")
        return None