import pyAgrum as gum

def learn_cpt_mdl(data, bn):
    learner = gum.BNLearner(data)
    learner.useScoreLog2Likelihood()
    learner.useScoreBDeu()
    learner.useScoreBIC()
    learner.useEM(epsilon=1e-4)
    
    # Create a new BN with only the variables present in both the data and the original BN
    new_bn = gum.BayesNet()
    for node in bn.nodes():
        var_name = bn.variable(node).name()
        if var_name in data.columns and not bn.isUtility(node):
            new_bn.add(bn.variable(node))
    
    for arc in bn.arcs():
        if bn.variable(arc[0]).name() in new_bn.names() and bn.variable(arc[1]).name() in new_bn.names():
            new_bn.addArc(arc[0], arc[1])
    
    # Learn parameters
    learned_bn = learner.learnParameters(new_bn)
    
    return learned_bn

def learn_cpt_bic(data, bn):
    learner = gum.BNLearner(data)
    learner.useScoreBIC()
    learner.useEM(epsilon=1e-4)
    
    # Create a new BN with only the variables present in both the data and the original BN
    new_bn = gum.BayesNet()
    for node in bn.nodes():
        var_name = bn.variable(node).name()
        if var_name in data.columns and not bn.isUtility(node):
            new_bn.add(bn.variable(node))
    
    for arc in bn.arcs():
        if bn.variable(arc[0]).name() in new_bn.names() and bn.variable(arc[1]).name() in new_bn.names():
            new_bn.addArc(arc[0], arc[1])
    
    # Learn parameters
    learned_bn = learner.learnParameters(new_bn)
    
    return learned_bn

def learn_cpt_mle(data, bn):
    learner = gum.BNLearner(data)
    learner.useEM(epsilon=1e-4)
    
    # Create a new BN with only the variables present in both the data and the original BN
    new_bn = gum.BayesNet()
    for node in bn.nodes():
        var_name = bn.variable(node).name()
        if var_name in data.columns and not bn.isUtility(node):
            new_bn.add(bn.variable(node))
    
    for arc in bn.arcs():
        if bn.variable(arc[0]).name() in new_bn.names() and bn.variable(arc[1]).name() in new_bn.names():
            new_bn.addArc(arc[0], arc[1])
    
    # Learn parameters
    learned_bn = learner.learnParameters(new_bn)
    
    return learned_bn