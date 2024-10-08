import pyAgrum as gum

def learn_cpt_mdl(data, bn):
    learner = gum.BNLearner(data)
    learner.useScoreLog2Likelihood()
    learner.useScoreBDeu()
    learner.useScoreBIC()
    learner.useEM(epsilon=1e-4)
    
    # Create a new BN with the same structure as the input BN
    new_bn = gum.BayesNet()
    for node in bn.nodes():
        new_bn.add(bn.variable(node))
    for arc in bn.arcs():
        new_bn.addArc(arc[0], arc[1])
    
    # Learn parameters
    learned_bn = learner.learnParameters(new_bn)
    
    return learned_bn

def learn_cpt_bic(data, bn):
    learner = gum.BNLearner(data)
    learner.useScoreBIC()
    learner.useEM(epsilon=1e-4)
    
    # Create a new BN with the same structure as the input BN
    new_bn = gum.BayesNet()
    for node in bn.nodes():
        new_bn.add(bn.variable(node))
    for arc in bn.arcs():
        new_bn.addArc(arc[0], arc[1])
    
    # Learn parameters
    learned_bn = learner.learnParameters(new_bn)
    
    return learned_bn

def learn_cpt_mle(data, bn):
    learner = gum.BNLearner(data)
    learner.useEM(epsilon=1e-4)
    
    # Create a new BN with the same structure as the input BN
    new_bn = gum.BayesNet()
    for node in bn.nodes():
        new_bn.add(bn.variable(node))
    for arc in bn.arcs():
        new_bn.addArc(arc[0], arc[1])
    
    # Learn parameters
    learned_bn = learner.learnParameters(new_bn)
    
    return learned_bn