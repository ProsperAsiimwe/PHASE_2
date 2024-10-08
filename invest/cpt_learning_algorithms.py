import pyAgrum as gum

def learn_cpt_mdl(data, bn):
    learner = gum.BNLearner(data)
    learner.useScoreLog2Likelihood()
    learner.useScoreBDeu()
    learner.useScoreBIC()
    learner.setInitialDAG(bn)
    learned_bn = learner.learnBN()
    return learned_bn

def learn_cpt_bic(data, bn):
    learner = gum.BNLearner(data)
    learner.useScoreBIC()
    learner.setInitialDAG(bn)
    learned_bn = learner.learnBN()
    return learned_bn

def learn_cpt_mle(data, bn, smoothing=0.01):
    learner = gum.BNLearner(data)
    learner.useSmoothing(smoothing)  # Setting smoothing to replace useAprioriSmoothing
    learner.setInitialDAG(bn)  # Using the provided Bayesian Network structure
    learned_bn = learner.learnParameters(bn.dag())  # Learning parameters based on the DAG
    return learned_bn