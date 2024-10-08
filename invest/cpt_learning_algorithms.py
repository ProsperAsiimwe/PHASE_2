import pyAgrum as gum

def learn_cpt_mdl(data, bn, smoothing=0.01):
    learner = gum.BNLearner(data)
    learner.useScoreLog2Likelihood()
    learner.useScoreBDeu()
    learner.useScoreBIC()
    learner.useSmoothing(smoothing)
    learner.setInitialDAG(bn)
    learned_bn = learner.learnParameters(bn.dag())
    return learned_bn

def learn_cpt_bic(data, bn, smoothing=0.01):
    learner = gum.BNLearner(data)
    learner.useScoreBIC()
    learner.useSmoothing(smoothing)
    learner.setInitialDAG(bn)
    learned_bn = learner.learnParameters(bn.dag())
    return learned_bn

def learn_cpt_mle(data, bn, smoothing=0.01):
    learner = gum.BNLearner(data)
    learner.useSmoothing(smoothing)
    learner.setInitialDAG(bn)
    learned_bn = learner.learnParameters(bn.dag())
    return learned_bn