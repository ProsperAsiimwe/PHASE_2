import os
import numpy as np
import pyAgrum as gum

class InvestmentRecommendationNetwork:
    def __init__(self, learned_cpt=None):
        self.model = gum.InfluenceDiagram()

        # Decision node
        investable = gum.LabelizedVariable('Investable', 'Investable share', 2)
        investable.changeLabel(0, 'Yes')
        investable.changeLabel(1, 'No')
        self.model.addDecisionNode(investable)

        # Chance nodes
        share_performance = gum.LabelizedVariable('Performance', '', 3)
        share_performance.changeLabel(0, 'Positive')
        share_performance.changeLabel(1, 'Stagnant')
        share_performance.changeLabel(2, 'Negative')
        self.model.addChanceNode(share_performance)

        value = gum.LabelizedVariable('Value', 'Value', 3)
        value.changeLabel(0, 'Cheap')
        value.changeLabel(1, 'FairValue')
        value.changeLabel(2, 'Expensive')
        self.model.addChanceNode(value)

        quality = gum.LabelizedVariable('Quality', 'Quality', 3)
        quality.changeLabel(0, 'High')
        quality.changeLabel(1, 'Medium')
        quality.changeLabel(2, 'Low')
        self.model.addChanceNode(quality)

        # Utility node
        investment_utility = gum.LabelizedVariable('I_Utility', '', 1)
        self.model.addUtilityNode(investment_utility)

        # Add arcs
        self.model.addArc(self.model.idFromName('Performance'), self.model.idFromName('Quality'))
        self.model.addArc(self.model.idFromName('Performance'), self.model.idFromName('Value'))
        self.model.addArc(self.model.idFromName('Performance'), self.model.idFromName('I_Utility'))
        self.model.addArc(self.model.idFromName('Value'), self.model.idFromName('Investable'))
        self.model.addArc(self.model.idFromName('Quality'), self.model.idFromName('Investable'))
        self.model.addArc(self.model.idFromName('Investable'), self.model.idFromName('I_Utility'))

        # Set utilities
        self.model.utility(self.model.idFromName('I_Utility'))[{'Investable': 'Yes'}] = [[300], [-100], [-250]]
        self.model.utility(self.model.idFromName('I_Utility'))[{'Investable': 'No'}] = [[-200], [100], [200]]

        # Set initial CPTs
        self.model.cpt(self.model.idFromName('Performance'))[0] = 1 / 3  # Positive
        self.model.cpt(self.model.idFromName('Performance'))[1] = 1 / 3  # Stagnant
        self.model.cpt(self.model.idFromName('Performance'))[2] = 1 / 3  # Negative

        self.model.cpt(self.model.idFromName('Value'))[{'Performance': 'Positive'}] = [0.85, 0.10, 0.05]
        self.model.cpt(self.model.idFromName('Value'))[{'Performance': 'Stagnant'}] = [0.20, 0.60, 0.20]
        self.model.cpt(self.model.idFromName('Value'))[{'Performance': 'Negative'}] = [0.05, 0.10, 0.85]

        self.model.cpt(self.model.idFromName('Quality'))[{'Performance': 'Positive'}] = [0.85, 0.10, 0.05]
        self.model.cpt(self.model.idFromName('Quality'))[{'Performance': 'Stagnant'}] = [0.20, 0.60, 0.20]
        self.model.cpt(self.model.idFromName('Quality'))[{'Performance': 'Negative'}] = [0.05, 0.10, 0.85]

        if learned_cpt:
            self.update_cpts(learned_cpt)

    def update_cpts(self, learned_cpt):
        for node in self.model.nodes():
            if node in learned_cpt:
                self.model.cpt(node).fillWith(learned_cpt[node])

    def make_decision(self, value_decision, quality_decision):
        ie = gum.ShaferShenoyLIMIDInference(self.model)

        # Set evidence based on Value and Quality decisions
        ie.addEvidence('Value', self.model.variable('Value').index(value_decision))
        ie.addEvidence('Quality', self.model.variable('Quality').index(quality_decision))

        ie.makeInference()
        decision_index = np.argmax(ie.posteriorUtility('Investable').toarray())
        decision = self.model.variable('Investable').label(int(decision_index))

        return decision
