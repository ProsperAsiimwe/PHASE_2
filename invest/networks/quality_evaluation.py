import os
import numpy as np
import pyAgrum as gum

class QualityNetwork:
    def __init__(self, learned_cpt=None, extension=False):
        self.model = gum.InfluenceDiagram()
        self.extension = extension

        # Decision node
        quality_decision = gum.LabelizedVariable('Quality', '', 3)
        quality_decision.changeLabel(0, 'High')
        quality_decision.changeLabel(1, 'Medium')
        quality_decision.changeLabel(2, 'Low')
        self.model.addDecisionNode(quality_decision)

        # Chance nodes
        future_share_performance = gum.LabelizedVariable('FutureSharePerformance', '', 3)
        future_share_performance.changeLabel(0, 'Positive')
        future_share_performance.changeLabel(1, 'Stagnant')
        future_share_performance.changeLabel(2, 'Negative')
        self.model.addChanceNode(future_share_performance)

        cagr_vs_inflation = gum.LabelizedVariable('CAGRvsInflation', '', 3)
        cagr_vs_inflation.changeLabel(0, 'InflationPlus')
        cagr_vs_inflation.changeLabel(1, 'Inflation')
        cagr_vs_inflation.changeLabel(2, 'InflationMinus')
        self.model.addChanceNode(cagr_vs_inflation)

        roe_vs_coe = gum.LabelizedVariable('ROEvsCOE', '', 3)
        roe_vs_coe.changeLabel(0, 'Above')
        roe_vs_coe.changeLabel(1, 'EqualTo')
        roe_vs_coe.changeLabel(2, 'Below')
        self.model.addChanceNode(roe_vs_coe)

        relative_debt_equity = gum.LabelizedVariable('RelDE', '', 3)
        relative_debt_equity.changeLabel(0, 'Above')
        relative_debt_equity.changeLabel(1, 'EqualTo')
        relative_debt_equity.changeLabel(2, 'Below')
        self.model.addChanceNode(relative_debt_equity)

        # Utility node
        quality_utility = gum.LabelizedVariable('Q_Utility', '', 1)
        self.model.addUtilityNode(quality_utility)

        # Add arcs
        self.model.addArc(self.model.idFromName('FutureSharePerformance'), self.model.idFromName('CAGRvsInflation'))
        self.model.addArc(self.model.idFromName('FutureSharePerformance'), self.model.idFromName('ROEvsCOE'))
        self.model.addArc(self.model.idFromName('FutureSharePerformance'), self.model.idFromName('RelDE'))
        self.model.addArc(self.model.idFromName('FutureSharePerformance'), self.model.idFromName('Q_Utility'))

        self.model.addArc(self.model.idFromName('CAGRvsInflation'), self.model.idFromName('Quality'))
        self.model.addArc(self.model.idFromName('ROEvsCOE'), self.model.idFromName('Quality'))
        self.model.addArc(self.model.idFromName('RelDE'), self.model.idFromName('Quality'))
        self.model.addArc(self.model.idFromName('Quality'), self.model.idFromName('Q_Utility'))

        # Set utilities
        self.model.utility(self.model.idFromName('Q_Utility'))[{'Quality': 'High'}] = [[100], [0], [-100]]
        self.model.utility(self.model.idFromName('Q_Utility'))[{'Quality': 'Medium'}] = [[50], [100], [-50]]
        self.model.utility(self.model.idFromName('Q_Utility'))[{'Quality': 'Low'}] = [[0], [50], [100]]

        # Set initial CPTs
        self.model.cpt(self.model.idFromName('FutureSharePerformance'))[0] = 1 / 3  # Positive
        self.model.cpt(self.model.idFromName('FutureSharePerformance'))[1] = 1 / 3  # Stagnant
        self.model.cpt(self.model.idFromName('FutureSharePerformance'))[2] = 1 / 3  # Negative

        self.model.cpt(self.model.idFromName('RelDE'))[{'FutureSharePerformance': 'Positive'}] = [0.05, 0.15, 0.80]
        self.model.cpt(self.model.idFromName('RelDE'))[{'FutureSharePerformance': 'Stagnant'}] = [0.15, 0.70, 0.15]
        self.model.cpt(self.model.idFromName('RelDE'))[{'FutureSharePerformance': 'Negative'}] = [0.80, 0.15, 0.05]

        self.model.cpt(self.model.idFromName('ROEvsCOE'))[{'FutureSharePerformance': 'Positive'}] = [0.80, 0.15, 0.05]
        self.model.cpt(self.model.idFromName('ROEvsCOE'))[{'FutureSharePerformance': 'Stagnant'}] = [0.20, 0.60, 0.20]
        self.model.cpt(self.model.idFromName('ROEvsCOE'))[{'FutureSharePerformance': 'Negative'}] = [0.05, 0.15, 0.80]

        self.model.cpt(self.model.idFromName('CAGRvsInflation'))[{'FutureSharePerformance': 'Positive'}] = [0.80, 0.15, 0.05]
        self.model.cpt(self.model.idFromName('CAGRvsInflation'))[{'FutureSharePerformance': 'Stagnant'}] = [0.15, 0.70, 0.15]
        self.model.cpt(self.model.idFromName('CAGRvsInflation'))[{'FutureSharePerformance': 'Negative'}] = [0.05, 0.15, 0.8]

        # Extension
        if self.extension:
            systematic_risk = gum.LabelizedVariable('SystematicRisk', '', 3)
            systematic_risk.changeLabel(0, 'greater')  # Greater than Market
            systematic_risk.changeLabel(1, 'EqualTo')
            systematic_risk.changeLabel(2, 'lower')
            self.model.addChanceNode(systematic_risk)

            self.model.addArc(self.model.idFromName('FutureSharePerformance'), self.model.idFromName('SystematicRisk'))
            self.model.addArc(self.model.idFromName('SystematicRisk'), self.model.idFromName('Quality'))

            # Add CPTs for Systematic Risk
            self.model.cpt(self.model.idFromName('SystematicRisk'))[{'FutureSharePerformance': 'Positive'}] = [0.80, 0.15, 0.05]
            self.model.cpt(self.model.idFromName('SystematicRisk'))[{'FutureSharePerformance': 'Stagnant'}] = [0.15, 0.70, 0.15]
            self.model.cpt(self.model.idFromName('SystematicRisk'))[{'FutureSharePerformance': 'Negative'}] = [0.05, 0.15, 0.8]

        if learned_cpt:
            self.update_cpts(learned_cpt)

    def update_cpts(self, learned_cpt):
        for node in self.model.nodes():
            var_name = self.model.variable(node).name()
            if var_name in learned_cpt.names() and not self.model.isUtility(node):
                self.model.cpt(node).fillWith(learned_cpt.cpt(var_name))

    def print_variable_names(self):
        print("Quality Network Variables:")
        for node in self.model.nodes():
            print(self.model.variable(node).name())

    def normalize_label(self, var, label):
        """Normalize label to match the model's labels for the specific variable."""
        label_map = {
            'ROEvsCOE': {
                'above': 'Above',
                'equalto': 'EqualTo',
                'below': 'Below'
            },
            'RelDE': {
                'above': 'Above',
                'equalto': 'EqualTo',
                'below': 'Below'
            },
            'CAGRvsInflation': {
                'above': 'InflationPlus',
                'equalto': 'Inflation',
                'below': 'InflationMinus',
                'inflationplus': 'InflationPlus',
                'inflation': 'Inflation',
                'inflationminus': 'InflationMinus'
            },
            'SystematicRisk': {
                'greater': 'greater',
                'equalto': 'EqualTo',
                'lower': 'lower'
            },
            'Quality': {
                'high': 'High',
                'medium': 'Medium',
                'low': 'Low'
            }
        }
        return label_map.get(var, {}).get(str(label).lower(), label)

    def normalize_evidence(self, evidence):
        """Normalize the evidence labels to match the model's labels."""
        normalized = {}
        for var, val in evidence.items():
            if pd.isna(val):
                continue  # Skip NaN values
            if isinstance(val, str):
                normalized[var] = self.normalize_label(var, val)
            elif val is not None:
                normalized[var] = val
        return normalized

    def make_decision(self, evidence):
        ie = gum.ShaferShenoyLIMIDInference(self.model)

        normalized_evidence = self.normalize_evidence(evidence)
        print("Normalized quality evidence:", normalized_evidence)  # Debugging output

        for var, val in normalized_evidence.items():
            if val is None:
                continue  # Skip None values
            elif isinstance(val, str):
                try:
                    variable = self.model.variable(var)
                    if val not in [variable.label(i) for i in range(variable.domainSize())]:
                        raise ValueError(f"Invalid label '{val}' for variable '{var}'")
                    ie.addEvidence(var, variable.index(val))
                except gum.OutOfBounds:
                    print(f"Error: Invalid label '{val}' for variable '{var}'")
                    print(f"Valid labels for '{var}': {[self.model.variable(var).label(i) for i in range(self.model.variable(var).domainSize())]}")
                    raise
            elif isinstance(val, list):
                ie.addEvidence(var, val)
            else:
                raise ValueError(f"Unsupported evidence type for {var}: {type(val)}")

        try:
            ie.makeInference()
            decision_index = np.argmax(ie.posteriorUtility('Quality').toarray())
            decision = self.model.variable('Quality').label(int(decision_index))
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            decision = "Medium"  # Default decision in case of error

        return decision