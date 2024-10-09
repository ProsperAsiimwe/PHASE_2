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
            var_name = self.model.variable(node).name()
            if var_name in learned_cpt.names() and self.model.isChanceNode(node):
                self.model.cpt(node).fillWith(learned_cpt.cpt(var_name))

    def print_variable_names(self):
        print(f"{self.__class__.__name__} Variables:")
        for node in self.model.nodes():
            var_name = self.model.variable(node).name()
            if self.model.isChanceNode(node):
                node_type = "Chance"
            elif self.model.isDecisionNode(node):
                node_type = "Decision"
            elif self.model.isUtilityNode(node):
                node_type = "Utility"
            else:
                node_type = "Unknown"
            print(f"{var_name} - {node_type}")

    def normalize_label(self, var, label):
        """Normalize label to match the model's labels for the specific variable."""
        label_map = {
            'Value': {
                'cheap': 'Cheap',
                'fairvalue': 'FairValue',
                'expensive': 'Expensive'
            },
            'Quality': {
                'high': 'High',
                'medium': 'Medium',
                'low': 'Low'
            },
            'Investable': {
                'yes': 'Yes',
                'no': 'No'
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

    def make_decision(self, value_decision, quality_decision):
        ie = gum.ShaferShenoyLIMIDInference(self.model)

        evidence = {
            'Value': value_decision,
            'Quality': quality_decision
        }
        normalized_evidence = self.normalize_evidence(evidence)
        print("Normalized investment evidence:", normalized_evidence)  # Debugging output

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
            else:
                raise ValueError(f"Unsupported evidence type for {var}: {type(val)}")

        try:
            ie.makeInference()
            decision_index = np.argmax(ie.posteriorUtility('Investable').toarray())
            decision = self.model.variable('Investable').label(int(decision_index))
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            decision = "No"  # Default decision in case of error

        return decision
