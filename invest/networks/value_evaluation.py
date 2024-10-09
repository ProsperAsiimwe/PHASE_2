import os
import numpy as np
import pyAgrum as gum

class ValueNetwork:
    def __init__(self, learned_cpt=None):
        self.model = gum.InfluenceDiagram()

        # Decision node for Expensive_E
        expensive_decision = gum.LabelizedVariable('Expensive_E', '', 2)
        expensive_decision.changeLabel(0, 'No')
        expensive_decision.changeLabel(1, 'Yes')
        self.model.addDecisionNode(expensive_decision)

        # Decision node for ValueRelativeToPrice
        value_relative_to_price_decision = gum.LabelizedVariable('ValueRelativeToPrice', '', 3)
        value_relative_to_price_decision.changeLabel(0, 'Cheap')
        value_relative_to_price_decision.changeLabel(1, 'FairValue')
        value_relative_to_price_decision.changeLabel(2, 'Expensive')
        self.model.addDecisionNode(value_relative_to_price_decision)

        # Add chance nodes
        future_share_performance = gum.LabelizedVariable('FutureSharePerformance', '', 3)
        future_share_performance.changeLabel(0, 'Positive')
        future_share_performance.changeLabel(1, 'Stagnant')
        future_share_performance.changeLabel(2, 'Negative')
        self.model.addChanceNode(future_share_performance)

        pe_relative_market = gum.LabelizedVariable('PERelative_ShareMarket', '', 3)
        pe_relative_market.changeLabel(0, 'Cheap')
        pe_relative_market.changeLabel(1, 'FairValue')
        pe_relative_market.changeLabel(2, 'Expensive')
        self.model.addChanceNode(pe_relative_market)

        pe_relative_sector = gum.LabelizedVariable('PERelative_ShareSector', '', 3)
        pe_relative_sector.changeLabel(0, 'Cheap')
        pe_relative_sector.changeLabel(1, 'FairValue')
        pe_relative_sector.changeLabel(2, 'Expensive')
        self.model.addChanceNode(pe_relative_sector)

        forward_pe_current_vs_history = gum.LabelizedVariable('ForwardPE_CurrentVsHistory', '', 3)
        forward_pe_current_vs_history.changeLabel(0, 'Cheap')
        forward_pe_current_vs_history.changeLabel(1, 'FairValue')
        forward_pe_current_vs_history.changeLabel(2, 'Expensive')
        self.model.addChanceNode(forward_pe_current_vs_history)

        # Utility nodes
        utility_expensive = gum.LabelizedVariable('Expensive_Utility', '', 1)
        self.model.addUtilityNode(utility_expensive)

        utility_value_relative_to_price = gum.LabelizedVariable('VRP_Utility', '', 1)
        self.model.addUtilityNode(utility_value_relative_to_price)

        # Add arcs
        self.model.addArc(self.model.idFromName('FutureSharePerformance'), self.model.idFromName('PERelative_ShareMarket'))
        self.model.addArc(self.model.idFromName('FutureSharePerformance'), self.model.idFromName('PERelative_ShareSector'))
        self.model.addArc(self.model.idFromName('FutureSharePerformance'), self.model.idFromName('ForwardPE_CurrentVsHistory'))
        self.model.addArc(self.model.idFromName('FutureSharePerformance'), self.model.idFromName('Expensive_Utility'))
        self.model.addArc(self.model.idFromName('FutureSharePerformance'), self.model.idFromName('VRP_Utility'))

        self.model.addArc(self.model.idFromName('PERelative_ShareMarket'), self.model.idFromName('Expensive_E'))
        self.model.addArc(self.model.idFromName('PERelative_ShareMarket'), self.model.idFromName('ValueRelativeToPrice'))

        self.model.addArc(self.model.idFromName('PERelative_ShareSector'), self.model.idFromName('Expensive_E'))
        self.model.addArc(self.model.idFromName('PERelative_ShareSector'), self.model.idFromName('ValueRelativeToPrice'))

        self.model.addArc(self.model.idFromName('ForwardPE_CurrentVsHistory'), self.model.idFromName('ValueRelativeToPrice'))

        self.model.addArc(self.model.idFromName('Expensive_E'), self.model.idFromName('ForwardPE_CurrentVsHistory'))
        self.model.addArc(self.model.idFromName('Expensive_E'), self.model.idFromName('ValueRelativeToPrice'))
        self.model.addArc(self.model.idFromName('Expensive_E'), self.model.idFromName('Expensive_Utility'))

        self.model.addArc(self.model.idFromName('ValueRelativeToPrice'), self.model.idFromName('VRP_Utility'))

        # Set utilities
        self.model.utility(self.model.idFromName('Expensive_Utility'))[{'Expensive_E': 'Yes'}] = [[-300], [150], [200]]
        self.model.utility(self.model.idFromName('Expensive_Utility'))[{'Expensive_E': 'No'}] = [[350], [-150], [-200]]

        self.model.utility(self.model.idFromName('VRP_Utility'))[{'ValueRelativeToPrice': 'Cheap'}] = [[200], [-75], [-200]]
        self.model.utility(self.model.idFromName('VRP_Utility'))[{'ValueRelativeToPrice': 'FairValue'}] = [[100], [0], [-75]]
        self.model.utility(self.model.idFromName('VRP_Utility'))[{'ValueRelativeToPrice': 'Expensive'}] = [[-100], [100], [150]]

        # Set initial CPTs
        self.model.cpt(self.model.idFromName('FutureSharePerformance'))[0] = 0.44444  # Positive
        self.model.cpt(self.model.idFromName('FutureSharePerformance'))[1] = 0.14815  # Stagnant
        self.model.cpt(self.model.idFromName('FutureSharePerformance'))[2] = 0.40741  # Negative

        self.model.cpt(self.model.idFromName('PERelative_ShareMarket'))[{'FutureSharePerformance': 'Positive'}] = [0.70, 0.20, 0.10]
        self.model.cpt(self.model.idFromName('PERelative_ShareMarket'))[{'FutureSharePerformance': 'Stagnant'}] = [0.25, 0.50, 0.25]
        self.model.cpt(self.model.idFromName('PERelative_ShareMarket'))[{'FutureSharePerformance': 'Negative'}] = [0.10, 0.20, 0.70]

        self.model.cpt(self.model.idFromName('PERelative_ShareSector'))[{'FutureSharePerformance': 'Positive'}] = [0.70, 0.20, 0.10]
        self.model.cpt(self.model.idFromName('PERelative_ShareSector'))[{'FutureSharePerformance': 'Stagnant'}] = [0.25, 0.50, 0.25]
        self.model.cpt(self.model.idFromName('PERelative_ShareSector'))[{'FutureSharePerformance': 'Negative'}] = [0.10, 0.20, 0.70]

        self.model.cpt(self.model.idFromName('ForwardPE_CurrentVsHistory'))[{'Expensive_E': 'Yes'}] = [[0.20, 0.30, 0.50], [0.20, 0.50, 0.30], [0.10, 0.17, 0.75]]
        self.model.cpt(self.model.idFromName('ForwardPE_CurrentVsHistory'))[{'Expensive_E': 'No'}] = [[0.70, 0.20, 0.10], [0.15, 0.70, 0.15], [0.20, 0.60, 0.20]]

        if learned_cpt:
            self.update_cpts(learned_cpt)

    def update_cpts(self, learned_bn):
        if learned_bn is None:
            print("No learned BN provided. Using original CPTs.")
            return

        for node in self.model.nodes():
            if self.model.isChanceNode(node):
                var_name = self.model.variable(node).name()
                if var_name in learned_bn.names():
                    self.model.cpt(node).fillWith(learned_bn.cpt(learned_bn.idFromName(var_name)))
                    print(f"Updated CPT for {var_name}")
                else:
                    print(f"Learned BN does not contain variable {var_name}. Keeping original CPT.")

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

    def normalize_label(self, label):
        label_map = {
            'cheap': 'Cheap',
            'fairvalue': 'FairValue',
            'expensive': 'Expensive',
            'positive': 'Positive',
            'stagnant': 'Stagnant',
            'negative': 'Negative'
        }
        return label_map.get(label.lower(), label)

    def normalize_evidence(self, evidence):
        normalized = {}
        for var, val in evidence.items():
            if pd.isna(val):
                continue  # Skip NaN values
            if isinstance(val, str):
                normalized[var] = self.normalize_label(val)
            elif val is not None:
                normalized[var] = val
        return normalized

    def make_decision(self, evidence):
        ie = gum.ShaferShenoyLIMIDInference(self.model)
        ie.addNoForgettingAssumption(['Expensive_E', 'ValueRelativeToPrice'])

        normalized_evidence = self.normalize_evidence(evidence)

        for var, val in normalized_evidence.items():
            if val is None:
                continue  # Skip None values
            elif isinstance(val, str):
                try:
                    ie.addEvidence(var, self.model.variable(var).index(val))
                except gum.OutOfBounds:
                    print(f"Error: Invalid label '{val}' for variable '{var}'")
                    print(f"Valid labels for '{var}': {[self.model.variable(var).label(i) for i in range(self.model.variable(var).domainSize())]}")
                    raise
            elif isinstance(val, list):
                ie.addEvidence(var, val)
            else:
                raise ValueError(f"Unsupported evidence type for {var}: {type(val)}")

        ie.makeInference()
        decision_index = np.argmax(ie.posteriorUtility('ValueRelativeToPrice').toarray())
        decision = self.model.variable('ValueRelativeToPrice').label(int(decision_index))

        # Forced Decisions logic
        if decision == 'Expensive':
            pe_relative_market_state = normalized_evidence.get('PERelative_ShareMarket')
            pe_relative_sector_state = normalized_evidence.get('PERelative_ShareSector')
            forward_pe_current_vs_history_state = normalized_evidence.get('ForwardPE_CurrentVsHistory')

            if (pe_relative_market_state == "Cheap" and pe_relative_sector_state == "Expensive") or \
               (pe_relative_market_state == "Expensive" and pe_relative_sector_state == "Cheap") or \
               (pe_relative_market_state == "FairValue" and pe_relative_sector_state == "FairValue" and 
                forward_pe_current_vs_history_state == "FairValue"):
                return 'FairValue'

        return decision