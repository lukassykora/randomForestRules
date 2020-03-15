import unittest
import pandas as pd
from randomForestRules import RandomForestRules


class TestRandomForestRules(unittest.TestCase):
    def test_fit(self):
        randomForestRules = RandomForestRules()
        randomForestRules.load_pandas(self._get_data_frame_input_data())
        antecedent = ["Age",
                       "Embarked",
                       "Fare",
                       "Pclass"]
        consequent = "Survived"
        randomForestRules._prepare_data(antecedent, consequent)
        randomForestRules._set_dummies()
        data_dummies = randomForestRules.data_dummies
        # Number of columns should be the same = 9
        # Number of columns should be:
        #   Age: 3 categories
        #   Embarked: 2 categories
        #   Fare: 3 categories
        #   Pclass: 3 categories
        # Total number of categories = 3 + 2 + 3 + 3 = 11
        self.assertEqual((9, 11), data_dummies.shape)

        rf = randomForestRules._get_random_forest(4, random_state = 42)
        trees = randomForestRules._get_trees(rf)
        # Four trees - description in _get_trees() method
        self.assertEqual(self._get_trees(), trees)

        tree_rules = randomForestRules._get_tree_rules(trees)
        # Rules from trees
        self.assertEqual(self._get_tree_rules(), tree_rules)

        no_dummies_rules = randomForestRules._get_no_dummies_rules(tree_rules)
        # No dummies rules from trees
        self.assertEqual(self._get_no_dummies_rules(), no_dummies_rules)

        divided_rules = randomForestRules._get_divided_rules(no_dummies_rules)
        # Always just one category in the rule
        self.assertEqual(self._get_divided_rules(), divided_rules)

        rules_frame = randomForestRules._get_rules_frame(divided_rules)
        supp_conf = randomForestRules._get_supp_conf(divided_rules)
        self.assertEqual(self._get_supp_conf(), supp_conf)



    @staticmethod
    def _get_data_frame_input_data():
        new_data = [
            ['young', 'S', 'very high', '1', 1],
            ['young', 'S', 'high', '2', 1],
            ['young', 'C', 'low', '3', 0],
            ['middle', 'S', 'very high', '1', 1],
            ['middle', 'C', 'high', '2', 0],
            ['middle', 'S', 'low', '3', 0],
            ['old', 'S', 'very high', '1', 1],
            ['old', 'C', 'very high', '3', 0],
            ['old', 'S', 'high', '3', 0]
        ]
        df_new_data = pd.DataFrame(new_data, columns=["Age",
                                                      "Embarked",
                                                      "Fare",
                                                      "Pclass",
                                                      "Survived"])
        return df_new_data

    @staticmethod
    def _get_trees():
        # trees[tree_idx][node_idx] = [feature, th, left, right, class_idx]
        # Tree 0 has 3 nodes, node 0 makes the decision, nodes 1 and 2 represent the decision
        #        0
        #      /   \
        #     1     2

        trees = {0: {0: [8, 0.5, 1, 2, None],
                     1: [-2, -2.0, -1, -1, 0],
                     2: [-2, -2.0, -1, -1, 1]},
                 1: {0: [8, 0.5, 1, 2, None],
                     1: [-2, -2.0, -1, -1, 0],
                     2: [-2, -2.0, -1, -1, 1]},
                 2: {0: [8, 0.5, 1, 2, None],
                     1: [-2, -2.0, -1, -1, 0],
                     2: [-2, -2.0, -1, -1, 1]},
                 3: {0: [10, 0.5, 1, 4, None],
                     1: [4, 0.5, 2, 3, None],
                     2: [-2, -2.0, -1, -1, 0],
                     3: [-2, -2.0, -1, -1, 1],
                     4: [-2, -2.0, -1, -1, 0]}}
        return trees

    @staticmethod
    def _get_tree_rules():
        tree_rules = [
            [
                [
                    ['Pclass_1', 0], ['Survived', 0]
                ],
                [
                    ['Pclass_1', 1], ['Survived', 1]
                ]
            ],
            [
                [
                    ['Pclass_1', 0], ['Survived', 0]
                ],
                [
                    ['Pclass_1', 1], ['Survived', 1]
                ]
            ],
            [
                [
                    ['Pclass_1', 0], ['Survived', 0]
                ],
                [
                    ['Pclass_1', 1], ['Survived', 1]
                ]
            ],
            [
                [
                    ['Pclass_3', 1], ['Survived', 0]
                ],
                [
                    ['Pclass_3', 0], ['Embarked_S', 0], ['Survived', 0]
                ],
                [
                    ['Pclass_3', 0], ['Embarked_S', 1], ['Survived', 1]
                ]
            ]
        ]
        return tree_rules

    @staticmethod
    def _get_no_dummies_rules():
        no_dummies_rules = [
            {'Pclass': ['2', '3'], 'Survived': [0]},
            {'Pclass': ['1'], 'Survived': [1]},
            {'Pclass': ['2', '3'], 'Survived': [0]},
            {'Pclass': ['1'], 'Survived': [1]},
            {'Pclass': ['2', '3'], 'Survived': [0]},
            {'Pclass': ['1'], 'Survived': [1]},
            {'Pclass': ['3'], 'Survived': [0]},
            {'Embarked': ['C'], 'Pclass': ['1', '2'], 'Survived': [0]},
            {'Embarked': ['S'], 'Pclass': ['1', '2'], 'Survived': [1]}]
        return no_dummies_rules

    @staticmethod
    def _get_divided_rules():
        divided_rules = [
            {'Pclass': '2', 'Survived': 0},
            {'Pclass': '3', 'Survived': 0},
            {'Pclass': '1', 'Survived': 1},
            {'Pclass': '2', 'Survived': 0},
            {'Pclass': '3', 'Survived': 0},
            {'Pclass': '1', 'Survived': 1},
            {'Pclass': '2', 'Survived': 0},
            {'Pclass': '3', 'Survived': 0},
            {'Pclass': '1', 'Survived': 1},
            {'Pclass': '3', 'Survived': 0},
            {'Embarked': 'C', 'Pclass': '1', 'Survived': 0},
            {'Embarked': 'C', 'Pclass': '2', 'Survived': 0},
            {'Embarked': 'S', 'Pclass': '1', 'Survived': 1},
            {'Embarked': 'S', 'Pclass': '2', 'Survived': 1}]
        return divided_rules

    @staticmethod
    def _get_supp_conf():
        # For example the last rule (index 13) {'Embarked': 'S', 'Pclass': '2', 'Survived': 1}
        # How many times is it in source data? 1x (index 1)
        # How many times just antecedent? 1x (index 1)
        # Support = 1 / 9 = 0.11
        # Confidence = 1 / 1 = 1
        #
        # For example the second rule (index 1) {'Pclass': '3', 'Survived': 0}
        # How many times is it in source data? 4x
        # How many times just antecedent? 4x
        # Support = 4 / 9 = 0.44
        # Confidence = 4 / 4 = 1
        #
        # For example the first rule (index 0) {'Pclass': '2', 'Survived': 0}
        # How many times is it in source data? 1x
        # How many times just antecedent? 2x
        # Support = 1 / 9 = 0.11
        # Confidence = 1 / 2 = 0.5

        supp_conf = [
            [0.1111111111111111, 0.5],
            [0.4444444444444444, 1.0],
            [0.3333333333333333, 1.0],
            [0.1111111111111111, 0.5],
            [0.4444444444444444, 1.0],
            [0.3333333333333333, 1.0],
            [0.1111111111111111, 0.5],
            [0.4444444444444444, 1.0],
            [0.3333333333333333, 1.0],
            [0.4444444444444444, 1.0],
            [0.0, None],
            [0.1111111111111111, 1.0],
            [0.3333333333333333, 1.0],
            [0.1111111111111111, 1.0]]
        return supp_conf

if __name__ == '__main__':
    unittest.main()
