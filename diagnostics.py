"""
1. Your class should make use of class BayesNet in the textbook code. 
   You will create your BayesNet in the constructor. 
   You should not use any other Bayesian network library.
2. Use the enumeration inference algorithm to calculate the probability of each disease. 
   This algorithm is implemented as the enumeration_ask function  in the textbook code.
3. The textbook class BayesNet only handles Boolean variables. 
   So, you will have to write code to convert Yes/No, Abnormal/Normal, Present/Absent to True/False.

"""


from probability4e import *

T, F = True, False

class Diagnostics:
    """ Use a Bayesian network to diagnose between three lung diseases """

    def __init__(self):
        # Build the BayesNet (parents must come before children)
        self.bn = BayesNet([
            # Priors
            ('Asia', '', 0.01),            # P(Asia=True) = 0.01
            ('Smoking', '', 0.5),          # P(Smoking=True) = 0.5

            # Conditionals
            ('TB', 'Asia', {T: 0.05, F: 0.01}),             # P(TB=True | Asia)
            ('Cancer', 'Smoking', {T: 0.10, F: 0.01}),      # P(Cancer=True | Smoking)
            ('Bronchitis', 'Smoking', {T: 0.60, F: 0.30}),  # P(Bronchitis=True | Smoking)

            # Deterministic OR node: TBorC = TB OR Cancer
            ('TBorC', 'TB Cancer', {
                (T, T): 1.0,
                (T, F): 1.0,
                (F, T): 1.0,
                (F, F): 0.0
            }),

            # Symptoms/tests
            ('XRay', 'TBorC', {T: 0.99, F: 0.05}),  # P(XRay=True | TBorC) where True = Abnormal
            ('Dyspnea', 'TBorC Bronchitis', {       # True = Present
                (T, T): 0.9,
                (T, F): 0.7,
                (F, T): 0.8,
                (F, F): 0.1
            })
        ])

    def _to_bool(self, value: str, true_token: str, false_token: str):
        """Convert a GUI string to True/False, or return None if NA/unknown."""
        if value is None:
            return None
        v = value.strip()
        if v == "NA":
            return None
        if v == true_token:
            return True
        if v == false_token:
            return False
        # If GUI ever passes unexpected strings, treat as unknown
        return None


    def diagnose (self, asia, smoking, xray, dyspnea):
        # To be implemented by the student
        """
        Inputs are strings:
          asia: Yes, No, NA
          smoking: Yes, No, NA
          xray: Abnormal, Normal, NA
          dyspnea: Present, Absent, NA

        Returns:
          [most_likely_disease_name ("TB"/"Cancer"/"Bronchitis"), probability]
        """

         # Build evidence dictionary for the BayesNet
        evidence = {}

        asia_b = self._to_bool(asia, "Yes", "No")
        if asia_b is not None:
            evidence['Asia'] = asia_b

        smoking_b = self._to_bool(smoking, "Yes", "No")
        if smoking_b is not None:
            evidence['Smoking'] = smoking_b

        xray_b = self._to_bool(xray, "Abnormal", "Normal")
        if xray_b is not None:
            evidence['XRay'] = xray_b

        dysp_b = self._to_bool(dyspnea, "Present", "Absent")
        if dysp_b is not None:
            evidence['Dyspnea'] = dysp_b

        # Compute posterior probability for each disease being True
        p_tb = enumeration_ask('TB', evidence, self.bn)[True]
        p_cancer = enumeration_ask('Cancer', evidence, self.bn)[True]
        p_bronchitis = enumeration_ask('Bronchitis', evidence, self.bn)[True]

        # Choose the most likely disease (tie-breaker: TB > Cancer > Bronchitis)
        candidates = [
            ("TB", p_tb),
            ("Cancer", p_cancer),
            ("Bronchitis", p_bronchitis),
        ]
        best_name, best_prob = max(candidates, key=lambda x: x[1])

        return [best_name, float(best_prob)]
        #return ["the disease", -1.0] # placeholder return value, to be replaced by the student



