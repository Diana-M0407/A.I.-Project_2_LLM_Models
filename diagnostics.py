"""
Project 1: Bayesian Network

1. Your class should make use of class BayesNet in the textbook code. 
   You will create your BayesNet in the constructor. 
   You should not use any other Bayesian network library.
2. Use the enumeration inference algorithm to calculate the probability of each disease. 
   This algorithm is implemented as the enumeration_ask function  in the textbook code.
3. The textbook class BayesNet only handles Boolean variables. 
   So, you will have to write code to convert Yes/No, Abnormal/Normal, Present/Absent to True/False.

"""

"""
Project 2: LLM-based diagnosis

This class keeps the same Diagnostics interface from Project 1,
but now uses an LLM to solve the diagnosis problem instead of
calling Bayesian-network inference functions directly.
"""
import os
import json
from dotenv import load_dotenv
from google import genai

# Load environment variables from the .env file
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
ALLOWED_DISEASES = {"TB", "Cancer", "Bronchitis"}


class Diagnostics:
    """ Use a Bayesian network to diagnose between three lung diseases """

    def __init__(self):
        # Gemini setup
        self.client = genai.Client(api_key=API_KEY)
        self.model = "gemini-2.5-flash"

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

    def _safe_parse_response(self, text):
        #print(text)
        cleaned = text.strip()

        # First try: parse the whole response directly
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # If Gemini included explanation text, extract the last {...} block
            start = cleaned.rfind("{")
            end = cleaned.rfind("}")

            if start != -1 and end != -1 and end > start:
                json_part = cleaned[start:end + 1]
                #print(json_part)

                try:
                    data = json.loads(json_part)
                except json.JSONDecodeError:
                    print("JSON parse failed")
                    return ["TB", 0.0]
            else:
                print("JSON parse failed")
                return ["TB", 0.0]

        disease = data.get("disease")
        prob = data.get("probability")

        if disease not in ALLOWED_DISEASES:
            print("Invalid disease:", disease)
            disease = "TB"

        try:
            prob = float(prob)
        except (TypeError, ValueError):
            print("Invalid probability:", prob)
            prob = 0.0

        prob = max(0.0, min(1.0, prob))
        return [disease, prob]


    def diagnose (self, visit_to_asia, smoking, xray_result, dyspnea):
        # To be implemented by the student
        prompt = f"""
        You must solve this diagnosis problem using the following Bayesian network, not general medical knowledge.

        Variables:
        - Asia: Yes/No
        - Smoking: Yes/No
        - TB: Yes/No
        - Cancer: Yes/No
        - Bronchitis: Yes/No
        - TBorC = TB OR Cancer
        - XRay: Abnormal/Normal
        - Dyspnea: Present/Absent

        Bayesian network probabilities:
        P(Asia=Yes)=0.01
        P(Asia=No)=0.99

        P(Smoking=Yes)=0.5
        P(Smoking=No)=0.5

        P(TB=Yes | Asia=Yes)=0.05
        P(TB=Yes | Asia=No)=0.01

        P(Cancer=Yes | Smoking=Yes)=0.10
        P(Cancer=Yes | Smoking=No)=0.01

        P(Bronchitis=Yes | Smoking=Yes)=0.60
        P(Bronchitis=Yes | Smoking=No)=0.30

        TBorC is true exactly when TB or Cancer is true.

        P(XRay=Abnormal | TBorC=Yes)=0.98
        P(XRay=Abnormal | TBorC=No)=0.05

        P(Dyspnea=Present | TBorC=Yes, Bronchitis=Yes)=0.90
        P(Dyspnea=Present | TBorC=Yes, Bronchitis=No)=0.70
        P(Dyspnea=Present | TBorC=No, Bronchitis=Yes)=0.80
        P(Dyspnea=Present | TBorC=No, Bronchitis=No)=0.10

        Evidence for this patient:
        - Visit to Asia: {visit_to_asia}
        - Smoking: {smoking}
        - XRay: {xray_result}
        - Dyspnea: {dyspnea}

        Task:
        Compute which disease is most likely among exactly these three:
        TB, Cancer, Bronchitis

        Return ONLY valid JSON.
        Do not include explanations.
        Do not show calculations.
        Do not use markdown.
        Use exactly this format:
        {{
          "disease": "TB",
          "probability": 0.0
        }}
        """
        
        # new LLM interface

        # Choose the most likely disease (tie-breaker: TB > Cancer > Bronchitis)
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        print(response.text)

        text = (response.text or "").strip()

        return self._safe_parse_response(text)



