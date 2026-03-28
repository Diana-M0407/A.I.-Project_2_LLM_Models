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

# Read the Gemini API key from the environment
API_KEY = os.getenv("GEMINI_API_KEY")

# Only these disease names are allowed in the final answer
ALLOWED_DISEASES = {"TB", "Cancer", "Bronchitis"}


class Diagnostics:
    """Use an LLM to diagnose between three lung diseases."""

    def __init__(self):
        # Create the Gemini client and choose the model
        self.client = genai.Client(api_key=API_KEY)
        self.model = "gemini-2.5-flash"

        print("Diagnostics object created.")
        print(f"Using model: {self.model}")

    def _to_bool(self, value: str, true_token: str, false_token: str):
        """
        Convert GUI string values into True/False when possible.
        Returns None if the value is NA or unknown.
        """
        print(f"Converting value '{value}' using tokens '{true_token}'/'{false_token}'")

        if value is None:
            print("Value is None -> returning None")
            return None

        v = value.strip()

        if v == "NA":
            print("Value is NA -> returning None")
            return None
        if v == true_token:
            print("Matched true token -> returning True")
            return True
        if v == false_token:
            print("Matched false token -> returning False")
            return False

        print("Unexpected value -> returning None")
        return None

    def _safe_parse_response(self, text):
        """
        Parse Gemini's response safely.

        First try to parse the full response as JSON.
        If that fails, try extracting the last {...} block
        from the response and parse that instead.
        """
        print("\n--- Parsing model response ---")
        print("Raw text received from model:")
        print(text)

        cleaned = text.strip()
        print("\nCleaned response text:")
        print(cleaned)

        # First attempt: parse the entire response directly as JSON
        try:
            data = json.loads(cleaned)
            print("\nParsed full response directly as JSON.")
        except json.JSONDecodeError:
            print("\nFull response was not valid JSON.")
            print("Attempting to extract the last JSON object...")

            start = cleaned.rfind("{")
            end = cleaned.rfind("}")

            if start != -1 and end != -1 and end > start:
                json_part = cleaned[start:end + 1]
                print("\nExtracted JSON portion:")
                print(json_part)

                try:
                    data = json.loads(json_part)
                    print("Successfully parsed extracted JSON portion.")
                except json.JSONDecodeError:
                    print("JSON parse failed even after extraction.")
                    return ["TB", 0.0]
            else:
                print("Could not find a JSON object in the response.")
                return ["TB", 0.0]

        # Extract the disease and probability fields
        disease = data.get("disease")
        prob = data.get("probability")

        print(f"\nParsed disease: {disease}")
        print(f"Parsed probability: {prob}")

        # Validate the disease name
        if disease not in ALLOWED_DISEASES:
            print("Disease name was invalid. Defaulting to TB.")
            disease = "TB"

        # Convert probability to float safely
        try:
            prob = float(prob)
        except (TypeError, ValueError):
            print("Probability was invalid. Defaulting to 0.0.")
            prob = 0.0

        # Clamp probability to the valid range [0, 1]
        prob = max(0.0, min(1.0, prob))

        print(f"Final parsed result: [{disease}, {prob}]")
        print("--- End parsing ---\n")

        return [disease, prob]

    def diagnose(self, visit_to_asia, smoking, xray_result, dyspnea):
        """
        Use Gemini to determine the most likely disease and probability
        based on the Bayesian network description and the patient's evidence.
        """
        print("=== Starting diagnosis ===")
        print(f"Visit to Asia: {visit_to_asia}")
        print(f"Smoking: {smoking}")
        print(f"XRay result: {xray_result}")
        print(f"Dyspnea: {dyspnea}")

        # Optional conversion step kept from Project 1 structure
        # so the code still demonstrates how the GUI inputs map
        # into boolean/unknown-style values.
        asia_bool = self._to_bool(visit_to_asia, "Yes", "No")
        smoking_bool = self._to_bool(smoking, "Yes", "No")
        xray_bool = self._to_bool(xray_result, "Abnormal", "Normal")
        dyspnea_bool = self._to_bool(dyspnea, "Present", "Absent")

        print("\nConverted values:")
        print(f"Asia -> {asia_bool}")
        print(f"Smoking -> {smoking_bool}")
        print(f"XRay -> {xray_bool}")
        print(f"Dyspnea -> {dyspnea_bool}")

        # Prompt the LLM with the full Bayesian network and current evidence
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

        print("\nPrompt being sent to Gemini:")
        print(prompt)

        # Send the prompt to Gemini
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )

        print("\nRaw Gemini response:")
        print(response.text)

        # Extract the text and parse it into [disease, probability]
        text = (response.text or "").strip()
        result = self._safe_parse_response(text)

        print(f"Diagnosis result returned to GUI: {result}")
        print("=== End diagnosis ===\n")

        return result



