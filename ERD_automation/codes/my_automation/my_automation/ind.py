from fuzzywuzzy import fuzz
import os
import numpy as np
from openai import OpenAI
import logging
from sentence_transformers import SentenceTransformer, util

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class IND:
    """
    Represents an Inclusion Dependency (IND) between two attributes.

    Attributes:
        dependent: The dependent attribute in the IND relationship.
        reference: The reference attribute in the IND relationship.
        name_similarity: A method to calculate the similarity between the names of the dependent and reference attributes.

    Methods:
        __init__(dependent, reference):
            Initializes the IND object with the given dependent and reference attributes.

        _calculate_name_similarity():
            Calculates the similarity score between the attribute names of the dependent and reference attributes using fuzzy string matching.
            Returns:
                float: The similarity score as a value between 0 and 1.
    """
    def __init__(self, dependent, reference):
        self.dependent = dependent
        self.reference = reference
        self.candidate_confirmation = False
        self.name_similarity = self.calculate_name_similarity()

    def get_embedding(self, text, model="text-embedding-3-small"):
        response = client.embeddings.create(
            input=[text],
            model=model
        )
        return response.data[0].embedding

    def cosine_similarity(self, vec1, vec2):
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def vectorized_similarity_score(self):
        emb1 = self.get_embedding(self.dependent.fullName)
        emb2 = self.get_embedding(self.reference.fullName)
        score = self.cosine_similarity(emb1, emb2)
        return score

    def calculate_name_similarity(self): 
        """
        Calculates the similarity between the attribute names of the dependent and reference objects using the partial ratio method from the `fuzz` library.

        Returns:
            float: The similarity score as a float between 0 and 1.
        """
        similarity = fuzz.partial_ratio(self.dependent.attribute_name, self.reference.attribute_name) / 100
        if similarity > 0.8:
            self.candidate_confirmation = True
            logging.info(f"{self.reference.fullName}->{self.dependent.fullName} : partial_ratio score={similarity}")
            return similarity
        else:
            return similarity
            vector_similarity = self.vectorized_similarity_score()
            logging.info(f"{self.reference.fullName}->{self.dependent.fullName} : cosine similarity score={vector_similarity}")
            if vector_similarity>0.8:
                self.candidate_confirmation = True
            return vector_similarity