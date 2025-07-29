from typing import Optional, Literal, Union
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import fasttext
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from fuzzywuzzy import fuzz
from main import load_csv_files, read_IND, evaluate
import os
import logging
import time


# Type alias for similarity methods
SimilarityMethod = Literal["fastText", "sentenceTransformer", "TF_IDF", "partial_ratio"]

class SimilarityCalculator:
    """
    A class to calculate text similarity using multiple methods.

    Parameters
    ----------
    sentence_transformer_model_name : str, optional
        Name of the sentence transformer model to use (default: 'all-MiniLM-L6-v2')
    fast_model_name : str, optional
        Path to the FastText model file (default: local path to cc.en.300.bin)

    Attributes
    ----------
    fast_text_model : fasttext.FastText._FastText
        Loaded FastText model instance
    sentence_transformer_model : SentenceTransformer
        Loaded Sentence Transformer model instance
    tfidf_vectorizer : TfidfVectorizer
        TF-IDF vectorizer instance configured for character n-grams
    """

    def __init__(
        self, 
        sentence_transformer_model_name: str = "all-MiniLM-L6-v2",
        fast_model_name: str = '/home/haseeb/Desktop/EKAI/ERD_automation/codes/my_automation/my_automation/cc.en.300.bin'
    ) -> None:
        self.fast_model_name: str = fast_model_name
        self.sentence_transformer_model_name: str = sentence_transformer_model_name
        
        self.fast_text_model: fasttext.FastText._FastText = fasttext.load_model(self.fast_model_name) 
        self.sentence_transformer_model: SentenceTransformer = SentenceTransformer(self.sentence_transformer_model_name)
        self.tfidf_vectorizer: TfidfVectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
        
    def fastTextModelSimilarity(self, dependent: str, reference: str) -> np.ndarray:
        """
        Calculate similarity between texts using FastText embeddings.

        Parameters
        ----------
        dependent : str
            First text for comparison
        reference : str
            Second text for comparison

        Returns
        -------
        numpy.ndarray
            Cosine similarity score between the text embeddings
        """
        emb1: np.ndarray = self.fast_text_model.get_word_vector(dependent)
        emb2: np.ndarray = self.fast_text_model.get_word_vector(reference)

        emb1_expanded: np.ndarray = np.expand_dims(emb1, 0)
        emb2_expanded: np.ndarray = np.expand_dims(emb2, 0)
        
        similarity: np.ndarray = cosine_similarity(emb1_expanded, emb2_expanded)

        return similarity
    
    def sentence_transformer_similarity(self, dependent: str, reference: str) -> np.ndarray:
        """
        Calculate similarity between texts using Sentence Transformers.

        Parameters
        ----------
        dependent : str
            First text for comparison
        reference : str
            Second text for comparison

        Returns
        -------
        numpy.ndarray
            Cosine similarity score between the text embeddings
        """
        emb1: np.ndarray = self.sentence_transformer_model.encode(dependent).reshape(1, -1)
        emb2: np.ndarray = self.sentence_transformer_model.encode(reference).reshape(1, -1)
        similarity: np.ndarray = cosine_similarity(emb1, emb2)

        return similarity 

    def TF_IDF_similarity(self, dependent: str, reference: str) -> np.ndarray:
        """
        Calculate similarity between texts using TF-IDF vectorization.

        Parameters
        ----------
        dependent : str
            First text for comparison
        reference : str
            Second text for comparison

        Returns
        -------
        numpy.ndarray
            Cosine similarity score between the TF-IDF vectors
        """
        tfidf_matrix = self.tfidf_vectorizer.fit_transform([dependent] + [reference])
        similarities: np.ndarray = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])

        return similarities
    
    def partial_ratio(self, dependent, reference):
        """
        Calculate partial string ratio similarity between texts using fuzzywuzzy.

        This method finds the best matching substring and calculates similarity,
        which is particularly useful when one string is a substring of another.

        Parameters
        ----------
        dependent : str
            First text for comparison
        reference : str
            Second text for comparison

        Returns
        -------
        float
            Similarity score between 0 and 1, where 1 indicates perfect match
        """
        
        similarity = fuzz.partial_ratio(dependent, reference) / 100
        return similarity
    
    def compare(self, method: SimilarityMethod, dependent: str, reference: str) -> Union[np.ndarray, float]:
        """
        Compare texts using the specified similarity method.

        Parameters
        ----------
        method : SimilarityMethod
            The similarity method to use. Must be one of:
            - "fastText"
            - "sentenceTransformer"
            - "TF_IDF"
            - "partial_ratio"
        dependent : str
            First text for comparison
        reference : str
            Second text for comparison

        Returns
        -------
        Union[numpy.ndarray, float]
            Similarity score between the texts using the specified method

        Raises
        ------
        ValueError
            If the specified method is not supported
        """
        method_map: dict[str, callable] = {
            "fastText": self.fastTextModelSimilarity,
            "sentenceTransformer": self.sentence_transformer_similarity,
            "TF_IDF": self.TF_IDF_similarity,
            "partial_ratio": self.partial_ratio
        }
        
        if method not in method_map:
            raise ValueError(f"Method '{method}' not supported.")
            
        result = method_map[method](dependent=dependent, reference=reference)
        return result.squeeze() if isinstance(result, np.ndarray) else result
    
def main():
    db_name=os.path.basename(CSV_DIR)
    logging.basicConfig(
                    filename=os.path.join(os.path.dirname(os.path.abspath('')),"my_automation","logs",f"{db_name}_similarity_method_evaluation.log"),
                    encoding="utf-8",
                    filemode="w",
                    format="{asctime} - {levelname} - {message}",
                    style="{",
                    datefmt="%Y-%m-%d %H:%M",
                    level=logging.INFO
    )

    attributes = load_csv_files(CSV_DIR)
    inds, total_inds = read_IND(SPIDER_IND_RESULT, attributes=attributes)

    SimilarityCalculator = SimilarityCalculator()   

    for method in METHODS:
        start = time.time()
        logging.info(f"{"-"*50} Evaluating {method} {"-"*50}")
        candidates = []
        for ind in inds:
            dependent_name = ind.dependent.attribute_name
            reference_name = ind.reference.attribute_name
            similarity = SimilarityCalculator.compare(method=method, dependent=dependent_name, reference=reference_name)
            if similarity > 0.7:
                candidates.append(ind)
            # break
            
        # break
        end = time.time()
        exec_time = end-start
        logging.info(f"Execution time: {exec_time}")
        evaluate(GT_PATH, candidates)

if "__main__" == "__main__":
    CSV_DIR = "/home/haseeb/Desktop/EKAI/ERD_automation/Dataset/train/northwind"
    SPIDER_IND_RESULT = "/home/haseeb/Desktop/EKAI/ERD_automation/codes/inclusionDependencyWithSpider/spider_results/northwind.txt"
    GT_PATH = "/home/haseeb/Desktop/EKAI/ERD_automation/Dataset/ground_truth/northwind.json"
    METHODS = ["fastText", "sentenceTransformer", "TF_IDF", "partial_ratio"]

    main()