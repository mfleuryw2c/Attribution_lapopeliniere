import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

class ImpostorsVerifier:
    def __init__(self, n_iterations=100, feature_fraction=0.5):
        self.n_iterations = n_iterations
        self.feature_fraction = feature_fraction
        
        # We use character 3-grams and 4-grams to capture subconscious stylistic habits
        self.vectorizer = TfidfVectorizer(
            analyzer='char', 
            ngram_range=(3, 4), 
            max_features=1000
        )

    def verify(self, known_texts, questioned_text, impostor_texts):
        """
        Returns the probability that the questioned_text was written by the author 
        of the known_texts, evaluated against an impostor pool.
        """
        # Combine all texts to build the global feature vocabulary
        all_texts = known_texts + questioned_text + impostor_texts
        tfidf_matrix = self.vectorizer.fit_transform(all_texts).toarray()
        
        # Slice the matrix back into its respective components
        n_known = len(known_texts)
        known_vectors = tfidf_matrix[:n_known]
        questioned_vector = tfidf_matrix[n_known]
        impostor_vectors = tfidf_matrix[n_known + 1:]
        
        # Aggregate known texts into a single average profile for Author X
        author_x_profile = np.mean(known_vectors, axis=0)
        
        n_features = tfidf_matrix.shape[1]
        features_to_select = int(n_features * self.feature_fraction)
        
        author_x_wins = 0
        
        # Monte Carlo bootstrapping over feature subsets
        for _ in range(self.n_iterations):
            # 1. Randomly sample a subset of features
            sampled_features = np.random.choice(n_features, features_to_select, replace=False)
            
            # 2. Extract vectors for this specific iteration
            q_sub = questioned_vector[sampled_features].reshape(1, -1)
            x_sub = author_x_profile[sampled_features].reshape(1, -1)
            imp_sub = impostor_vectors[:, sampled_features]
            
            # 3. Calculate Cosine Distance
            dist_to_x = cosine_distances(q_sub, x_sub)[0][0]
            dist_to_impostors = cosine_distances(q_sub, imp_sub)[0]
            
            # 4. Check if Author X is closer than ANY impostor
            if dist_to_x < np.min(dist_to_impostors):
                author_x_wins += 1
                
        # Calculate final probability score
        probability = author_x_wins / self.n_iterations
        return probability

