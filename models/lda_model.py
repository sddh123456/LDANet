import numpy as np
from sklearn.decomposition import LatentDirichletAllocation

class ColorLDAModel:
    def __init__(self, n_topics=10):
        self.lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42
        )
        
    def fit(self, color_documents):
        return self.lda.fit(color_documents)
    
    def transform(self, color_documents):
        if isinstance(color_documents, list):
            color_documents = np.array(color_documents)
            color_documents = color_documents.reshape(-1, color_documents.shape[-1])    
        return self.lda.transform(color_documents)