import numpy as np 

def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )

def mmr(query_embeddings, document_embeddings, relevance_scores, top_n=3, lambda_param=0.7):
    selected = []
    remaining = list(range(len(document_embeddings)))

    while remaining and len(selected) < top_n:
        best_index = None
        best_score = float("-inf")

        for index in remaining:
            relevance = relevance_scores[index]

            if not selected:
                diversity = 0
            
            else:
                diversity = max(
                    cosine_similarity(
                        document_embeddings[index],
                        document_embeddings[selected_index],
                    )
                    for selected_index in selected
                )
            
            score = (
                lambda_param * relevance
                - (1 - lambda_param) * diversity
            )

            if score > best_score:
                best_score = score
                best_index = index
        
        selected.append(best_index)
        remaining.remove(best_index)
    
    return selected
