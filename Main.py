from sentence_transformers import SentenceTransformer , util
import numpy as np

def calculateSimilarity(filePath1, filePath2):

    model = SentenceTransformer('all-MiniLM-L6-v2')

    with open(filePath1, 'r', encoding='utf-8') as f1, open(filePath2, 'r' , encoding='utf-8') as f2:
        pargraphs1 = f1.read().split('\n\n')
        pargraphs2 = f2.read().split('\n\n')

    file1Embeddings = model.encode(pargraphs1 , convert_to_tensor=True)
    file2Embeddings = model.encode(pargraphs2 , convert_to_tensor=True)

    scores = []
    for emb1 in file1Embeddings:
        score = util.cos_sim(emb1, file2Embeddings).max().item()
        scores.append(score)

    similarity = np.mean(scores) * 100
    if similarity <= 0:
        similarity = 0
    return similarity


path1 = 'text1.txt'
path2 = 'text2.txt'

similarity = calculateSimilarity(path1, path2)
print(f'Similarity score between two books is : {similarity}')