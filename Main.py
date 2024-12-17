from sentence_transformers import SentenceTransformer, util
import numpy as np
from langdetect import detect

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

def calculate_similarity(filePath1, filePath2):
    with open(filePath1, 'r', encoding='utf-8') as f1, open(filePath2, 'r', encoding='utf-8') as f2:
        text1 = f1.read()
        text2 = f2.read()
        paragraphs1 = text1.split('\n\n')
        paragraphs2 = text2.split('\n\n')

    lang1 = detect_language(text1)
    lang2 = detect_language(text2)

    if lang1 == "fa" or lang2 == "fa":
        model_name = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    else:
        model_name = 'all-MiniLM-L6-v2'

    model = SentenceTransformer(model_name)

    file1_embeddings = model.encode(paragraphs1, convert_to_tensor=True)
    file2_embeddings = model.encode(paragraphs2, convert_to_tensor=True)

    scores = []
    for emb1 in file1_embeddings:
        score = util.cos_sim(emb1, file2_embeddings).max().item()
        scores.append(score)

    similarity = np.mean(scores) * 100
    if similarity <= 0:
        similarity = 0
    return similarity

path1 = 'text1.txt'
path2 = 'text2.txt'

similarity = calculate_similarity(path1, path2)
print(f'Similarity score between two books is : {similarity}')