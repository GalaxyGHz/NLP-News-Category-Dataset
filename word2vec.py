import numpy as np
import pandas as pd
import plotly.express as px
from gensim.models import Word2Vec
from sklearn.decomposition import PCA

dataset = pd.read_json(r'data\preprocess_data.json', lines=True)

tokenized_data = [data.split() for data in dataset['clean_text']]

vector_size = 128

w2v_model = Word2Vec(tokenized_data, vector_size=vector_size, window=7, min_count=1, workers=16, epochs=16)

def w2v_of_data_points(data):
    sumation = 0
    count = 0
    for data_point in data:
        if data_point in w2v_model.wv:
            sumation += w2v_model.wv[data_point]
            count += 1
    if count != 0:
        return sumation / count
    else:
        return [0] * vector_size  # Return zero vector if no word found
    
dataset['word2vec'] = dataset.apply(lambda x: w2v_of_data_points(x["clean_text"]), axis=1)
dataset.to_json(r"data\word2vec_data.json", orient="records", lines=True)

mat = []
for idx, row in dataset.iterrows():
    mat.append(row['word2vec'])
matrix = np.array(mat)

pca = PCA(n_components=2)
word_vecs_2d = pca.fit_transform(mat)

data = {
    'x': word_vecs_2d[:, 0],
    'y': word_vecs_2d[:, 1],
    'category': dataset['category']
}

df = pd.DataFrame(data)

# Plotly scatter plot with hover text
fig = px.scatter(df, x='x', y='y', color='category',
                 title='Word2Vec News Article Embeddings', labels={'x': '', 'y': ''},
                 width=800, height=600)

fig.show()