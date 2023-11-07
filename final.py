#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import math
import ast
import torchaudio
import openai


# In[8]:


def chat_gpt(query):
  if not hasattr(chat_gpt, 'cnt'):
    chat_gpt.cnt = -1
  if not hasattr(chat_gpt, 'df'):
    chat_gpt.df = pd.Series([])

  chat_gpt.cnt += 1

  openai.api_key = 'your_api_key'

  response = openai.ChatCompletion.create(
     model='gpt-3.5-turbo',
      messages=[
          {'role': 'user', 'content': '''

         When a tourist attraction is suggested, Please provide a description focused on the characteristics of each tourist attraction.

  Here's an example that best represents the features of New York as a tourist destination:
  1. The Statue of Liberty. Located in New York Harbor, this massive statue symbolizes America's freedom and independence.
  2. Times Square. One of the most famous squares in the world, it is a symbolic place of New York.
  4. Empire State Building. One of the most famous landmarks in the world, it provides a beautiful view.
  5. Brooklyn Bridge. An iconic bridge connecting Brooklyn and Manhattan, it offers beautiful architecture and scenery.
  6. Museums. New York is rich in world-renowned museums, including the Metropolitan Museum, Guggenheim Museum, art galleries, and other museums.
  7. The fresh smell of grass in Central Park, the scent of fallen leaves in autumn, and the fragrance of blooming flowers in spring add a touch of nature to the city. Also, the fresh sea smell coming from the Hudson River momentarily makes you forget the noise and hustle of the city.

          '''},
          {'role': 'user', 'content': query}
          ])

  answer = response.choices[0].message.content

  df = pd.DataFrame([answer], columns=['input'])
  return df


# In[10]:


df = chat_gpt('삿포로')


# # Preprocessing

# Clean Text

# In[12]:


from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
import re


# In[11]:


import nltk
nltk.download('stopwords')
nltk.download('punkt')



# In[24]:


# 어간 추출
def stem_words(text):
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)
    return text


# 소문자 변환
def make_lower_case(text):
    return " ".join([word.lower() for word in text.split()])

# 불용어 제거
def remove_stop_words(text):
    stops = set(stopwords.words("english"))
    text = [w for w in text.split() if not w in stops] 
    text = " ".join(text)
    return text


# 따옴표 제거
def remove_quotes(text):
    if isinstance(text, str):
        return text.replace('"', "")
    elif isinstance(text, list):
        return ["".join(t).replace('"', "") for t in text]
    else:
        return text

# 쉼표 제거
def remove_commas(text):
    if isinstance(text, list):
        return [t.rstrip(',') for t in text]
    else:
        return text

        import re

#특수문자 및 숫자 제거
def remove_special_characters(text):
    text = re.sub(r'[:.]|\d', '', text)
    return text

def preprocess_text(text):
    text = remove_special_characters(text)
    text = make_lower_case(text)
    text = remove_stop_words(text)
    text = stem_words(text.split()) # split()을 사용하여 단어 단위로 분리한 후 어간 추출
    text = remove_quotes(text)
    text = remove_commas(text)
    return text


# In[25]:


#전처리 함수 적용 
df['input_pre'] = df['input'].apply(preprocess_text)


# In[26]:


df['input_pre']


# # Word2vec imbedding (using TF-IDF)

# TF-IDF 벡터화를 이용하면 단어의 빈도가 높고 문서 간에 잘 분포되어 있는 단어일수록 높은 가중치를 가지게 된다. 이렇게 계산된 TF-IDF 가중치를 Word2Vec 임베딩 벡터에 곱하면, 가중치가 큰 단어의 임베딩 벡터 값이 더 크게 반영되어  모델 성능의 향상을 기대할 수 있을 것이다
# .

# ## TF-IDF 가중치 계산

# In[28]:


import pickle


# In[39]:


#훈련된 pickle 로드 
with open('./Data/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)


# In[40]:


from sklearn.feature_extraction.text import TfidfVectorizer


# 'input_pre' 컬럼으로부터 문장 리스트 생성
sentences = df['input_pre'].tolist()

# TF-IDF 벡터화
tfidf_matrix = vectorizer.fit_transform(sentences)

# 단어별 TF-IDF 가중치 계산
word2weight = {word: tfidf_matrix.getcol(idx).sum() for word, idx in vectorizer.vocabulary_.items()}


# In[41]:


#top10 가중치 확인 
top_10_weights = sorted(word2weight.items(), key=lambda x: x[1], reverse=True)[:10]

for word, weight in top_10_weights:
    print(f'단어: {word}, 가중치: {weight}')


# ## Word2vec을 통한 연관단어 임베딩

# In[42]:


#훈련된 pickle 로드 
with open('./Data/word2vec_model.pkl', 'rb') as f:
     loaded_model = pickle.load(f)


# In[43]:


from gensim.models import Word2Vec
import numpy as np

# 단어 임베딩 추출
word_vectors = loaded_model.wv

# TF-IDF 가중치 적용하여 임베딩 벡터 추출
X = []
for sentence in sentences:
    embedding = []
    for word in sentence:
        if word in word_vectors.key_to_index:  # 단어가 Word2Vec 모델에 있는지 확인
            # TF-IDF 가중치 적용하여 단어 임베딩 계산 (단어가 없으면 기본값 1.0 사용)
            weighted_embedding = word_vectors.get_vector(word) * word2weight.get(word, 1.0)
            embedding.append(weighted_embedding)
    # 각 문장에 대해 단어 임베딩의 평균 계산 (단어가 하나도 없는 경우 제외)
    if embedding:
        average_embedding = np.mean(embedding, axis=0)
        X.append(average_embedding)


# In[44]:


X


# # Predict Cluster
# 향수 군집과의 연결을 위해 최적의 Cluster를 예측한다.

# In[45]:


with open('./Data/kmeans_model.pkl', 'rb') as f:
    kmeans_model = pickle.load(f)


# In[49]:


#최적의 Cluster 예측 
cluster = kmeans_model.predict(X)
cluster


# # 유사도 계산 및 Top 3 출력

# In[56]:


#임베딩된 향수 데이터 pkl 로드
with open('./Data/embedding_result_perfume.pkl', 'rb') as f:
    perfume_embeddings = pickle.load(f)

#군집화된 향수 데이터베이스 로드 
perfume = pd.read_csv('./Data/Result_Clustering.csv', index_col = 0)


# In[74]:


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans


# 각 여행지에 대해 동일한 클러스터 레이블을 가진 향수의 임베딩만 선택
top3_perfumes = []

# perfume 데이터프레임에 'Embedding' 컬럼 추가
perfume['Embedding'] = perfume_embeddings

for i, label in enumerate(cluster):
    # 동일한 클러스터 향수 추출
    same_cluster_perfumes = perfume[perfume['Cluster'] == label]
    
    # 동일한 클러스터 향수의 임베딩 추출
    same_cluster_perfume_embeddings = same_cluster_perfumes['Embedding'].apply(pd.Series).values
    
    # 입력 데이터(여행지) 임베딩벡터와 동일한 클러스터를 가진 향수 임베딩 벡터간의 코사인 유사도 계산
    cosine_similarities = cosine_similarity(X[i].reshape(1, -1), same_cluster_perfume_embeddings)
    
    #유사도가 가장 높은 Top3 향수 선택
    top3_indices = cosine_similarities[0].argsort()[-3:][::-1]
    top3_perfumes.append(same_cluster_perfumes.iloc[top3_indices])

# 리스트 -> 데이터프레임 
top3_perfumes_df = pd.concat(top3_perfumes, ignore_index=True)

#불필요한 피처 제거 
del top3_perfumes_df['Cluster']
del top3_perfumes_df['Embedding']

#최종 Top 3출력 
top3_perfumes_df


# In[ ]:




