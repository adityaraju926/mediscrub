import re
import numpy as np
from typing import List, Dict
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import pickle
import os

MEDICAL_KEYWORDS = [
    'diagnosis', 'treatment', 'medication', 'symptoms', 'patient',
    'doctor', 'hospital', 'medical', 'condition', 'disease',
    'prescription', 'dosage', 'blood pressure', 'heart rate',
    'temperature', 'lab results', 'test results', 'procedure',
    'surgery', 'therapy', 'recovery', 'follow-up', 'appointment'
]

def summarize_classical_ml(text, max_sentences=5):
    sentences = split_into_sentences(text)
    
    if len(sentences) <= max_sentences:
        return text
    
    tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2), min_df=2, max_df=0.8)
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    
    importance_scores = calculate_importance_scores(sentences, tfidf_matrix)
    
    top_indices = np.argsort(importance_scores)[-max_sentences:]
    top_indices = sorted(top_indices)
    
    summary_sentences = []
    for i in top_indices:
        summary_sentences.append(sentences[i])
    
    summary = ' '.join(summary_sentences)

    return summary

def split_into_sentences(text):
    sentences = re.split(r'[.!?]+', text)
    cleaned_sentences = []
    for s in sentences:
        if s.strip() and len(s.strip()) > 10:
            cleaned_sentences.append(s.strip())
    return cleaned_sentences

def calculate_importance_scores(sentences, tfidf_matrix):
    scores = np.zeros(len(sentences))
    
    tfidf_scores=tfidf_scoring(sentences, tfidf_matrix)
    scores+=tfidf_scores * 0.4
    
    position_scores=position_scoring(sentences)
    scores+=position_scores * 0.2
    
    keyword_scores=keyword_scoring(sentences)
    scores+=keyword_scores * 0.3
    
    length_scores=length_scoring(sentences)
    scores+=length_scores * 0.1
    
    return scores

def tfidf_scoring(sentences, tfidf_matrix):
    scores = np.zeros(len(sentences))
    
    for i in range(len(sentences)):
        if tfidf_matrix.shape[0] > i:
            sentence_tfidf = tfidf_matrix[i].toarray()[0]
            scores[i] = np.mean(sentence_tfidf)
    
    if np.max(scores) > 0:
        scores = scores / np.max(scores)
    
    return scores

def position_scoring(sentences):
    scores = np.zeros(len(sentences))
    for i in range(len(sentences)):
        if i == 0:
            scores[i] = 1.0
        elif i == len(sentences) - 1:
            scores[i] = 0.8
        elif i < len(sentences) * 0.3:
            scores[i] = 0.6
        elif i < len(sentences) * 0.7:
            scores[i] = 0.3
        else:
            scores[i] = 0.2
    
    return scores

def keyword_scoring(sentences):
    scores = np.zeros(len(sentences))
    for i, sentence in enumerate(sentences):
        sentence_lower = sentence.lower()
        keyword_count = 0
        for keyword in MEDICAL_KEYWORDS:
            if keyword.lower() in sentence_lower:
                keyword_count += 1
        scores[i] = keyword_count * 2.0
    
    if np.max(scores) > 0:
        scores=scores/np.max(scores)
    
    return scores

def length_scoring(sentences):
    scores = np.zeros(len(sentences))
    for i, sentence in enumerate(sentences):
        words = sentence.split()
        if 10 <= len(words) <= 25:
            scores[i] = 1.0
        elif 5 <= len(words) < 10:
            scores[i] = 0.7
        elif 25 < len(words) <= 35:
            scores[i] = 0.5
        else:
            scores[i] = 0.2
    
    return scores

def extract_key_points(text):
    sentences = split_into_sentences(text)
    key_points = []
    
    tfidf_vectorizer = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    
    importance_scores = calculate_importance_scores(sentences, tfidf_matrix)
    
    top_indices = np.argsort(importance_scores)[-5:]
    key_points = []
    for i in sorted(top_indices):
        key_points.append(sentences[i])
    
    return key_points 