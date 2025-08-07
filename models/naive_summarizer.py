import re
from typing import List, Dict
from collections import Counter
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

MEDICAL_KEYWORDS = [
    'diagnosis', 'treatment', 'medication', 'symptoms', 'patient',
    'doctor', 'hospital', 'medical', 'condition', 'disease',
    'prescription', 'dosage', 'blood pressure', 'heart rate',
    'temperature', 'lab results', 'test results', 'procedure',
    'surgery', 'therapy', 'recovery', 'follow-up', 'appointment'
]

# Using NLTK library stopwords
STOP_WORDS = set(stopwords.words('english'))

def summarize_naive(text, max_sentences=5):
    sentences = split_into_sentences(text)
    if len(sentences) <= max_sentences:
        return text  
    
    scored_sentences = []
    for i, sentence in enumerate(sentences):
        score = score_sentence(sentence, i, len(sentences))
        scored_sentences.append((sentence, score))
    
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    top_sentences = []
    for sentence, _ in scored_sentences[:max_sentences]:
        top_sentences.append(sentence)
    
    top_sentences.sort(key=lambda x: sentences.index(x))
    summary = ' '.join(top_sentences)

    return summary

# Using regex to split into sentences
def split_into_sentences(text):
    sentences = re.split(r'[.!?]+', text)
    cleaned_sentences = []
    for s in sentences:
        if s.strip():
            cleaned_sentences.append(s.strip())
    return cleaned_sentences

def score_sentence(sentence, position, total_sentences):
    score = 0.0
    sentence_lower = sentence.lower()
    
    keyword_count = 0
    for keyword in MEDICAL_KEYWORDS:
        if keyword.lower() in sentence_lower:
            keyword_count += 1
    score += keyword_count * 2.0
    
    # Score based on position
    if position == 0:  
        score += 1.5
    elif position == total_sentences - 1:  
        score += 1.0
    elif position < total_sentences * 0.3:  
        score += 0.5
    
    # Score based on sentence length
    words = sentence.split()
    if 10 <= len(words) <= 25:
        score += 0.5
    elif len(words) > 25:
        score-=0.5
    
    return score

def extract_key_points(text):
    sentences = split_into_sentences(text)
    key_points = []
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        for keyword in MEDICAL_KEYWORDS:
            if keyword.lower() in sentence_lower:
                key_points.append(sentence)
                break
    
    top_key_points=key_points[:5]

    return top_key_points 