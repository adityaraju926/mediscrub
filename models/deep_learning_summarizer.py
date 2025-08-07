import re
import numpy as np
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

MEDICAL_KEYWORDS = [
    'diagnosis', 'treatment', 'medication', 'symptoms', 'patient',
    'doctor', 'hospital', 'medical', 'condition', 'disease',
    'prescription', 'dosage', 'blood pressure', 'heart rate',
    'temperature', 'lab results', 'test results', 'procedure',
    'surgery', 'therapy', 'recovery', 'follow-up', 'appointment'
]

t5_model = None
t5_tokenizer = None

# Initlaizing model and tokenizer
def load_t5_model():
    global t5_model, t5_tokenizer
    
    if t5_model is None:
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
        t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')

def summarize_deep_learning(text, max_sentences=5):
    if len(text.split()) <= 100:
        return text
    
    load_t5_model()
    
    if t5_model is not None and t5_tokenizer is not None:
        t5_summary = summarize_with_t5(text)
        return t5_summary
    else:
        return text

def summarize_with_t5(text):
    input_text = f"summarize: {text}"
    inputs = t5_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    summary_ids = t5_model.generate(inputs,max_length=300,min_length=80,length_penalty=1.5,num_beams=4,early_stopping=True,do_sample=True,temperature=0.8,top_k=50,top_p=0.9)
    summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

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

    score+=keyword_count*2.0
    
    # Score based on position
    if position == 0:
        score += 1.5
    elif position == total_sentences-1:
        score += 1.0
    elif position < total_sentences*0.3:
        score += 0.5
    
    # Score based on sentence length
    words = sentence.split()
    if 10 <= len(words) <= 25:
        score+=0.5
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
    
    top_key_points = key_points[:5] 

    return top_key_points