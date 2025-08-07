from gliner import GLiNER
import pdfplumber
import re
import json
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import logging
from models.naive_summarizer import summarize_naive
from models.classic_summarizer import summarize_classical_ml
from models.deep_learning_summarizer import summarize_deep_learning

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

gliner_model = GLiNER.from_pretrained("gretelai/gretel-gliner-bi-large-v1.0")

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        all_text = []
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if text:
                all_text.append(text.strip())
            else:
                logger.warning(f"No text found on page {page_num}")
        
        combined_text = ' '.join(all_text)
        
        return combined_text

def detect_phi_entities(text):
    entities = []
    phi_labels = ["PERSON", "DATE", "PHONE", "EMAIL", "ADDRESS", "ID", "MEDICAL_RECORD_NUMBER", "SSN", "INSURANCE_ID"]
    predictions = gliner_model.predict_entities(text, labels=phi_labels)
    
    for pred in predictions:
        entity = {'text': pred['text'],'entity_type': pred['label'],'confidence': pred.get('confidence', 0.0)}
        entities.append(entity)
        
    return entities

def scrub_phi_from_text(text, phi_entities):
    scrubbed_text = text
    sorted_entities = sorted(phi_entities, key=lambda x: text.find(x['text']), reverse=True)
    
    for entity in sorted_entities:
        placeholder = get_placeholder_for_entity_type(entity['entity_type'])
        scrubbed_text = scrubbed_text.replace(entity['text'], placeholder)
        
    return scrubbed_text

def get_placeholder_for_entity_type(entity_type):
    placeholders = {'PERSON': '[PERSON]','DATE': '[DATE]','PHONE': '[PHONE]','EMAIL': '[EMAIL]','ADDRESS': '[ADDRESS]','ID': '[ID]','MEDICAL_RECORD_NUMBER': '[MRN]','SSN': '[SSN]','INSURANCE_ID': '[INSURANCE_ID]'}
    placeholder = placeholders.get(entity_type, '[REDACTED]')
    return placeholder

def process_local_pdf_and_generate_summaries(pdf_path):
    full_text = extract_text_from_pdf(pdf_path)
    
    phi_entities = detect_phi_entities(full_text)
    scrubbed_text = scrub_phi_from_text(full_text, phi_entities)
    
    summaries = {}
    
    naive_summary = summarize_naive(scrubbed_text)
    summaries['naive'] = naive_summary
    
    classical_summary = summarize_classical_ml(scrubbed_text)
    summaries['classical_ml'] = classical_summary
    
    deep_learning_summary = summarize_deep_learning(scrubbed_text)
    summaries['deep_learning'] = deep_learning_summary
    
    result = {'pdf_path': pdf_path,'timestamp': datetime.now().isoformat(),'summaries': summaries,'phi_entities_found': len(phi_entities),'original_text': full_text,'scrubbed_text': scrubbed_text}
    return result



