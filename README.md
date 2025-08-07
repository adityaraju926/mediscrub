# MediScrub

## Problem Statement
In the healthcare industry today, there are thousands that visit hospitals where technology is dated and patient files are being viewed via physical files. This project addresses the critical challenge of processing medical documents while ensuring patient privacy and generating meaningful summaries. This is done by doing a role-based PHI removal process and document summarization to ensure that sensitive data is not leaked while being efficient.

## Data Source
This project does not use a formal dataset to help with training. Given that the documents are provided by the user, this project uses the inputted data for summary generation.

## Review of Relevant Previous Efforts and Literature
- [**Exploring Medical Text Summarization with John Snow Labs**](https://github.com/databricks-industry-solutions/jsl-medical-text-summarization): This project is a solution that provides summarization models for healthcare applications. This includes pre-trained models such as clinical summarizers, biomedical research summarizers, etc.
- [**Towards Improved Recall in Medical Document Summarization**](https://github.com/Neilus03/recsum): This project focuses on improving recall in medical document summarization, ensuring that medical information is not lost during the summarization process. 
- [**Medical Text Summarization using LLMs**](https://github.com/tiru-patel/Medical_Text_Summarization_using_LLMs): This project focuses on using LLMs for medical text summarization tasks, comapring different architectures and their effectiveness.

## How is MediScrub different?
This project focuses on privacy by implementing role-based access and PHI detection that the other projects do not address. Furthermore, the other applications are not in a deployable state where as this project contains the UI to allow users to get their information in a straightforward manner. Finally, MediScrub uses various models while two out of the three other projects only use one model.

## Model Evaluation Process & Metric Selection
The evaluation metrics were calculated by calculating the accuracy, precision, recall and f-1 score based on the raw text extracted from the pdf and the summary files for each of the models. These metrics were chosen as they ensure that there is no information loss and false information while still being balanced.

## Modeling Approach

### 1. Naive Approach
- **Method**: Simple sentence extraction based on keyword frequency and position
- **Features**: Medical keyword scoring, sentence position weighting, length-based filtering
- **Use Case**: Baseline comparison and lightweight processing

### 2. Classical Machine Learning Approach
- **Method**: TF-IDF vectorization with multi-criteria scoring
- **Features**: TF-IDF importance, position scoring, keyword density, length optimization
- **Use Case**: Balanced performance with interpretable results

### 3. Deep Learning Approach
- **Method**: Google T5 transformer model for abstractive summarization
- **Features**: Neural text generation, medical domain adaptation, beam search
- **Use Case**: High-quality, contextually aware summaries

## Data Preprocessing Pipeline
The phi.py contains functions for detecting sensitive entities like names, dates, phone numbers, and medical record numbers, then replaces them with placeholders to ensure patient privacy. The file also includes a comprehensive function that processes local PDFs through the entire pipeline of extracting text, detecting PHI entities, scrubbing sensitive information, and generating summaries using all three model approaches.

## Models Evaluated and Model Selected

### 1. Naive Model
- **Type**: Rule-based extractive summarization
- **Advantages**: Fast, interpretable, no dependencies
- **Disadvantages**: Limited context understanding, basic scoring

### 2. Classical ML Model
- **Type**: TF-IDF + multi-criteria scoring
- **Advantages**: Good performance, interpretable, domain-aware
- **Disadvantages**: Limited to extractive summarization

### 3. Deep Learning Model (Selected for UI)
- **Type**: T5 transformer for abstractive summarization
- **Advantages**: High-quality output, context understanding, flexible length
- **Disadvantages**: Higher computational cost, requires GPU for optimal performance

## Comparison to Naive Approach

### Classical ML vs. Naive:
- **Speed**: Classical ML is slower but still fast
- **Quality**: TF-IDF provides more sophisticated scoring than simple keyword matching
- **Reliability**: Classical ML offers more consistent results
- **Resource Usage**: Moderate computational requirements with scikit-learn

### Deep Learning vs. Naive:
- **Quality**: T5 produces more coherent, contextually aware summaries
- **Length**: T5 generates longer, more detailed summaries
- **Flexibility**: T5 can handle complex medical terminology better
- **Cost**: T5 requires more computational resources

## How to Run Project
```bash
pip install -r requirements.txt
streamlit run ui.py
```

1. **Login**: Use "doctor" (password: doc123) or "frontdesk" (password: front123)
2. **Upload**: Select one or more PDF medical documents
3. **Process**: Click "Deep Learning Summaries" button
4. **View Results**: See generated summaries in the UI

## Results and Conclusions
![MediScrub Results](images\performance_metrics.png)

The Classical ML approach performs the best overall by having the most comprehensive summaries while maintaing the precision, while the Deep Learning model provides the most concise outputs but misses more content from the original text.

## Ethics Statement
This project prioritizes patient privacy and data protection by implementing role-based access control and using the GLiNER pre-trained model, which is designed for privacy compliant PHI detection and removal in healthcare use cases. The data extracted from the documents are scrubbed for sensitive information while still maintaing the core meaning for the summarization process. The summaries are all done in real-time and not stored anywhere ensuring privacy as well.

## Project Structure
```
├── README.md
├── requirements.txt
├── ui.py
├── preprocessing/
│   └── phi.py
├── models/
│   ├── naive_summarizer.py
│   ├── classic_summarizer.py
│   └── deep_learning_summarizer.py
├── data/
│   ├── full_text.txt
│   └── output/
│       ├── doctor/
│       └── front_desk/
└── images/
    └── performance_metrics.png
```
