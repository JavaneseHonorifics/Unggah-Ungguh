LABEL_NAMES = {
    0: "Ngoko-Indonesia",
    1: "Ngoko Alus-Indonesia",
    2: "Krama-Indonesia",
    3: "Krama Alus-Indonesia"
}


JAVANESE_CORPUS = [
    'JVWIKI',
    'The identifikasi-bahasa',
    'Javanese dialect identification',
    'Korpus-Nusantara (Jawa)',
    'Korpus-Nusantara (Jawa Ngoko)',
    'JVID-ASR',
    'JVID TTS (female)',
    'JVID TTS (male)',
    'OSCAR-2301 Javanese',
    'Unggah-Ungguh',
]

def map_models(model):
    model_mapping = {
        'model1': 'SahabatAI v1 Instruct (Gemma2 9B)',
        'model2': 'SahabatAI v1 Instruct (Llama3 8B)',
        'model3': 'Llama3.1 8B Instruct',
        'model4': 'Sailor2 8B',
        'model6': 'Gemma2 9B Instruct',
        'modelOpenAI': 'GPT4o',
        'modelGemini': 'Gemini 1.5 Pro'
    }
    return model_mapping.get(model, model)

def map_labels(label):
    label_mapping = {
        0: 'Ngoko',
        1: 'Ngoko Alus',
        2: 'Krama',
        3: 'Krama Alus'
    }
    return label_mapping.get(label, f'Class {label}')