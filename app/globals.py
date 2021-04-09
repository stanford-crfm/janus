TEXT_GENERATION_ATTRIBUTES = [
    'Common Sense',
    'Storytelling',
    'Informative',
    'Logical Reasoning',
    'Question Answering',
    'Factual',
    'Toxic',
    'Biased',
]

MODEL_SOURCES = ['Mercury', 'Huggingface']

MERCURY_MODELS = {
    'GPT2-Small: Aurora': 'aurora-gpt2-small-x21',
    'GPT2-Small: Blizzard': 'blizzard-gpt2-small-x49',
}

MERCURY_PATHS = {
    'aurora-gpt2-small-x21':
        '/u/scr/nlp/mercury/mistral-runs/3-28/aurora-gpt2-small-x21/',
    'blizzard-gpt2-small-x49':
        '/u/scr/nlp/mercury/mistral-runs/4-5/blizzard-gpt2-small-x49/',
}

HUGGINGFACE_MODELS = {
    'GPT2-Small': 'gpt2',
}