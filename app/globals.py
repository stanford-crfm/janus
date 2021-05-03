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

MODEL_SOURCES = ['Mercury', 'Huggingface', 'Platelet']

MERCURY_MODELS = {
    'GPT2-Small: Aurora': 'aurora-gpt2-small-x21',
    'GPT2-Small: Blizzard': 'blizzard-gpt2-small-x49',
}

MERCURY_PATHS = {
    'aurora-gpt2-small-x21':
        '/u/scr/nlp/mercury/community/gpt2-small/aurora-gpt2-small-x21/',
    'blizzard-gpt2-small-x49':
        '/u/scr/nlp/mercury/community/gpt2-small/blizzard-gpt2-small-x49/',
}

HUGGINGFACE_MODELS = {
    'GPT2-Small': 'gpt2',
    'GPT2-Medium': 'gpt2-medium',
    'GPT2-Large': 'gpt2-large',
    'GPT2-XL': 'gpt2-xl',
    'EleutherAI/gpt-neo-125M': 'EleutherAI/gpt-neo-125M',
    'EleutherAI/gpt-neo-350M': 'EleutherAI/gpt-neo-350M',
    'EleutherAI/gpt-neo-1.3B': 'EleutherAI/gpt-neo-1.3B',
    'EleutherAI/gpt-neo-2.7B': 'EleutherAI/gpt-neo-2.7B',
}

PLATELET_MODELS = {
    "GPT2-Small-Ents": "ents_eli5",
    "GPT2-Small-EntsNulled": "entsnulled_eli5",
    "GPT2-Small-NoEnts": "noents_eli5",
}

PLATELET_PATHS = {
    'ents_eli5': '/dfs/scratch0/lorr1/projects/platelet/logs/ents_eli5/eli5_04-08-2021-16-27-24/outputs/last_model',
    'entsnulled_eli5': '/dfs/scratch0/lorr1/projects/platelet/logs/ents_eli5/eli5_04-08-2021-16-27-24/outputs/last_model',
    'noents_eli5': '/dfs/scratch0/lorr1/projects/platelet/logs/no_ents_eli5_2/eli5_04-29-2021-18-59-52/outputs/last_model',
}