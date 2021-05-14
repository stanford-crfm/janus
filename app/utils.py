import glob
import os
import re
import pickle
from types import SimpleNamespace
from typing import List, Set

import streamlit as st
from dataclasses import dataclass
from natsort import natsorted
from bootleg.utils.utils import get_lnrm


def clean_blocks(blocks):
    """
    Take streamlit placeholders and empty them.
    """
    for block in blocks:
        block.empty()


def setup_blocks(num_blocks):
    """
    Return num_blocks in the sidebar.
    """
    return [st.sidebar.empty() for _ in range(num_blocks)]


def get_user_history(username: str):
    """
    Load a user's history of interaction from session pickle files.
    """
    last_session_id = 0
    session_history = []
    if os.path.exists(f'data/{username}'):
        session_files = natsorted(list(glob.glob(f'data/{username}/session_*.pkl')))
        if session_files:
            last_session_id = int(session_files[-1].split("_")[1].split(".")[0])
            session_history = [pickle.load(open(file, 'rb')) for file in session_files]

    return SimpleNamespace(
        session_id=last_session_id + 1,
        session_history=session_history
    )


@dataclass
class Generation:
    # Model name
    model: str
    # Model checkpoint info
    checkpoint: dict
    # Model config info
    config: dict

    # Input text
    input: str
    # Generated output text
    output: str

    # Optional labels and other annotations
    labels: Set[str] = None
    annotations: List[str] = None


@dataclass
class Session:
    # Application mode
    mode: str
    # Session name and ID
    name: str
    id: int
    # Description of the session
    description: str

    # List of generations
    generations: List[Generation]

    # Favorited indices
    favorites: Set[int]

    # Attributes to label generated text
    attributes: list
    
    def to_pickle(self, path: str):
        with open(path, 'wb') as outfile:
            pickle.dump(self, outfile)


def custom_entity_extractor(text, qid2title, annotator):
    assert text.count("{") == text.count("}"), "Curly brackets are unbalanced"
    assert text.count("[") == text.count("]"), "Brackets are unbalanced"

    # Same as input text, but remove the user's entity annotations
    clean_text = text.replace("{", "").replace("}", "")
    clean_text = re.sub("\[Q\d+\]", "", clean_text)

    assert clean_text.count("[") == 0 and clean_text.count("]") == 0, "Please ensure that your input follows the instructions exactly"

    text = text.split()

    spans = []
    qids = []
    pointer = 0
    while pointer < len(text):
        if text[pointer].startswith("{"):
            if "}" in text[pointer]: # then the mention is only 1 word long
                spans.append((pointer, pointer+1))
            else: # the mention is >1 word long. move ahead to the end of the mention
                start_pointer = pointer
                while "}" not in text[pointer]:
                    pointer += 1
                spans.append((start_pointer, pointer+1))
            bracket_start_idx = text[pointer].index("[")
            bracket_end_idx = text[pointer].index("]")
            qid = text[pointer][bracket_start_idx+1 : bracket_end_idx]
            assert re.search("^Q\d+$", qid) is not None
            qids.append(qid)
        pointer += 1

    def get_mention(qid2title, qid):
        """Returns the lower cased title with punctuation stripped"""
        return get_lnrm(qid2title[qid], strip=True, lower=True)

    # Create input for "contextualized static embeddings" (like the VMware example created by Laurel).
    # This list will later be passed into BootlegAnnotator.label_mentions in the extracted_examples field to
    # create the "contextualized static embeddings"
    extracted_exs = [
        {
            "sentence": qid2title[qid],
            "aliases": [get_mention(qid2title, qid)],
            "spans": [[0, len(qid2title[qid].split())]],
            "cands": [[qid]],
        }
        for qid in qids
    ]
    # Run the above entities through Bootleg to get the "contextualized static embeddings"
    out_dict = annotator.label_mentions(extracted_examples=extracted_exs)

    # Format things so that we can return only the necessary outputs
    return_dict = {}
    assert qids == [x[0] for x in out_dict['qids']]
    return_dict['qids'] = [qids]
    return_dict['probs'] = [[x[0] for x in out_dict['probs']]]
    return_dict['spans'] = [[list(x) for x in spans]]
    return_dict['embs'] = [[x[0] for x in out_dict['embs']]]

    return return_dict, clean_text