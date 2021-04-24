from collections import Callable, OrderedDict
from glob import glob
from pathlib import Path
from types import SimpleNamespace

import os
import numpy as np
import streamlit as st
import torch
from bootleg.end2end.bootleg_annotator import BootlegAnnotator
from transformers import pipeline, set_seed, TextGenerationPipeline, \
    AutoModelForCausalLM, AutoTokenizer

from app.globals import MODEL_SOURCES, MERCURY_MODELS, MERCURY_PATHS, HUGGINGFACE_MODELS
from app.platelet.models.gptent_encoder import GPT2EntLMHeadModel
from app.platelet.utils.generation_logits_process import LogitsProcessorList, MinLengthLogitsProcessor
from app.platelet.utils.generation_stopping_criteria import StoppingCriteriaList, MaxLengthCriteria


class TextGenerator:
    """
    TextGenerator encapsulates a pre-trained language model used for generating text.
    """

    def __init__(
            self,
            model_source: str,
            model_name: str,
            checkpoint: str,
            checkpoint_path: Path,
            seed: int = 42,
            device: str = None,
    ):

        # store the parameters
        self.model_source = model_source
        self.model_name = model_name
        self.checkpoint = checkpoint
        self.checkpoint_path = checkpoint_path

        # set the seed for generation
        self._seed = seed
        set_seed(seed)

        # set the device
        self._device = device
        if device is None:
            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # load the language model
        self.generator = self.load_generator()

    def get_checkpoint_info(self):
        return {
            'model_source': self.model_source,
            'model_name': self.model_name,
            'checkpoint': self.checkpoint,
            'checkpoint_path': self.checkpoint_path,
        }

    def get_checkpoint_info_string(self):
        if self.model_source == "Mercury":
            return f'Source: {self.model_source}\n' \
                   f'Model: {self.model_name}\n' \
                   f'Checkpoint: {self.checkpoint}'
        elif self.model_source == 'Huggingface':
            return f'Source: {self.model_source}\n' \
                   f'Model: {self.model_name}'
        else:
            raise NotImplementedError(
                f"Model source {self.model_source} not recognized."
            )

    def _load_mercury(self, checkpoint_path: Path):
        """
        Load a language model and its tokenizer from a HF checkpoint.

        :param checkpoint_path: a single checkpoint directory (e.g.
        /path/to/checkpoint-1000)
        """
        assert (
            checkpoint_path.is_dir()
        ), f"`checkpoint_path` must be a directory but {checkpoint_path} " \
           f"is not a directory."
        assert (
                checkpoint_path / "config.json"
        ).exists(), "Checkpoint directory must contain a `config.json` file."
        assert (
                checkpoint_path / "tokenizer_config.json"
        ).exists(), "Checkpoint directory must contain a `tokenizer_config.json` file."

        # Load things
        model = AutoModelForCausalLM.from_pretrained(f"{checkpoint_path}"). \
            eval().to(self._device)
        tokenizer = AutoTokenizer.from_pretrained(f"{checkpoint_path}")
        return pipeline("text-generation", model=model, tokenizer=tokenizer,
                        device={'cpu': -1, 'cuda': 0}[self._device])

    def load_generator(self) -> TextGenerationPipeline:
        """
        Load the language model.
        """

        if self.model_source == 'Mercury':
            return self._load_mercury(self.checkpoint_path)
        elif self.model_source == 'Huggingface':
            # model: gpt2, gpt2-large
            return pipeline(
                'text-generation',
                model=HUGGINGFACE_MODELS[self.model_name],
                device={'cpu': -1, 'cuda': 0}[self._device],
            )
        elif self.model_source == 'Platelet':
            return self._load_mercury(self.checkpoint_path)
        else:
            raise NotImplementedError(
                f"Model source {self.model_source} not recognized."
            )

    def generate_text(
            self,
            starting_text: str,
            max_length: int = 100,
            num_return_sequences: int = 1,
            temperature: float = 1.0,
            top_p: float = 1.0,
            do_sample: bool = False
    ) -> str:
        """
        Generate text using the language model.
        """
        return self.generator(
            starting_text,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample
        )[0]['generated_text']


class TextWithEntityGenerator:
    """
    TextWithEntityGenerator encapsulates a pre-trained language model that incorporates entity embeddings
    used for generating text.
    """

    def __init__(
            self,
            model_source: str,
            model_name: str,
            checkpoint: str,
            checkpoint_path: Path,
            use_ents: bool = True,
            seed: int = 42,
            device: str = None,
            annotator = None,
    ):

        # store the parameters
        self.model_source = model_source
        self.model_name = model_name
        self.checkpoint = checkpoint
        self.checkpoint_path = checkpoint_path

        # set the seed for generation
        self._seed = seed
        set_seed(seed)

        # set the device
        self._device = device
        if device is None:
            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self._bootleg_cache = "/dfs/scratch0/lorr1/projects/bootleg/tutorial_data"
        self._bootleg_threshold = 0.2
        self._bootleg_dim = 512
        self._use_ents = use_ents

        # load the language model
        self.model, self.tokenizer = self.load_generator()
        if annotator is None:
            self.annotator = self.load_annotator()
        else:
            self.annotator = annotator
        
    def get_checkpoint_info(self):
        return {
            'model_source': self.model_source,
            'model_name': self.model_name,
            'checkpoint': self.checkpoint,
            'checkpoint_path': self.checkpoint_path,
        }

    def get_checkpoint_info_string(self):
        if self.model_source == "Platelet":
            return f'Source: {self.model_source}\n' \
                   f'Model: {self.model_name}\n' \
                   f'Checkpoint: {self.checkpoint}'
        else:
            raise NotImplementedError(
                f"Model source {self.model_source} not recognized."
            )

    def _load_ent_model(self, checkpoint_path: Path):
        """
        Load a language model and its tokenizer from a HF checkpoint.

        :param checkpoint_path: a single checkpoint directory (e.g.
        /path/to/checkpoint-1000)
        """
        assert (
            checkpoint_path.is_dir()
        ), f"`checkpoint_path` must be a directory but {checkpoint_path} " \
           f"is not a directory."
        assert (
                checkpoint_path / "config.json"
        ).exists(), "Checkpoint directory must contain a `config.json` file."
        assert (
                checkpoint_path / "tokenizer_config.json"
        ).exists(), "Checkpoint directory must contain a `tokenizer_config.json` file."

        # Load things
        model = GPT2EntLMHeadModel.from_pretrained(f"{checkpoint_path}"). \
            eval().to(self._device)
        tokenizer = AutoTokenizer.from_pretrained(f"{checkpoint_path}")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return model, tokenizer

    @staticmethod
    def _create_eli5_context(subreddit, title, selftext):
        """Combine the pieces of the example that constitute the context and prefix
        them with "Q: " for question sand "A: " for answer. We mask these out when
        scoring as they will be provided by the user."""
        return " ".join(
            ("Q: " + subreddit + " ; " + title + " ; " + selftext + " A:").split()
        )

    @staticmethod
    def _unwrap_eli5_text(text):
        context, answer = text.split(" A:")
        subreddit, question, context = context.split(";")
        return SimpleNamespace(
            subreddit=subreddit.split("Q: ")[1].strip(),
            question=question.strip(),
            context=context.split(" A:")[0].strip(),
            answer=answer
        )

    def _tokenize_entities(self, unwrapped_text, spans, probs, embs) -> (torch.Tensor, np.ndarray):
        """Turns the Bootleg outputs into entity ids for the forward to the model. Also outputs
        the embedding matrix."""
        assert len(embs) == len(spans)

        def _create_entity_context(subreddit_text, title_ents, selftext_ents):
            """Combine ent ids of the pieces of the example that constitute the context and prefix
            them with -1 to match the "Q: " and "A: ". We have one ent id for each word."""
            return (
                [-1]
                + [-1] * len(subreddit_text.split())
                + [-1]
                + title_ents
                + [-1]
                + selftext_ents
                + [-1]
            )
        merged_sentence = f"{unwrapped_text.question} ||| {unwrapped_text.context}"
        text_len = len(merged_sentence.split())
        # Add the PAD/UNK entity embeddings
        final_embs = [np.zeros(self._bootleg_dim), np.zeros(self._bootleg_dim)]
        entity_ids = [-1]*text_len
        ent_id = 0
        for i in range(len(spans)):
            span = spans[i]
            start, end = span[0], span[1]
            span_len = end - start

            prob = probs[i]

            if prob < self._bootleg_threshold:
                continue

            # contextual
            entity_ids[start:end] = [ent_id] * span_len
            ent_id += 1
            final_embs.append(embs[i])

        joint_sentence = merged_sentence.split()
        assert len(entity_ids) == len(joint_sentence)
        index_to_split = joint_sentence.index("|||")
        title_ents = entity_ids[:index_to_split]
        selftext_ents = entity_ids[index_to_split+1:]
        parsed_entity_ids = _create_entity_context(unwrapped_text.subreddit, title_ents, selftext_ents)
        # Add 2 for the unk/pad entity
        parsed_entity_ids = list(map(lambda x: x + 2, parsed_entity_ids))
        final_embs = np.vstack(final_embs)
        return parsed_entity_ids, final_embs

    def _tokenize_text_and_ents(
        self, starting_text, entity_ids, tokenizer, add_space_first_tok=False
    ):
        """Converts word split tokens and entity ids into subword equivalents."""
        tokenized_ents = []
        tokenized_ctxs = []
        # If we have ent ids, these are ids per word in the question.
        # When tokenizing the words, they may get split to subwords. We need
        # to ensure to repeat the entity id for each subword
        sent_tokens = starting_text.split()
        assert len(sent_tokens) == len(
            entity_ids
        ), f"{sent_tokens} {entity_ids}"
        for i, token in enumerate(sent_tokens):
            # This will allow tokenizers that use spaces to know it's a middle word (GPT changes with spaces)
            if i > 0 or add_space_first_tok:
                token = " " + token
            for j, sub_token in enumerate(tokenizer.tokenize(token)):
                tokenized_ctxs.append(tokenizer.convert_tokens_to_ids(sub_token))
                tokenized_ents.append(entity_ids[i])
        return tokenized_ents, tokenized_ctxs

    def load_generator(self) -> TextGenerationPipeline:
        """
        Load the language model.
        """

        if self.model_source == 'Platelet':
            return self._load_ent_model(self.checkpoint_path)
        else:
            raise NotImplementedError(
                f"Model source {self.model_source} not recognized."
            )

    def load_annotator(self) -> BootlegAnnotator:
        ann = BootlegAnnotator(cache_dir=self._bootleg_cache, device=self._device, return_embs=True)
        return ann

    def generate_text(
            self,
            starting_text: str,
            max_length: int = 100,
            num_return_sequences: int = 1,
            temperature: float = 1.0,
            top_p: float = 1.0,
            do_sample: bool = False
    ) -> str:
        """
        Generate text using the language model.
        """
        unwrapped_text = TextWithEntityGenerator._unwrap_eli5_text(starting_text)
        bootleg_entities = self.annotator.label_mentions(f"{unwrapped_text.question} ||| {unwrapped_text.context}")
        # Text: who is my padre loco with purple rain
        # input_ent_ids = [0, 0, 0, 1, 1, 0, 2, 2]
        # entity_matrix = [row of 0, row of 0, padre loco embeddings, purple rain embeddings]
        
        # Geting entity inputs for model
        entity_ids, entity_matrix = self._tokenize_entities(unwrapped_text,
                                                            bootleg_entities["spans"][0],
                                                            bootleg_entities["probs"][0],
                                                            bootleg_entities["embs"][0])
        # Tokenizes the ids to be at the subword level
        tokenized_ent_ids, tokenized_text = self._tokenize_text_and_ents(starting_text, entity_ids, self.tokenizer)
        # Hacky way of saving embeddings for model to use in forward pass
        self.model.entity_embeddings = torch.from_numpy(entity_matrix)
        input_ent_ids = torch.tensor(tokenized_ent_ids).to(self._device).unsqueeze(0)
        # Set to 0 embedding
        if not self._use_ents:
            input_ent_ids = torch.zeros_like(input_ent_ids)
        generated_sequence = self.model.generate(
            input_ids=torch.tensor(tokenized_text).to(self._device).unsqueeze(0),
            input_ent_ids=input_ent_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample
        )
        generated_sequence = generated_sequence.squeeze(0).cpu()
        text = self.tokenizer.decode(
            generated_sequence,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return text


@st.cache(allow_output_mutation=True)
def instantiate_generator(
        model_source: str,
        model_name: str,
        checkpoint: str,
        checkpoint_path: Path,
        seed: int = 42,
        device: str = None,
):
    """
    Create a generator.
    """
    return TextGenerator(
        model_source=model_source,
        model_name=model_name,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        seed=seed,
        device=device,
    )


def get_checkpoints(
        path: Path,
        is_checkpoint_fn: Callable = None,
        step_extraction_fn: Callable = None,
) -> OrderedDict:
    """Given the path to a directory that contains one or more checkpoints for
    a single training run, return all checkpoints in the directory.
    :param path: directory where checkpoints are stored
    :param is_checkpoint_fn: given a path, checks if it is a valid checkpoint directory
    :param step_extraction_fn: given a valid checkpoint directory, returns the step
    at which the checkpoint was stored
    """

    # Get the list of all files and folders in the directory
    print(path)
    candidate_paths = [Path(p) for p in glob(str(path / "*"))]
    print(candidate_paths)
    # Filter out only the checkpoints
    if not is_checkpoint_fn:
        is_checkpoint_fn = lambda p: p.resolve().name.startswith("checkpoint-")
    checkpoint_paths = [p for p in candidate_paths if is_checkpoint_fn(p)]

    # Function to extract the step that a checkpoint corresponds to
    if not step_extraction_fn:
        step_extraction_fn = lambda p: int(p.resolve().name.split("-")[-1])

    # Construct an ordered dictionary mapping {step: checkpoint}, sorted by step
    return OrderedDict(
        sorted(
            [(step_extraction_fn(p), p) for p in checkpoint_paths], key=lambda k: k[0]
        )
    )


def checkpoint_widget():
    """Widget for selecting a model and checkpoint."""
    # Check if mercury paths exists before offering as option
    for model, path in MERCURY_PATHS.items():
        if not os.path.exists(path):
            if 'Mercury' in MODEL_SOURCES:
                MODEL_SOURCES.remove('Mercury')

    # Select the model source
    model_source = st.sidebar.radio("Model Source", options=MODEL_SOURCES)

    if model_source == 'Mercury':
        # Hardcode the models that are supported by Mercury
        model_name = st.sidebar.selectbox(
            "Model", options=list(MERCURY_MODELS.keys())
        )

        # Select from the list of available model checkpoints
        checkpoints_available = get_checkpoints(
            path=Path(MERCURY_PATHS[MERCURY_MODELS[model_name]]),
        )
        checkpoint = st.sidebar.selectbox(
            "Checkpoint Step", options=list(checkpoints_available.keys())
        )
        checkpoint_path = checkpoints_available[checkpoint]

    elif model_source == 'Huggingface':
        # Hardcode the models that are supported by Huggingface
        model_name = st.sidebar.selectbox(
            "Model", options=list(HUGGINGFACE_MODELS.keys())
        )

        # No checkpoints available for Huggingface models
        checkpoint = None
        checkpoint_path = None
    elif model_source == 'Platelet':
        # Write out a path where the model directory is
        model_name = None
        checkpoint = None
        checkpoint_path = Path(st.sidebar.text_input("Checkpoint Path"))

    else:
        raise NotImplementedError(f"Model source {model_source} not recognized.")

    return SimpleNamespace(
        model_source=model_source,
        model_name=model_name,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
    )
