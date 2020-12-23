import streamlit as st
from transformers import pipeline, set_seed, TextGenerationPipeline


class TextGenerator:
    """
    TextGenerator encapsulates a pre-trained language model used for generating text.
    """

    def __init__(self,
                 model: str,
                 seed: int = 42):

        # set the seed for generation
        self._seed = seed
        set_seed(seed)

        # load the language model
        self.model = model
        self.generator = self.load_generator(model=model)

    def load_generator(self, model: str) -> TextGenerationPipeline:
        """
        Load the language model.
        """

        if self.model == 'gpt2':
            # model: gpt2, gpt2-large
            return pipeline('text-generation', model='gpt2')
        else:
            raise NotImplementedError(f"{self.model} is not implemented.")

    def generate_text(
            self,
            starting_text: str,
            max_length: int = 100,
            num_return_sequences: int = 1,
            temperature: float = 1.0,
            top_p: float = 1.0
    ) -> str:
        """
        Generate text using the language model.
        """
        # TODO(karan): check that this all works with Tempest
        return self.generator(
            starting_text,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            top_p=top_p
        )[0]['generated_text']


@st.cache(allow_output_mutation=True)
def instantiate_generator(model: str):
    """
    Create a generator.
    """
    return TextGenerator(model=model)
