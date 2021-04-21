import copy
import os
import pandas as pd
from pathlib import Path

from typing import List
from types import SimpleNamespace


import streamlit as st

from app.model import TextGenerator, instantiate_generator, get_checkpoints
from app.utils import Generation, Session
from app.globals import (
    TEXT_GENERATION_ATTRIBUTES,
    MODEL_SOURCES,
    MERCURY_MODELS,
    MERCURY_PATHS,
    HUGGINGFACE_MODELS,
)


class Janus:
    def __init__(
        self,
        generator: TextGenerator,
        current_session: Session,
        model_settings: dict,
        session_history: List[Session],
        username: str,
        device: str,
    ):

        # Text generator
        self.generator = generator

        self.current_session = current_session
        self.model_settings = model_settings
        self.checkpoint_info = generator.get_checkpoint_info()
        self.session_history = session_history
        self.device = device

        self.modes = {
            "Exploratory": 0,
            "Annotation": 1,
            "Review": 2,
            "Compare": 3,
            "Compare Across Time": 4,
        }

        self.username = username

    def start(self):
        """
        Start the main application.
        """
        # Layout the sidebar
        self.layout_sidebar()

        # Start the application
        if self.current_session.mode in self.modes:
            if self.current_session.mode == "Review":
                self.layout_review()
            elif self.current_session.mode == "Compare":
                self.layout_compare()
            elif self.current_session.mode == "Compare Across Time":
                self.layout_compare_across_time()
            else:
                self.layout_body()
        else:
            st.stop()

    def reset_current_session(self, mode="Exploratory"):
        """
        Reset the current session.
        Adds current session to session history and starts a new session.
        """
        # Append the current session to the session history
        self.session_history.append(copy.deepcopy(self.current_session))

        # Create a new empty session
        self.current_session.mode = mode
        self.current_session.id += 1
        self.current_session.name = None
        self.current_session.description = None
        self.current_session.generations = []
        self.current_session.favorites = set()
        self.current_session.attributes = TEXT_GENERATION_ATTRIBUTES

    def save_current_session(self):
        """
        Save the current session object.
        """
        if not os.path.exists(os.path.join("data", self.username)):
            os.makedirs(os.path.join("data", self.username))
        self.current_session.to_pickle(
            os.path.join(
                "data", self.username, f"session_{self.current_session.id}.pkl"
            )
        )

    def save_all_sessions(self):
        """
        Save the user's session history.
        """
        self.reset_current_session()
        if not os.path.exists(os.path.join("data", self.username)):
            os.makedirs(os.path.join("data", self.username))
        for sess in self.session_history:
            sess.to_pickle(
                os.path.join("data", self.username, f"session_{sess.id}.pkl")
            )

    def layout_sidebar(self):
        """
        Layout the options in the sidebar.
        """
        # Button to create a new session
        new_session = st.sidebar.button("New Session")

        # Mode selection
        mode = st.sidebar.radio("Mode", list(self.modes.keys()))

        # If mode changed or new session started, and the old session was in progress
        if (
            self.current_session.mode != mode or new_session
        ) and self.current_session.generations:
            # Save the current session and reset it
            self.save_current_session()
            self.reset_current_session(mode)
        elif self.current_session.mode != mode:
            self.current_session.mode = mode

        # Set the session name
        st.sidebar.write("**Session ID:**", self.current_session.id)
        if not self.current_session.name:
            session_name = st.sidebar.text_area("Session Name", height=20)
            self.current_session.name = session_name
        else:
            st.sidebar.write("**Session Name:**", self.current_session.name)

        with st.sidebar.beta_expander("Model Settings"):
            num_tokens = st.slider(
                "How many tokens to generate?",
                min_value=0,
                max_value=500,
                value=30,
            )
            temperature = st.slider(
                "What temperature to use for generation "
                "(high = more stochastic, low = determinstic)?",
                min_value=1e-10,
                max_value=1.0,
                value=1.0,
            )
            top_p = st.slider(
                "Top_p: the most probable tokens with probabilities that "
                "add up to top_p or higher are kept for generation.",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
            )
            top_k = int(
                st.slider(
                    "Top_k: the number of highest probability vocabulary "
                    "tokens to keep for top-k-filtering.",
                    min_value=1,
                    max_value=1000,
                    value=50,
                )
            )

            self.model_settings = {
                "num_tokens": num_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            }

        if st.sidebar.button("Save All Sessions"):
            self.save_all_sessions()

    def layout_compare_across_time(self):
        # Add drop down for models + checkpoints
        def get_available_models():
            for model, path in MERCURY_PATHS.items():
                if not os.path.exists(path):
                    if "Mercury" in MODEL_SOURCES:
                        MODEL_SOURCES.remove("Mercury")

        # create mapping between models and available checkpoints
        model_to_ckpts = {"Mercury": {}, "Huggingface": {}}
        for model_source in MODEL_SOURCES:
            if model_source == "Mercury":
                for model_name in list(MERCURY_MODELS.keys()):
                    # Select from the list of available model checkpoints
                    checkpoints_available = get_checkpoints(
                        path=Path(MERCURY_PATHS[MERCURY_MODELS[model_name]]),
                    )
                    model_to_ckpts["Mercury"][
                        model_name
                    ] = checkpoints_available
            elif model_source == "Huggingface":
                for model_name in list(HUGGINGFACE_MODELS.keys()):
                    model_to_ckpts["Huggingface"][model_name] = None

        hf_models, mercury_models, checkpoints = st.beta_columns(3)

        with hf_models:
            selected_hf_models = st.multiselect(
                "Available HuggingFace Models:",
                list(model_to_ckpts["Huggingface"].keys()),
            )
        with mercury_models:
            selected_mercury_models = st.multiselect(
                "Available Mercury Models:",
                list(model_to_ckpts["Mercury"].keys()),
            )
        with checkpoints:
            available_ckpts = sum(
                [
                    list(model_to_ckpts["Mercury"][model].keys())
                    for model in model_to_ckpts["Mercury"].keys()
                ],
                [],
            )
            if len(available_ckpts) == 0:
                available_ckpts = ["best_ckpt"]
            selected_ckpts = st.multiselect(
                "Available Checkpoints:", available_ckpts
            )

        # map models to
        mercury_models = {}
        for ckpt in selected_ckpts:
            mercury_models[ckpt] = {}
            for mercury_model in selected_mercury_models:
                if ckpt in model_to_ckpts["Mercury"][mercury_model].keys():
                    mercury_models[ckpt][mercury_model] = {}
                    mercury_models[ckpt][mercury_model][
                        "model_info"
                    ] = SimpleNamespace(
                        model_source="Mercury",
                        model_name=mercury_model,
                        checkpoint=ckpt,
                        checkpoint_path=model_to_ckpts["Mercury"][mercury_model][ckpt],
                    )

        hf_models = {}
        for hf_model in selected_hf_models:
            hf_models[hf_model] = {}
            hf_models[hf_model]["model_info"] = SimpleNamespace(
                model_source="Huggingface",
                model_name=hf_model,
                checkpoint=None,
                checkpoint_path=None,
            )

        # Add generators
        for ckpt, models_dict in mercury_models.items():
            for model, model_attr in models_dict.items():
                mercury_models[ckpt][model][
                    "generator"
                ] = instantiate_generator(
                    **model_attr["model_info"].__dict__, device=self.device
                )

        for hf_model, model_attr in hf_models.items():
            generator = instantiate_generator(
                **model_attr["model_info"].__dict__, device=self.device
            )
            hf_models[hf_model]["generator"] = generator

        # Heading for main body
        text = st.text_area(
            "Prime GPT-X: Generate conditional outputs", height=100
        )

        generate_text_button = st.button("Generate")
        if generate_text_button:
            # add generations to table
            generation_df = pd.DataFrame(
                index=selected_ckpts,
                columns=selected_mercury_models + selected_hf_models,
            )

            for ckpt, models_dict in mercury_models.items():
                for model, model_attr in models_dict.items():
                    generator = model_attr["generator"]
                    output = generator.generate_text(
                        starting_text=text,
                        max_length=self.model_settings["num_tokens"],
                        temperature=self.model_settings["temperature"],
                    )
                    generation_df.loc[ckpt, model] = str(output)

            for model, model_attr in hf_models.items():
                generator = model_attr["generator"]
                output = generator.generate_text(
                    starting_text=text,
                    max_length=self.model_settings["num_tokens"],
                    temperature=self.model_settings["temperature"],
                )
                generation_df.loc[:, model] = str(output)

            st.table(generation_df)

    def layout_body(self):
        """
        Layout the main body of the application.
        """
        # Heading for main body
        text = st.text_area(
            "Prime GPT-X: Generate conditional outputs", height=100
        )

        generate_text_button = st.button("Generate")
        if generate_text_button:
            out = self.generator.generate_text(
                starting_text=text,
                max_length=self.model_settings["num_tokens"],
                temperature=self.model_settings["temperature"],
            )

            # Add a generation
            self.current_session.generations.append(
                Generation(
                    model=self.generator.model_name,
                    config=self.model_settings,
                    checkpoint=self.checkpoint_info,
                    input=text,
                    output=out,
                    labels=set(),
                    annotations=[],
                )
            )

        variation, attribute = st.beta_columns([3, 2])

        with variation:
            st.subheader("**Current variation**")
            if self.current_session.generations:
                st.write(
                    "**Input:** ", self.current_session.generations[-1].input
                )
                st.write(
                    "**Output:** ", self.current_session.generations[-1].output
                )
            else:
                st.write("Nothing yet!")

        with attribute:
            with st.beta_expander("Add New Attribute"):
                new_attribute = st.text_input(
                    label="New Attribute",
                    max_chars=40,
                    key=len(self.current_session.attributes),
                )
                if st.button("Add Attribute"):
                    self.current_session.attributes.append(new_attribute)

            if self.current_session.mode == "Exploratory":
                st.subheader("**Attributes**")
                att = st.multiselect(
                    "Select descriptive attributes",
                    self.current_session.attributes,
                    key=len(self.current_session.generations)
                    + len(self.current_session.attributes),
                )
            elif self.current_session.mode == "Annotation":
                st.subheader("**Label**")
                att = st.radio(
                    "Select descriptive attributes",
                    ("Success", "Failure"),
                    key=len(self.current_session.generations),
                )

        # Set the attributes
        if self.current_session.generations:
            self.current_session.generations[-1].labels = set(att)

        save = st.button("Save Variation")
        if save and self.current_session.generations:
            # Store as a favorite
            self.current_session.favorites.add(
                len(self.current_session.generations) - 1
            )

        # Display saved variations
        with st.beta_expander("Show Saved Variations"):
            for i, idx in enumerate(self.current_session.favorites):
                generation = self.current_session.generations[idx]
                st.write("**Variation:** ", i)
                st.write("**Input:** ", generation.input)
                st.write("**Output:** ", generation.output)
                st.write("**Attributes:** ", ", ".join(generation.labels))

    def layout_review(self):
        """
        Create an area for reviewing old or saved sessions.
        """
        if self.current_session.id == 2:
            st.write("Add at least one more session to start reviewing.")
            st.stop()

        session_id = st.select_slider(
            "Select Session", options=range(1, self.current_session.id)
        )
        session = self.session_history[session_id - 1]

        only_saved = st.radio(
            "Generations to Display", options=["Saved", "All"]
        )

        if only_saved == "Saved":
            for i, idx in enumerate(session.favorites):
                generation = session.generations[idx]
                st.write("**Variation:** ", i)
                st.write("**Input:** ", generation.input)
                st.write("**Output:** ", generation.output)
                st.write("**Attributes:** ", ", ".join(generation.labels))
        elif only_saved == "All":
            for i, generation in enumerate(session.generations):
                st.write("**Generation:** ", i)
                st.write("**Input:** ", generation.input)
                st.write("**Output:** ", generation.output)
                st.write("**Attributes:** ", ", ".join(generation.labels))

    def layout_compare(self):
        """
        Create an area for comparing generations.
        """

        if self.current_session.id == 2:
            st.write("Add at least one more session to start reviewing.")
            st.stop()

        def _layout_column(key: str):
            # Select a session
            session_id = st.select_slider(
                "Select Session",
                options=range(1, self.current_session.id),
                key=key,
            )
            session = self.session_history[session_id - 1]

            # Select what generations to display
            if len(session.favorites) > 0:
                only_saved = st.radio(
                    "Generations to Display", options=["All", "Saved"], key=key
                )
            else:
                only_saved = st.radio(
                    "Generations to Display", options=["All"], key=key
                )

            # Select the generation index
            if only_saved == "Saved":
                generation_index = (
                    st.select_slider(
                        "Select Saved Generation",
                        options=list(session.favorites),
                        key=key,
                    )
                    if len(session.favorites) > 1
                    else list(session.favorites)[0]
                )
            else:
                generation_index = (
                    st.select_slider(
                        "Select Generation",
                        options=range(len(session.generations)),
                        key=key,
                    )
                    if len(session.generations) > 1
                    else 0
                )

            # Display the generation
            generation = session.generations[generation_index]
            st.write("**Variation:** ", generation_index)
            st.write("**Model Source**", generation.checkpoint["model_source"])
            st.write("**Model Name**", generation.model)
            st.write("**Checkpoint**", generation.checkpoint["checkpoint"])
            st.write("**Input:** ", generation.input)
            st.write("**Output:** ", generation.output)
            st.write("**Attributes:** ", ", ".join(generation.labels))

        # Create a split view
        col1, _, col2 = st.beta_columns([0.4, 0.2, 0.4])

        # Populate both columns
        with col1:
            _layout_column("1")

        with col2:
            _layout_column("2")
