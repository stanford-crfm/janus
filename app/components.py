import copy
import os
from typing import List

import streamlit as st

from app.model import TextGenerator
from app.utils import Generation, Session


class Janus:

    def __init__(self,
                 generator: TextGenerator,
                 current_session: Session,
                 model_settings: dict,
                 session_history: List[Session],
                 username: str):

        # Text generator
        self.generator = generator

        self.current_session = current_session
        self.model_settings = model_settings
        self.session_history = session_history

        # List of attributes used for labeling text generation
        self.attributes = [
            'Common Sense',
            'Storytelling',
            'Informative',
            'Logical Reasoning',
            'Question Answering',
            'Factual',
            'Toxic',
            'Biased',
        ]

        self.modes = {
            'Exploratory': 0,
            'Annotation': 1,
            'Review': 2,
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
            if self.current_session.mode == 'Review':
                self.layout_review()
            else:
                self.layout_body()
        else:
            st.stop()

    def reset_current_session(self, mode="Exploratory"):
        # Append the current session to the session history
        self.session_history.append(copy.deepcopy(self.current_session))

        # Create a new empty session
        self.current_session.mode = mode
        self.current_session.id += 1
        self.current_session.name = None
        self.current_session.description = None
        self.current_session.generations = []
        self.current_session.favorites = set()

    def save_current_session(self):
        """
        Save the current session object.
        """
        self.current_session.to_pickle(os.path.join('data', self.username, f'session_{self.current_session.id}.pkl'))

    def save_all_sessions(self):
        """
        Save the user's session history.
        """
        self.reset_current_session()
        for sess in self.session_history:
            sess.to_pickle(os.path.join('data', self.username, f'session_{sess.id}.pkl'))

    def layout_sidebar(self):
        """
        Layout the options in the sidebar.
        """
        # Button to create a new session
        new_session = st.sidebar.button("New Session")

        # Mode selection
        mode = st.sidebar.radio("Mode", list(self.modes.keys()))

        # If mode changed or new session started, and the old session was in progress
        if (self.current_session.mode != mode or new_session) and self.current_session.generations:
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
            num_tokens = st.slider("How many tokens to generate?",
                                   min_value=0, max_value=500, value=30)
            temperature = st.slider(
                "What temperature to use for generation (high = more stochastic, low = determinstic)?",
                min_value=0., max_value=1., value=1.)
            top_p = st.slider(
                "Top_p: the most probable tokens with probabilities that "
                "add up to top_p or higher are kept for generation.",
                min_value=0., max_value=1., value=1.)
            top_k = int(st.slider(
                "Top_k: the number of highest probability vocabulary tokens to keep for top-k-filtering.",
                min_value=1, max_value=1000, value=50))

            self.model_settings = {
                'num_tokens': num_tokens,
                'temperature': temperature,
                'top_p': top_p,
                'top_k': top_k
            }

        with st.sidebar.beta_expander("Attribute Settings"):
            new_attribute = st.text_input(label="New Attribute", max_chars=40)
            if st.button("Add Attribute"):
                self.attributes.append(new_attribute)

        if st.sidebar.button("Save All Sessions"):
            self.save_all_sessions()

    def layout_body(self):
        """
        Layout the main body of the application.
        """
        # Heading for main body
        text = st.text_area("Prime GPT-X: Generate conditional outputs", height=100)

        generate_text_button = st.button('Generate')
        if generate_text_button:
            out = self.generator.generate_text(
                starting_text=text,
                max_length=self.model_settings['num_tokens'],
                temperature=self.model_settings['temperature'],
            )

            # Add a generation
            self.current_session.generations.append(
                Generation(
                    model=self.generator.model,
                    config=self.model_settings,
                    input=text,
                    output=out,
                    labels=set(),
                    annotations=[],
                )
            )

        variation, attribute = st.beta_columns(2)

        with variation:
            st.subheader('**Current variation**')
            if self.current_session.generations:
                st.write('**Input:** ', self.current_session.generations[-1].input)
                st.write('**Output:** ', self.current_session.generations[-1].output)
            else:
                st.write("Nothing yet!")

        with attribute:
            if self.current_session.mode == 'Exploratory':
                st.subheader('**Attributes**')
                att = st.multiselect(
                    'Select descriptive attributes',
                    self.attributes,
                )
            elif self.current_session.mode == 'Annotation':
                st.subheader('**Label**')
                att = st.radio(
                    'Select descriptive attributes',
                    ('Success', 'Failure'),
                )

        # Set the attributes
        if self.current_session.generations:
            self.current_session.generations[-1].labels = set(att)

        save = st.button("Save Variation")
        if save and self.current_session.generations:
            # Store as a favorite
            self.current_session.favorites.add(len(self.current_session.generations) - 1)

        # Display saved variations
        with st.beta_expander("Show Saved Variations"):
            for i, idx in enumerate(self.current_session.favorites):
                generation = self.current_session.generations[idx]
                st.write('**Variation:** ', i)
                st.write('**Input:** ', generation.input)
                st.write('**Output:** ', generation.output)
                st.write('**Attributes:** ', ", ".join(generation.labels))

    def layout_review(self):
        """
        Create an area for reviewing old or saved sessions.
        """
        if self.current_session.id == 2:
            st.write("No sessions to review.")
            st.stop()

        session_id = st.select_slider('Select Session', options=range(1, self.current_session.id))
        session = self.session_history[session_id - 1]

        only_saved = st.radio('Generations to Display', options=['Saved', 'All'])

        if only_saved == 'Saved':
            for i, idx in enumerate(session.favorites):
                generation = session.generations[idx]
                st.write('**Variation:** ', i)
                st.write('**Input:** ', generation.input)
                st.write('**Output:** ', generation.output)
                st.write('**Attributes:** ', ", ".join(generation.labels))
        elif only_saved == 'All':
            for i, generation in enumerate(session.generations):
                st.write('**Generation:** ', i)
                st.write('**Input:** ', generation.input)
                st.write('**Output:** ', generation.output)
                st.write('**Attributes:** ', ", ".join(generation.labels))