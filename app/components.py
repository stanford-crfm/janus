import copy
import os
from types import SimpleNamespace
from typing import List
import pandas as pd

import streamlit as st

from app.model import TextGenerator, checkpoint_widget, instantiate_generator
from app.utils import Generation, Session
from app.globals import TEXT_GENERATION_ATTRIBUTES


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

        self.modes = {
            'Exploratory': 0,
            'Annotation': 1,
            'Review': 2,
            'Compare': 3,

            'QA: ELI5': 5,
        }

        self.username = username
        self.device = device

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
            elif self.current_session.mode == 'Compare':
                self.layout_compare()
            elif self.modes[self.current_session.mode] == 5:
                self.layout_qa_eli5()
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
        if not os.path.exists(os.path.join('data', self.username)):
            os.makedirs(os.path.join('data', self.username))
        self.current_session.to_pickle(
            os.path.join('data', self.username,
                         f'session_{self.current_session.id}.pkl')
        )

    def save_all_sessions(self):
        """
        Save the user's session history.
        """
        self.reset_current_session()
        if not os.path.exists(os.path.join('data', self.username)):
            os.makedirs(os.path.join('data', self.username))
        for sess in self.session_history:
            sess.to_pickle(
                os.path.join('data', self.username, f'session_{sess.id}.pkl'))

    def layout_sidebar(self):
        """
        Layout the options in the sidebar.
        """
        # Button to create a new session
        new_session = st.sidebar.button("New Session")

        # Mode selection
        mode = st.sidebar.radio("Mode", list(self.modes.keys()))

        # If mode changed or new session started, and the old session was in progress
        if (self.current_session.mode != mode or new_session) \
                and self.current_session.generations:
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
                "What temperature to use for generation "
                "(high = more stochastic, low = determinstic)?",
                min_value=1e-10, max_value=1., value=1.)
            top_p = st.slider(
                "Top_p: the most probable tokens with probabilities that "
                "add up to top_p or higher are kept for generation.",
                min_value=0., max_value=1., value=1.)
            top_k = int(st.slider(
                "Top_k: the number of highest probability vocabulary "
                "tokens to keep for top-k-filtering.",
                min_value=1, max_value=1000, value=50))

            self.model_settings = {
                'num_tokens': num_tokens,
                'temperature': temperature,
                'top_p': top_p,
                'top_k': top_k
            }

        if st.sidebar.button("Save All Sessions"):
            self.save_all_sessions()

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
            st.subheader('**Current variation**')
            if self.current_session.generations:
                st.write('**Input:** ', self.current_session.generations[-1].input)
                st.write('**Output:** ', self.current_session.generations[-1].output)
            else:
                st.write("Nothing yet!")

        with attribute:
            with st.beta_expander('Add New Attribute'):
                new_attribute = st.text_input(label="New Attribute", max_chars=40,
                                              key=len(self.current_session.attributes))
                if st.button("Add Attribute"):
                    self.current_session.attributes.append(new_attribute)

            if self.current_session.mode == 'Exploratory':
                st.subheader('**Attributes**')
                att = st.multiselect(
                    'Select descriptive attributes',
                    self.current_session.attributes,
                    key=len(self.current_session.generations) + len(
                        self.current_session.attributes)
                )
            elif self.current_session.mode == 'Annotation':
                st.subheader('**Label**')
                att = st.radio(
                    'Select descriptive attributes',
                    ('Success', 'Failure'),
                    key=len(self.current_session.generations)
                )

        # Set the attributes
        if self.current_session.generations:
            self.current_session.generations[-1].labels = set(att)

        save = st.button("Save Variation")
        if save and self.current_session.generations:
            # Store as a favorite
            self.current_session.favorites.add(
                len(self.current_session.generations) - 1)

        # Display saved variations
        with st.beta_expander("Show Saved Variations"):
            for i, idx in enumerate(self.current_session.favorites):
                generation = self.current_session.generations[idx]
                st.write('**Variation:** ', i)
                st.write('**Input:** ', generation.input)
                st.write('**Output:** ', generation.output)
                st.write('**Attributes:** ', ", ".join(generation.labels))

    def layout_qa_eli5(self):
        """
        Create an area for answering ELI5 questions.
        """
        st.write("## ELI5: Answer Questions!")
        st.write("##### Subreddit: explainlikeimfive")

        st.write("### Option 1: Choose a pre-written question from the ELI5 test set")
        with st.beta_expander("Choose a pre-written question"):
            # Dropdown where user can just run the model on ELI5 examples from the test set
            eli5_test_qs = load_eli5_test_qs()
            eli5_dropdown = st.selectbox(
                "Select a question",
                options=range(len(eli5_test_qs['whole_question'])), # using workaround so I can get the index
                format_func=lambda idx: eli5_test_qs['whole_question'][idx]
            )
            st.write("You selected:")
            st.write(eli5_test_qs['whole_question'][eli5_dropdown])
            generate_text_button_1 = st.button('Answer', key='answer1')


        st.write("### Option 2: Write your own question!")
        with st.beta_expander("Write your own question"):
            # Get user input for the question and context
            subreddit = "explainlikeimfive"
            question = st.text_area("Question", height=40)
            context = st.text_area(
                "Context [Optional Text to Expand on Your Question]",
                height=200
            )
            generate_text_button_2 = st.button('Answer', key='answer2')

        out_qids = '' # only need this to silence an unnecessary error message that was occurring
        if generate_text_button_1 or generate_text_button_2:
            if generate_text_button_1:
                # Combine the subreddit, question, context
                text = self._create_eli5_context(subreddit, 
                                                 eli5_test_qs['title'][eli5_dropdown],
                                                 eli5_test_qs['selftext'][eli5_dropdown])
            elif generate_text_button_2:
                # Combine the subreddit, question, context
                text = self._create_eli5_context(subreddit, question, context)

            out_text, out_qids = self.generator.generate_text(
                starting_text=text,
                max_length=self.model_settings['num_tokens'],
                temperature=self.model_settings['temperature'],
            )

            # out_qids should be either (A) in a format like [['Q123', 'Q456']] or [[]], if the model uses
            # entities, or (B) should be a string, if the model does NOT use entities
            assert isinstance(out_qids, str) or (len(out_qids) == 1 and isinstance(out_qids[0], list))
            if not isinstance(out_qids, str):
                out_qids = out_qids[0]

            # Add a generation
            self.current_session.generations.append(
                Generation(
                    model=self.generator.model_name,
                    config=self.model_settings,
                    checkpoint=self.checkpoint_info,
                    input=text,
                    output=out_text,
                    ##### TODO: add out_qids to Generation as well???
                    labels=set(),
                    annotations=[],
                )
            )

        variation, raw_variation = st.beta_columns([3, 2])

        with variation:
            st.subheader('**Current variation**')
            if self.current_session.generations:
                unwrapped_output = self._unwrap_eli5_text(
                    self.current_session.generations[-1].output
                )
                st.write('**Subreddit:** ', unwrapped_output.subreddit)
                st.write('**Question:** ', unwrapped_output.question)
                st.write('**Context:** ', unwrapped_output.context)
                
                # Print entities identified by Bootleg (with links to Wikipedia pages)
                if isinstance(out_qids, str):
                    entity_text = out_qids # if model doesn't use entities, just print the N/A message
                else:
                    wiki_base_url = "https://en.wikipedia.org/wiki/"
                    wiki_titles = [self.generator.qid2title[qid] for qid in out_qids]
                    entity_text = ', '.join([f"[{title}]({wiki_base_url + '_'.join(title.split())})" for title in wiki_titles])
                st.write('**Entities labeled by Bootleg:**', entity_text)
                
                st.write('**Answer:** ', unwrapped_output.answer)

            else:
                st.write("Nothing yet!")

        with raw_variation:
            with st.beta_expander("View Raw"):
                st.write('**Raw Input:** ',
                         self.current_session.generations[-1].input if len(self.current_session.generations) > 0 else "")
                st.write('**Raw Output:** ',
                         self.current_session.generations[-1].output if len(self.current_session.generations) > 0 else "")

    def layout_review(self):
        """
        Create an area for reviewing old or saved sessions.
        """
        if self.current_session.id == 2:
            st.write("Add at least one more session to start reviewing.")
            st.stop()

        session_id = st.select_slider(
            'Select Session',
            options=range(1, self.current_session.id)
        )
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
                'Select Session',
                options=range(1, self.current_session.id),
                key=key,
            )
            session = self.session_history[session_id - 1]

            # Select what generations to display
            if len(session.favorites) > 0:
                only_saved = st.radio(
                    'Generations to Display', options=['All', 'Saved'], key=key)
            else:
                only_saved = st.radio(
                    'Generations to Display', options=['All'], key=key)

            # Select the generation index
            if only_saved == 'Saved':
                generation_index = st.select_slider(
                    'Select Saved Generation',
                    options=list(session.favorites),
                    key=key
                ) if len(session.favorites) > 1 else list(session.favorites)[0]
            else:
                generation_index = st.select_slider(
                    'Select Generation',
                    options=range(len(session.generations)),
                    key=key
                ) if len(session.generations) > 1 else 0

            # Display the generation
            generation = session.generations[generation_index]
            st.write('**Variation:** ', generation_index)
            st.write('**Model Source**', generation.checkpoint['model_source'])
            st.write('**Model Name**', generation.model)
            st.write('**Checkpoint**', generation.checkpoint['checkpoint'])
            st.write('**Input:** ', generation.input)
            st.write('**Output:** ', generation.output)
            st.write('**Attributes:** ', ", ".join(generation.labels))

        # Create a split view
        col1, _, col2 = st.beta_columns([0.4, 0.2, 0.4])

        # Populate both columns
        with col1:
            _layout_column('1')

        with col2:
            _layout_column('2')


@st.cache
def load_eli5_test_qs():
    eli5_test_set_path = "/dfs/scratch0/lorr1/projects/platelet-data/data/eli5/eli5-test_eli5_ent.json"
    data = pd.read_json(eli5_test_set_path, lines=True)
    data = data[['title', 'selftext']]
    data['whole_question'] = data['title'] + " " + data['selftext']
    # Just return 15000 random questions, not all of them (feel free to choose a different number)
    data = data.sample(n=15000, random_state=123)
    data = data.reset_index(drop=True)
    return data