from types import SimpleNamespace

import streamlit as st

from app.SessionState import get as get_app_state
from app.components import Janus
from app.login import login_widget
from app.model import instantiate_generator
from app.utils import Session, get_user_history

if __name__ == '__main__':
    # Create a session state
    application_state = get_app_state(
        login_state=SimpleNamespace(username=None, successful=False, registering=False),
        current_session=Session(mode="Exploratory",
                                id=1,
                                name="",
                                description="",
                                generations=[],
                                favorites=set()),
        model_settings={},
        session_history=[],
        app_state={},
    )

    # The main title
    st.title(":rocket: Janus: Write with Mercury")

    # Login the user
    login_state = login_widget(login_state=application_state.login_state)

    # Load back the user's information
    user_history = get_user_history(login_state.username)
    application_state.current_session.id = user_history.session_id
    application_state.session_history = user_history.session_history

    # Create a generator
    generator = instantiate_generator(model='gpt2')

    # Create the application
    janus = Janus(
        generator=generator,
        current_session=application_state.current_session,
        model_settings=application_state.model_settings,
        session_history=application_state.session_history,
        username=login_state.username,
    )
    janus.start()
