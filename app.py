# -------------------------------------------------------------------------
# 참고: 이 코드의 일부는 다음 GitHub 리포지토리에서 참고하였습니다:
# https://github.com/lim-hyo-jeong/Wanted-Pre-Onboarding-AI-2407
# 해당 리포지토리의 라이센스에 따라 사용되었습니다.
# -------------------------------------------------------------------------

import streamlit as st
import time
from langchain_core.messages import HumanMessage, AIMessage
from utils import load_model, load_prompt, set_memory, initialize_chain, generate_message

st.title("페르소나 챗봇")

# load_model(model_name='gpt_4o')

character_name = st.selectbox(
    "**캐릭터 골라줘!**", ("trump", "biden"),
    index=0, key="character_name_select"
)

st.session_state.character_name = character_name

model_name = st.selectbox(
    "**모델을 골라줘!**",
    ("gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo", "gpt-4o-mini"),
    index=0,
    key="model_name_select",
)

st.session_state.model_name = model_name

if "chat_started" not in st.session_state:
    st.session_state.chat_started = False
    st.session_state.memory = None
    st.session_state.chain = None

def start_chat() -> None:
    """선택된 모델을 기반으로 채팅을 시작하는 함수입니다.
    """
    llm = load_model(st.session_state.model_name)
    st.session_state.chat_started = True
    st.session_state.memory = set_memory()
    st.session_state.chain = initialize_chain(
        llm = llm,
        character_name=st.session_state.character_name,
        memory = st.session_state.memory
    )

if st.button("Start Chat"):
    start_chat()

if st.session_state.chat_started:
    if st.session_state.memory is None or st.session_state.chain is None:
        start_chat()
    for message in st.session_state.memory.chat_memory.messages:
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            continue
        with st.chat_message(role):
            st.markdown(message.content)

if prompt := st.chat_input():
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        response = generate_message(
            st.session_state.chain, prompt
        )

        for chunk in response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(full_response+"▌")
        message_placeholder.markdown(full_response.strip())