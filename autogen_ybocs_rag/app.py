import streamlit as st
import asyncio
import re
import os

from autogen_ybocs_rag.agents import teamConfig, orchestrate

def getFileName(msg):
    match = re.search(r'GENERATED:([^\s]+\.png)', msg)
    if match:
        return match.group(1)
    return None

def showMessage(container, msg):
    with container:
        if msg.startswith('chatbot'):
            with st.chat_message("ai"):
                st.markdown(msg[54:])
            if filename:=getFileName(msg):
                st.image(os.path.join("temp", filename), caption=filename)
        elif msg.startswith('CodeExecutor'):
            with st.chat_message("Executor", avatar="🤖"):
                st.markdown(msg)
        elif msg.startswith('Stop reason'):
            with st.chat_message("user"):
                st.markdown(msg)

st.title("Y-BOCS Chatbot assistant")

desc = st.chat_input("Hi I'm a chatbot that can refer to Y-BOCS documents. Please ask me your YBOCS questions :)")

# clicked = st.button("Send", type="primary")

# create "memory" for chatbot
if 'messages' not in st.session_state:
    st.session_state.messages = []

chat_container = st.container()

## using method from literaturereview
# if clicked:
    # async def main(desc):
    #     team = teamConfig()
    #     async for message in orchestrate(team, desc):
    #         with chat_container:
    #             if message.startswith("chatbot"):
    #                 with st.chat_message("human"):
    #                     st.markdown(message[54:])
    #             elif message.startswith("user"):
    #                 with st.chat_message("user"):
    #                     st.markdown(message)
    #             # handle input from user
    #             elif message.startswith("Enter your response:"):
    #                 with st.chat_message("user"):
    #                     # st.markdown(message)
    #                     user_response = st.text_input(label="", value=message)
    #                     if user_response != message: # if the user has entered a new response
    #                         st.session_state["messages"].append({"role": "user", "content": user_response}) # store the new response in session state
    #             else:
    #                 with st.expander("Tool Call"):
    #                     st.markdown(message)
    # with st.spinner("Finding your answer..."):
    #     asyncio.run(main(desc))
    # # st.success("Done!")
    # # st.balloons()

## using method from talkwithyourdataset
if desc:
    async def main():
        team = teamConfig()
        if "team_state" in st.session_state:
            await team.load_state(st.session_state["team_state"])
        async for message in orchestrate(team, desc):
            st.session_state["messages"].append(message) # store the new response in session state
            showMessage(chat_container, message)
        st.session_state["team_state"] = await team.save_state()
    with st.spinner("Finding your answer..."):
        asyncio.run(main())
    # st.success("Done!")
    # st.balloons()