import streamlit as st
from streamlit_chat import message

# Assume qa_chain is already defined and initialized somewhere in your application.
# It should be an instance of a LangChain Question-Answering Chain or similar.

# Initialize Streamlit app
st.set_page_config(page_title="Chatbot: The Brothers Karamazov", layout="centered")
st.title("Chatbot: The Brothers Karamazov")
st.write("Ask any question about the book 'The Brothers Karamazov'. Type 'exit' to end the chat.")

# Session state to maintain conversation history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for i, chat in enumerate(st.session_state.messages):
    if chat['role'] == 'user':
        message(chat['content'], is_user=True, key=f"user_{i}")
    else:
        message(chat['content'], key=f"bot_{i}")

# Input box for user question
user_input = st.text_input("Your question:", placeholder="Type your question here and press Enter...")

if user_input:
    if user_input.lower() in ['exit', 'выход', 'quit']:
        st.write("Chat ended. Refresh the page to start a new conversation.")
    else:
        # Add user message to the conversation
        st.session_state.messages.append({"role": "user", "content": user_input})

        try:
            # Generate response using the QA chain
            # TODO: replace with the code that gets the real model answer
            # response = qa_chain.run(user_input)
            response = "Answer Stub: Have a nice day!"
        except Exception as e:
            response = f"Sorry, an error occurred: {e}"

        # Add bot response to the conversation
        st.session_state.messages.append({"role": "bot", "content": response})

        # Display new messages
        message(user_input, is_user=True, key=f"user_{len(st.session_state.messages) - 2}")
        message(response, key=f"bot_{len(st.session_state.messages) - 1}")
