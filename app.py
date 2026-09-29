import streamlit as st

from rag import generate_answer


# ==========================================
# Page configuration
# ==========================================

st.set_page_config(
    page_title="NLPBot",
    page_icon="🤖",
    layout="centered"
)


# ==========================================
# Custom styling
# ==========================================

st.markdown(
    """
    <style>

    .main {
        max-width: 900px;
        margin: auto;
    }

    .title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        color: #777;
        font-size: 16px;
        margin-top: 0;
        margin-bottom: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================
# Header
# ==========================================

st.markdown(
    '<div class="title">🤖 NLPBot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Your intelligent NLP Course Assistant</div>',
    unsafe_allow_html=True
)


# ==========================================
# Sidebar
# ==========================================

with st.sidebar:

    st.header("NLPBot")

    st.write(
        "Ask questions about your NLP lecture material "
        "and get answers using Retrieval-Augmented Generation."
    )

    st.divider()

    st.subheader("📚 Knowledge Base")

    st.write("11 NLP lecture PDFs")
    st.write("335 indexed chunks")

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.rerun()


# ==========================================
# Initialize chat history
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==========================================
# Welcome message
# ==========================================

if not st.session_state.messages:

    with st.chat_message("assistant"):

        st.markdown(
            """
            👋 **Hi! I'm NLPBot.**

            I'm your NLP course assistant. Ask me anything
            about your lecture material.

            **Try asking:**
            - What is Natural Language Processing?
            - What is stemming?
            - Explain TF-IDF.
            - What is the Vector Space Model?
            - What is LSTM?
            """
        )


# ==========================================
# Display previous messages
# ==========================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander("📚 Lecture Sources"):

                for source in message["sources"]:
                    st.write(f"• {source}")


# ==========================================
# Chat input
# ==========================================

question = st.chat_input(
    "Ask anything about your NLP lectures..."
)


# ==========================================
# Process question
# ==========================================

if question:

    # Display user message
    with st.chat_message("user"):
        st.markdown(question)

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner(
            "Searching your NLP lectures..."
        ):

            try:

                answer, retrieved_documents = generate_answer(
                    question,
                    chat_history=st.session_state.messages[:-1]
                )

                st.markdown(answer)

                # Get unique lecture sources
                unique_sources = list(
                    dict.fromkeys(
                        document["source"]
                        for document in retrieved_documents
                    )
                )

                # Display sources
                if unique_sources:

                    with st.expander(
                        "📚 Lecture Sources"
                    ):

                        for source in unique_sources:
                            st.write(
                                f"• {source}"
                            )

                # Save assistant response
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": unique_sources
                    }
                )

            except Exception as e:

                st.error(
                    f"Something went wrong: {str(e)}"
                )