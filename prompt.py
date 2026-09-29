SYSTEM_PROMPT = """
You are NLPBot, an intelligent assistant for an NLP university course.

Your job is to answer questions using the provided lecture context.

Rules:
1. Answer using the provided context whenever possible.
2. Do not make up information that is not supported by the context.
3. Explain concepts clearly and simply.
4. If the answer is not available in the provided context, say:
   "I couldn't find this information in your NLP lecture material."
5. When useful, mention the lecture source.
6. For formulas, definitions, algorithms, or examples, preserve the terminology
   used in the lecture material.
"""


def build_prompt(question, retrieved_documents, chat_history=None):

    context = "\n\n".join(
        [
            f"Source: {doc['source']}\n{doc['text']}"
            for doc in retrieved_documents
        ]
    )

    history_text = ""

    if chat_history:
        history_text = "\n".join(
            [
                f"{message['role'].capitalize()}: {message['content']}"
                for message in chat_history
            ]
        )

    prompt = f"""
{SYSTEM_PROMPT}

Previous conversation:
{history_text}

Relevant lecture context:
{context}

Student question:
{question}

Answer the student's question clearly and directly.
"""

    return prompt