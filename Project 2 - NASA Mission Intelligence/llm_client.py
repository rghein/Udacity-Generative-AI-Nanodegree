from typing import Dict, List
from openai import OpenAI

def generate_response(openai_key: str, user_message: str, context: str, 
                      conversation_history: List[Dict], 
                      model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI with context"""

    system_prompt = """
    You are an AI expert on NASA missions.

    You MUST follow these rules:
    - Use ONLY the provided context.
    - Do NOT use outside knowledge.
    - If the answer is not in the context, say:
    "I do not have enough information in the provided documents to answer that."
    - Your answer MUST be directly supported by the context.
    - Include at least one exact quote from the context that supports your answer.
    - Do NOT paraphrase key facts.

    Answer format:
    - First, give a clear answer.
    - Then provide supporting evidence from the context.
    - Cite sources when possible.
    """

    # add system prompt to messages
    messages = [{"role": "system", "content": system_prompt}]

    # TODO: Add chat history
    for message in conversation_history:
        if "role" in message and "content" in message:
            messages.append(message)

    # TODO: Set context in messages
    messages.append({"role": "system", "content": f"Context:\n{context}"})

    # add the current user message
    messages.append({"role": "user", "content": f"Question:\n{user_message}"})

    # TODO: Create OpenAI Client
    client = OpenAI(api_key=openai_key)

    # TODO: Send request to OpenAI
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
    except Exception as e:
        return f"API Error: {str(e)}"

    # TODO: Return response
    if response.choices:
        return response.choices[0].message.content
    return "No response generated."