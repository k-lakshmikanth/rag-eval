import litellm

SYSTEM_PROMPT = (
    "Answer the question using only the provided context. "
    "If the context is insufficient, say so."
)


def generate_answer(llm, question, context, temperature=0.0):
    joined = "\n\n".join(c["text"] for c in context)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{joined}\n\nQuestion: {question}"},
    ]
    resp = litellm.completion(model=llm, messages=messages, temperature=temperature)
    answer = resp.choices[0].message.content
    usage = resp.usage.model_dump() if resp.usage else {}
    return answer, usage
