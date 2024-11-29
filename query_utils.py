import numpy as np


# Function to retrieve context
def retrieve_context(query, embedding_model, index, documents, top_k=5):
    query_embedding = embedding_model.encode(query, convert_to_numpy=True)
    distances, indices = index.search(np.array([query_embedding]), top_k)
    retrieved_docs = [documents[i] for i in indices.flatten()]
    return " ".join(retrieved_docs)


def generate_response(context, query, tokenizer, model, device, max_position_embeddings=2048):
    full_prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    tokenized_input = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=max_position_embeddings)["input_ids"].to(device)
    output = model.generate(
        tokenized_input,
        max_new_tokens=500,
        repetition_penalty=1.2,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        do_sample=True,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)
