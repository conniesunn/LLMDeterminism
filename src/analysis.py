import openai
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score

# Function to get LLM responses
def get_llm_response(prompt, openai_api_key):
    openai.api_key = openai_api_key
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  
            messages=[{"role": "user", "content": prompt}]
        )
        return response['choices'][0]['message']['content']
    except openai.error.OpenAIError as e:
        print(f"An error occurred: {e}")

# Function to calculate Jaccard Similarity
def calculate_jaccard_similarity(response1, response2):
    vectorizer = CountVectorizer(binary=True)  # Convert counts to binary
    vectors = vectorizer.fit_transform([response1, response2]).toarray()
    jaccard = jaccard_score(vectors[0], vectors[1], average=None)
    return jaccard.mean()  # If you still want a single score, take the mean of the scores for all labels

# Sample test prompts
prompts = [
    "What is the capital city of France?",
    # Add other base prompts and variations here
]

# API key (you must use your actual API key here)
openai_api_key = "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"

# Collect responses from LLM
responses = {}
for prompt in prompts:
    responses[prompt] = get_llm_response(prompt, openai_api_key)
    print(responses[prompt])

# Calculate Jaccard Similarity between base prompt response and variations
jaccard_similarities = {}
for base_prompt in prompts:
    variations = [
        # Add the variations for each base prompt here
        "Can you tell me what the capital of France is?",
        "France's capital is...?",
        "What is considered the capital city of the French Republic?",
        "I'm wondering, what city is the capital of France?"
    ]
    base_response = responses[base_prompt]
    jaccard_similarities[base_prompt] = []
    for var in variations:
        var_response = get_llm_response(var, openai_api_key)
        print(var_response)
        similarity = calculate_jaccard_similarity(base_response, var_response)
        jaccard_similarities[base_prompt].append(similarity)

# Print Jaccard Similarities
for prompt, similarities in jaccard_similarities.items():
    print(f"Base Prompt: {prompt}")
    print(f"Jaccard Similarities: {similarities}")
    print(f"Mean Similarity: {np.mean(similarities)}\n")

# Thematic consistency analysis would be more qualitative and could involve human judgment or further NLP techniques
