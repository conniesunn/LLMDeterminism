import openai
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score
import random
import nltk
from nltk.corpus import wordnet
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

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
openai_api_key = "key"

# Collect responses from LLM
responses = {}
for prompt in prompts:
    responses[prompt] = get_llm_response(prompt, openai_api_key)
    print(responses[prompt])


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

def replace_synonyms(sentence):
    words = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(words)

    # Randomly choose a noun or adjective to replace with a synonym
    candidates = [word for word, pos in pos_tags if pos in ["NN", "JJ"]]
    if candidates:
        to_replace = random.choice(candidates)
        synonyms = get_synonyms(to_replace)
        if synonyms:
            synonym = random.choice(synonyms)
            return sentence.replace(to_replace, synonym, 1)
    return sentence

def generate_variations(base_prompt):
    variation1 = replace_synonyms(base_prompt)
    
    # Simple sentence restructuring
    if base_prompt.startswith("What is the"):
        variation2 = base_prompt.replace("What is the", "Can you tell me what the", 1)
    else:
        variation2 = "Can you tell me: " + base_prompt
    
    # Asking the question in a different way
    variation3 = base_prompt.replace("What is the", "I'm curious about the", 1)
    
    return [variation1, variation2, variation3]

# Calculate Jaccard Similarity between base prompt response and variations
jaccard_similarities = {}
for base_prompt in prompts:
    variations = generate_variations(base_prompt)
    base_response = responses[base_prompt]
    jaccard_similarities[base_prompt] = []
    for var in variations:
        print(var)
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
