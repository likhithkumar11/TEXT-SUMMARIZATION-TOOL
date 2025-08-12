# text_summarizer.py

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

def summarize_text(text):
    # Load pre-trained T5 model and tokenizer
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Preprocess input text
    input_text = "summarize: " + text.strip().replace("\n", " ")
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Generate summary (adjust max_length for summary size)
    summary_ids = model.generate(input_ids, max_length=150, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

# Example use
if __name__ == "__main__":
    long_text = """
    Artificial Intelligence (AI) is rapidly transforming the world we live in. 
    From healthcare to transportation, AI systems are becoming an integral part of every industry.
    AI enables machines to mimic human intelligence by learning from data and improving performance over time.
    Its applications range from virtual assistants and language translators to medical diagnosis and self-driving cars.
    As AI continues to evolve, it holds the potential to solve some of humanity’s most pressing challenges.
    """

    print("Original Text:\n", long_text)
    print("\n---\n")
    summary = summarize_text(long_text)
    print("Summary:\n", summary)
