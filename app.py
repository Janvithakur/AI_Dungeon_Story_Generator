# Title and genre select
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import datetime
import torch

# Set seed for reproducibility
torch.manual_seed(42)

st.title("🧙‍♂️ AI Dungeon Story Generator")

# Genre selection
genres = ['Fantasy', 'Mystery', 'Sci-Fi', 'Adventure']
genre = st.selectbox("Choose a genre:", genres)

# Prompt input
prompt = st.text_area("Enter your story prompt:", height=150)

# Story length slider
max_len = st.slider("Select story length (tokens):", min_value=50, max_value=300, value=150)

# Load GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def generate_story(prompt, max_length=150):
    result = generator(prompt, max_length=max_length, num_return_sequences=3, do_sample=True)
    return result

def save_story(text):
    filename = f"story_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(filename, "w", encoding="utf-8") as file:
        file.write(text)
    return filename

# Generate Story button
if st.button("Generate Story"):
    if prompt:
        st.subheader("Story Continuations:")
        full_prompt = f"{genre} story: {prompt}"
        outputs = generate_story(full_prompt, max_length=max_len)

        # Store generated outputs
        generated_texts = []
        for i, output in enumerate(outputs):
            story = output['generated_text']
            generated_texts.append(story)
            st.text_area(f"Option {i+1}:", value=story, height=200)

        # Save generated stories in session state
        st.session_state["generated_texts"] = generated_texts
    else:
        st.warning("Please enter a prompt before generating a story.")

# Save Story button
if st.button("Save Story"):
    if "generated_texts" in st.session_state:
        all_text = "\n\n".join([f"Option {i+1}:\n{text}" for i, text in enumerate(st.session_state["generated_texts"])])
        filename = save_story(all_text)
        st.success(f"Story saved as {filename}")
    else:
        st.warning("Please generate a story before saving.")
