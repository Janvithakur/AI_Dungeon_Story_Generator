import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import io

# Load pretrained GPT-2 model and tokenizer from Hugging Face
model_name = "gpt2"  # You can change to 'EleutherAI/gpt-neo-125M' if you prefer GPT-Neo
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)

# Streamlit app title and description
st.title("📝 AI Dungeon Story Generator")
st.markdown("""
    Enter your prompt and select a genre to create your interactive fantasy story!
    The AI will generate multiple story continuations for you to explore.
""")

# Prompt input field
prompt = st.text_area("Enter your story prompt:")

# Genre selection dropdown
genre = st.selectbox("Choose a Genre:", ["Fantasy", "Mystery", "Sci-Fi", "Adventure"])

# Full prompt with genre
full_prompt = f"{genre} Story:\n{prompt}"

# Function to generate story continuations
def generate_story(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    outputs = model.generate(
        inputs, 
        max_length=300, 
        num_return_sequences=3,  # You can increase this for more options
        do_sample=True, 
        temperature=0.7, 
        top_p=0.9
    )
    
    story_options = []
    for output in outputs:
        story = tokenizer.decode(output, skip_special_tokens=True)
        story_options.append(story)
    
    return story_options

# Button to generate the story
if st.button("Generate Story"):
    if prompt.strip() == "":
        st.error("Please enter a prompt!")
    else:
        # Generate and display story options
        story_options = generate_story(full_prompt)
        
        for i, story in enumerate(story_options):
            st.subheader(f"Option {i+1}")
            st.write(story)

        # Allow users to download the generated story
        story_text = "\n\n".join(story_options)
        story_file = io.StringIO(story_text)
        st.download_button("Download All Stories", story_file, file_name="generated_story.txt")
