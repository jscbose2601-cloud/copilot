import os
import streamlit as st
from openai import OpenAI

# Retrieve the tunnel URL from Streamlit secrets
ollama_url = st.secrets.get("OLLAMA_BASE_URL", "http://localhost:11434")

# If using the official OpenAI-compatible client configuration
client = OpenAI(
    base_url=f"{ollama_url}/v1",
    api_key="ollama",  # Ollama doesn't require a real key, but the client expects one
)

st.title("💬 Remote Ollama + Streamlit App")

prompt = st.text_input("Ask your local model:")
if prompt:
  with st.spinner("Thinking..."):
    response = client.chat.completions.create(
        model="llama3.2",  # Replace with your local model name
        messages=[{"role": "user", "content": prompt}],
    )
    st.write(response.choices[0].message.content)
