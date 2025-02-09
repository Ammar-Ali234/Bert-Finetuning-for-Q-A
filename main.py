import streamlit as st
import pandas as pd
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch
from sentence_transformers import SentenceTransformer, util

# Load fine-tuned BERT model and tokenizer
MODEL_PATH = "./"  # Update if needed
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_PATH, use_safetensors=True)

# Load the CSV file
csv_path = "./s.csv"  # Update with the correct path
df = pd.read_csv(csv_path)

# Ensure column names match your CSV
problem_column = "PROBLEM"  # Update if the column name is different
solution_column = "ACTION"  # Update if the column name is different

# Load Sentence-BERT for context retrieval
retrieval_model = SentenceTransformer("all-MiniLM-L6-v2")  # Efficient similarity model

st.title("Vehicle Mechanics QA System")
st.write("Ask a question about vehicle issues, and the system will find the best solution.")

# User input
question = st.text_input("Describe the vehicle issue:", "Type your question here...")

if st.button("Get Solution"):
    if question.strip():
        # Convert all problems into embeddings
        problem_texts = df[problem_column].astype(str).tolist()
        problem_embeddings = retrieval_model.encode(problem_texts, convert_to_tensor=True)
        question_embedding = retrieval_model.encode(question, convert_to_tensor=True)

        # Find the most relevant problem
        similarities = util.pytorch_cos_sim(question_embedding, problem_embeddings)
        best_idx = similarities.argmax().item()
        best_problem = problem_texts[best_idx]
        best_solution = df.iloc[best_idx][solution_column]  # Get corresponding solution

        # Display results
        st.subheader("Most Relevant Problem Found:")
        st.write(best_problem)

        st.subheader("Recommended Solution:")
        st.write(f"**{best_solution}**")
    else:
        st.warning("Please enter a vehicle problem description.")
