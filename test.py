import streamlit as st
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer, util

# Groq API Function
def get_explanation(problem, solution):
    GROQ_API_KEY = "GROQ API KEY"
    url = "https://api.groq.com/openai/v1/chat/completions"  # Groq API endpoint

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "gemma2-9b-it",
        "messages": [
            {"role": "system", "content": "You are a professional and an expert vehicle mechanic.Just explain the solution, Be precise and provide short solution to the customers, Just give the Soltution of the problem in 50-80 words. You dont need to have to introduce yourself or the solution again in the token. just straight forward to the soltuion of the problem. If you have been asked anything irrelevent, just appologize and move on."},
            {"role": "user", "content": f"Problem: {problem}\nSolution: {solution}\nExplain the solution like a professional mechanic."}
        ],
        "max_tokens": 80,
        "temperature": 1
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response_data = response.json()

        if "error" in response_data:
            return "Error retrieving explanation. Please try again."

        return response_data["choices"][0]["message"]["content"]

    except Exception as e:
        return "Error retrieving explanation. Please try again."

# Load CSV file
csv_path = "./s.csv"  # Update with correct path
df = pd.read_csv(csv_path)

# Ensure column names match your CSV
problem_column = "PROBLEM"
solution_column = "ACTION"

# Load Sentence-BERT for retrieval
retrieval_model = SentenceTransformer("all-MiniLM-L6-v2")

# Streamlit UI
st.title("Vehicle Mechanics QA System")
st.write("Ask a question about vehicle issues, and the system will find the best solution.")

question = st.text_input("Describe the vehicle issue:", "Type your question here...")

if st.button("Get Solution"):
    if question.strip():
        # Convert problems into embeddings
        problem_texts = df[problem_column].astype(str).tolist()
        problem_embeddings = retrieval_model.encode(problem_texts, convert_to_tensor=True)
        question_embedding = retrieval_model.encode(question, convert_to_tensor=True)

        # Find the most relevant problem
        similarities = util.pytorch_cos_sim(question_embedding, problem_embeddings)
        best_idx = similarities.argmax().item()
        best_problem = problem_texts[best_idx]
        best_solution = df.iloc[best_idx][solution_column]

        # Get explanation from Groq API
        explanation = get_explanation(best_problem, best_solution)

        # Display results
        st.subheader("Most Relevant Problem Found:")
        st.write(best_problem)

        st.subheader("Recommended Solution:")
        st.write(f"**{best_solution}**")

        st.subheader("Expert Mechanic Explanation:")
        st.write(f"💡 {explanation}")

    else:
        st.warning("Please enter a vehicle problem description.")
