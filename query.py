import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import wikipedia
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
import os

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY environment variable is not set")

# Configure the Gemini API key
genai.configure(api_key=API_KEY)

# Create FastAPI app instance
app = FastAPI()

# Pydantic model for input data validation
class QueryRequest(BaseModel):
    context: str  # The financial context or topic
    question: str  # The user's financial query

# Function to retrieve financial information (replace Wikipedia with a financial data source)
def retrieve_relevant_data(context: str) -> str:
    try:
        # Replace this with a financial API or dataset for better results
        summary = wikipedia.summary(context, sentences=3)
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Disambiguation Error: {str(e)}"
    except wikipedia.exceptions.PageError:
        return f"Page for '{context}' not found."
    return summary

# Function to query the LLM with a financial advisor role
def query_with_context(retrieved_data: str, question: str) -> str:
    # Set the model prompt for a financial advisor role
    prompt = f"You are a financial advisor. Use the following context to answer the query:\n\n" \
             f"Context: {retrieved_data}\n\n" \
             f"Question: {question}\n\n" \
             f"Provide detailed, actionable advice as a professional financial advisor."
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.candidates[0].content.parts[0].text

# Evaluation metrics: ROUGE and BLEU
def evaluate_generation(generated_response: str, reference_answer: str):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference_answer, generated_response)
    bleu_score = sentence_bleu([reference_answer.split()], generated_response.split())
    return rouge_scores, bleu_score

# FastAPI route for the query endpoint
@app.post("/api/query")
async def query(request: QueryRequest):
    try:
        # Retrieve financial data
        retrieved_data = retrieve_relevant_data(request.context)
        
        # Generate a response from the model
        generated_answer = query_with_context(retrieved_data, request.question)

        # Example of a realistic reference answer (this could come from financial experts or reliable datasets)
        reference_answer = "Diversifying your portfolio and using hedging strategies can help mitigate market risk effectively."

        # Evaluate the generated response using ROUGE and BLEU
        rouge_scores, bleu_score = evaluate_generation(generated_answer, reference_answer)

        # Return the evaluation results
        return {
            "context": retrieved_data,
            "question": request.question,
            "answer": generated_answer,
            "rouge_score": rouge_scores,
            "bleu_score": bleu_score
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
