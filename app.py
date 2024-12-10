from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import wikipedia
import google.generativeai as genai

# Configure the Gemini API key
genai.configure(api_key="AIzaSyAiSeon2RzZZrJPy2EM2yNGALy61siYBO4")

# Create FastAPI app instance
app = FastAPI()

# Pydantic model for input data validation
class MitigationRequest(BaseModel):
    topics: dict  # A dictionary of topic:value pairs

# Function to fetch information from Wikipedia
def fetch_info_from_sources(topic: str):
    try:
        summary = wikipedia.summary(topic, sentences=3)
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Disambiguation Error: {str(e)}"
    except wikipedia.exceptions.PageError:
        return f"Page for '{topic}' not found."
    return summary

# Function to query LLM for mitigation strategies
def get_mitigation_solution(info: str, input_value: str):
    prompt = f"Given the following information from Wikipedia: {info}\n" \
             f"and the provided input: {input_value}, provide mitigation strategies."
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.candidates[0].content.parts[0].text

# FastAPI route for the mitigation endpoint
@app.post("/api/mitigation")
async def mitigation(request: MitigationRequest):
    results = {}
    try:
        for topic, input_value in request.topics.items():
            # Fetch Wikipedia information based on topic
            info = fetch_info_from_sources(topic)
            # Get mitigation strategy from LLM based on both topic info and input value
            solution = get_mitigation_solution(info, input_value)
            results[topic] = {
                "topic_info": info,
                "mitigation_solution": solution
            }
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))