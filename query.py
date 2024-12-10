from ragas import Evaluator

# Initialize the evaluator
evaluator = Evaluator()

# Define query, ground truth, and generated answer
query = "What is the price of the S&P 500 index in 2015?"
ground_truth = "The price of the S&P 500 index in 2015 varies by date, but historical data can be found on financial platforms like Yahoo Finance or Bloomberg."
generated_answer = """
The provided text about quests is irrelevant to the question "what is the price of the S&P 500 index in 2015?". 
There's no connection between the definition of a quest and the price of a financial index.

The mitigation strategy here is simply to ignore the irrelevant information about quests and focus on finding the answer to the question about the S&P 500 index. 
This involves using a reliable financial data source like:
* Google Finance
* Yahoo Finance
* Financial News Websites
"""

# Evaluate the response
scores = evaluator.evaluate(query, generated_answer, ground_truth)

# Print the evaluation scores
print("Evaluation Scores:")
for metric, score in scores.items():
    print(f"{metric}: {score:.2f}")
