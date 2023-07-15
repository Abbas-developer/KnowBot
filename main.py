import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import operator


# Load the knowledge base from a JSON file
def load_knowledge_base(file_path: str):
    with open(file_path, 'r') as file:
        data: dict = json.load(file)
    return data


# Save the updated knowledge base to the JSON file
def save_knowledge_base(file_path: str, data: dict):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=2)


# Train the intent recognition model
def train_intent_recognition_model(knowledge_base: dict) -> Pipeline:
    X = [q["question"] for q in knowledge_base["questions"]]
    y = [q["intent"] for q in knowledge_base["questions"]]

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LinearSVC(dual=False))  # Suppress the future warning about the 'dual' parameter
    ])

    pipeline.fit(X, y)
    return pipeline


# Evaluate math expressions safely
def evaluate_math_expression(expression: str) -> str:
    operators = {
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
        '/': operator.truediv
    }

    try:
        parts = expression.split()
        operand1 = int(parts[0])
        operator_symbol = parts[1]
        operand2 = int(parts[2])

        if operator_symbol in operators:
            result = operators[operator_symbol](operand1, operand2)
            return str(result)
        else:
            return "Sorry, I couldn't perform the math operation."
    except:
        return "Sorry, I couldn't perform the math operation."


# Retrieve the answer based on the predicted intent
def get_answer_for_intent(intent: str, knowledge_base: dict, user_input: str) -> str:
    if intent == "math":
        expression = user_input.replace("What is", "").strip()
        return evaluate_math_expression(expression)
    else:
        for q in knowledge_base["questions"]:
            if q["intent"] == intent:
                return q["answer"]
        return "I'm sorry, but I don't have an answer for that."


# Main function to handle user input and respond
def chatbot():
    knowledge_base: dict = load_knowledge_base('knowledge_base.json')
    intent_recognition_model = train_intent_recognition_model(knowledge_base)

    while True:
        user_input: str = input("You: ")

        if user_input.lower() == 'quit':
            break

        predicted_intent = intent_recognition_model.predict([user_input])[0]
        answer = get_answer_for_intent(predicted_intent, knowledge_base, user_input)
        print(f"Bot: {answer}")


if __name__ == "__main__":
    chatbot()
