# Install dependencies before running:
# pip install flask transformers torch

from flask import Flask, request, jsonify
from transformers import pipeline

# Initialize Flask app
app = Flask(__name__)

# Load a pretrained model for question answering
qa_pipeline = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad')

# Example FAQ knowledge base
knowledge_base = {
    "return policy": "Our return policy allows returns within 30 days of purchase with a receipt.",
    "shipping time": "Shipping typically takes 5-7 business days depending on your location.",
    "support hours": "Our customer support is available from 9 AM to 6 PM, Monday to Friday."
}

def search_knowledge_base(question):
    """Simple keyword matching search"""
    for key, answer in knowledge_base.items():
        if key in question.lower():
            return answer
    return None

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')

    # First, search the knowledge base
    kb_answer = search_knowledge_base(user_input)
    if kb_answer:
        return jsonify({"response": kb_answer})

    # If no direct match, use QA model to find an intelligent answer
    context = " ".join(knowledge_base.values())  # Combine all FAQs into a single context
    result = qa_pipeline(question=user_input, context=context)

    return jsonify({"response": result['answer']})

if __name__ == '__main__':
    app.run(debug=True)# Group--09
