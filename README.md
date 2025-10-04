# Medical Chatbot

A conversational AI assistant designed to answer medical questions, provide symptom assessment, and offer health-related advice. This project integrates NLP techniques, knowledge sources, and chat APIs to build an intelligent medical support bot.

---

## 📌 Motivation
Getting reliable health information can be difficult for non-experts. This chatbot is built to:
- Provide preliminary guidance (not diagnosis).  
- Act as a first step before consulting a healthcare professional.  
- Demonstrate the use of NLP, retrieval methods, and conversational AI.  

---

## ✨ Features
- Natural language interface for medical queries.  
- Symptom assessment through rules/models.  
- Conversational memory to maintain multi-turn dialogue.  
- Response generation using a language model.  
- Safety layer with disclaimers for sensitive or critical cases.  

---

## 🏗️ Architecture


- **Chat Handler**: Controls conversation flow.  
- **Parser**: Detects intent (symptom, treatment, condition).  
- **Knowledge Base**: Uses datasets or embeddings for answers.  
- **Response Generator**: Provides natural responses.  
- **Safety Layer**: Adds disclaimers, avoids risky outputs.  

---

## ⚙️ Installation
```bash
git clone https://github.com/shreyasjadhav996/medical-chatbot.git
cd medical-chatbot

# (Optional) create virtual environment
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## How to run
```bash
python main.py
```
## Example
```bash
You: I have fever and headache  
Bot: How long have you been experiencing these symptoms? ...
```

## 🔑 Configuration
- OPENAI_API_KEY=your_api_key_here
- MODEL_NAME=gpt-3.5-turbo
- DATA_PATH=./data/medical_docs.json
- MAX_HISTORY=5

