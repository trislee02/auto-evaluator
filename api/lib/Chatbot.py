import requests
import json

class Chatbot:
    def __init__(self, chat_endpoint):
        self.chat_endpoint = chat_endpoint

    def run(self, query, logger):
        history = []

        new_message = {"user": query}
        history.append(new_message)

        data = {
            "approach": "rrr",
            "history": history
        }
        
        headers = {'Content-type': 'application/json'}
        response = requests.post(self.chat_endpoint, data=json.dumps(data), headers=headers)
        return json.loads(response.content)["answer"]