import requests

AI_SERVER_URL = "https://meta-llama-3-70b1.p.rapidapi.com/"
class AIServer:
    def __init__(self):
        self.headers = {
            'x-rapidapi-key': "c231564601mshf9ebdbfb9bf8045p14bc3ajsn58b3138b1e56",
            'x-rapidapi-host': "meta-llama-3-70b1.p.rapidapi.com",
            'Content-Type': "application/json"
        }

    def generate_triple(self, text):
        msg = []
        content = "Please generate a triple from following text.\n"
        content += text
        msg.append(
            {
                "role": "user",
                "content": content
            }
        )
        data = {
            "model": "meta-llama/Llama-3-70b-chat-hf",
            "temperature": 0,
            "messages": msg
        }
        response = requests.post(AI_SERVER_URL, json=data, headers=self.headers)

        result = response.json()
        result = result["choices"][0]["message"]["content"]
        return result