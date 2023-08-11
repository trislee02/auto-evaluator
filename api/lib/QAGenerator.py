import openai
import json

PROMPT = """You are a smart assistant designed to help teachers come up with questions for a final Computer Science test.
Given a piece of text, you must come up with a clear and detailed question and answer pair than can be used to test a student.
When coming up with this question-answer pair, you must respond in the following JSON format:
```
{
    "question": "$YOUR_QUESTION_HERE",
    "answer": "$THE_ANSWER_HERE"
}
```
Remember:
++ Your response must be a valid json. 
++ Do not mention the tutorial.
++ When giving a question regarding code, add the code snippet in the question if applicable"""

class QAGenerator:
    def __init__(self, api_base, azure_openai_key, engine):
        openai.api_type = "azure"
        openai.api_base = api_base
        openai.api_version = "2023-03-15-preview"
        openai.api_key = azure_openai_key
        self.engine = engine

    def run(self, text, logger):
        """
        Generate a question-answer pair from text.
        If the llm response is not in json type, raise JSONDecodeError
        """
        user_content = f"""Text:
<text>
{text}
</text>"""
        response = openai.ChatCompletion.create(
            engine=self.engine,
            messages = [{"role":"system","content": PROMPT},
                        {"role":"user","content": user_content}],
            temperature=0.7,
            max_tokens=800,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None)
        return json.loads(response.choices[0].message.content)

