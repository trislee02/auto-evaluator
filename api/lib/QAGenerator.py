import openai
import json

PROMPT = """You are a smart assistant designed to help high school teachers come up with reading comprehension questions. 
Given a piece of text, you must come up with a question and answer pair than can be used to test a student's reading comprehension abilities.
When coming up with this question-answer pair, you must respond in the following format:
```
{
    "question": "$YOUR_QUESTION_HERE",
    "answer": "$THE_ANSWER_HERE",
}
```
Everything between the ``` must be valid json."""

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
        user_content = f"""Text: {text}"""
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

