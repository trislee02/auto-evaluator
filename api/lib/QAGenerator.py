import openai
import json

HOW_QUESTION = """Questions must start with 'how'"""
CODING_QUESTION = """Questions must ask students to write/rewrite a code to accomplish some functions in the text. Questions can ask students to fill in missing parts of the code in the text"""

PROMPT = """Given a piece of text/code, you must come up with a question and answer pair than can be used in the test. {question_type}.
Since the test-takers do not have access to the text you are asking about, make sure to come up with detailed questions.
Attach code snippet that your question mentions. If no code needed, leave the 'code_snippet' field blank ("")
Respond in the following JSON format:
```
{{
    "question": "$YOUR_QUESTION_HERE",
    "answer": "$THE_ANSWER_HERE",
    "code_snippet": "$FULL_CODE_HERE"
}}
```
## Remember: Your response must be a valid json"""

PROMPT_CHECK_GIVEN_CODE = """You are helping a teacher to determine whether a code snippet should be attached to a test question.
If the question miss a code snippet or file that needed to show to students, you respond "True".
If the question wants students to provide code or file, you respond "False".
Your response must be valid to cast to a Python Boolean value."""

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
        
        generation_prompt = PROMPT.format(question_type=CODING_QUESTION)

        response = openai.ChatCompletion.create(
            engine=self.engine,
            messages = [{"role":"system","content": generation_prompt},
                        {"role":"user","content": user_content}],
            temperature=0.7,
            max_tokens=800,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None)
        logger.info(response)

        eval_pair = json.loads(response.choices[0].message.content)

        if eval_pair["code_snippet"] != "":
            prompt_for_question = f"Sentence: {eval_pair['question']}\nYour response:"
            response = openai.ChatCompletion.create(
                engine=self.engine,
                messages = [{"role":"system","content": PROMPT_CHECK_GIVEN_CODE},
                            {"role":"user","content": prompt_for_question}],
                temperature=0.7,
                max_tokens=800,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None)
            logger.info(response)

            if response.choices[0].message.content == "True":
                eval_pair["question"] += "\n" + eval_pair["code_snippet"]
            else:
                eval_pair["answer"] += "\n" + eval_pair["code_snippet"]

        return eval_pair

