from difflib import Differ

import gradio as gr

import openai

import json

import pandas as pd

async def diff_texts(text1, text2):
    
    result = await generate_text(text1, text2)

    print(result)

    try:
        parsed_json = json.loads(result)
    except Exception as e:
        print(e)
        gr.Warning("Invalid JSON string!")
        return [
            "Invalid JSON string!",
            "Invalid JSON string!",
            "Invalid JSON string!",
            0,
        ]

    reviewed_text = parsed_json["reviewed_text"]
    explanation = parsed_json["explanation"]
    score = "Overall score: {0}".format(parsed_json["score"])
    sentences = pd.DataFrame.from_dict(parsed_json["sentences"])

    d = Differ()
    return [
        [(token[2:], token[0] if token[0] != " " else None) for token in d.compare(text1, reviewed_text)],
        reviewed_text,
        explanation,
        score,
        sentences,
    ]

async def generate_text(text_prompt, context):

    text_system = f"""You are an excellent English teacher. 
    Review the following text and return it with corrections. Do not change the text structure.
    Feel free to add or remove words or suggest different expressions if you think they will improve the text, but do not change the text structure.
    Consider that the text needs to be polite and friendly.
    Your return must be in JSON format with four properties: 
    "reviewed_text" which should include the text with corrections, 
    "explanation" which should include a very detailed explanation of the corrections you made,
    For instance, ['Grammar', 'Punctuation', 'Preposition', 'Verb tense', 'Expression'].
    "score" which should be a number between 0 and 100 indicating how well you think the text was written. 
    "sentences" which should be an array with objects with four properties:
    "sentence" which should be the sentences of the text,
    "review" which should be the reviewed sentence,
    "comments" which should be your comments about this sentence,
    and "score" which should be a number between 0 and 100 indicating how well you think the sentence was written.
    Consider the follwoing context for the text: "{context}"."""

    prompt = [
        {
            "role": "system",
            "content": f"""{text_system}""",
        },
        {
            "role": "user",
            "content": f"""{text_prompt}""",
        }
    ]

    print(prompt)
    return await completion(prompt, model="gpt-3.5-turbo-16k", temperature=0.1, max_tokens=1000)
    
async def generate_practice(sample):
    text_system = f"""
    You are an excellent English teacher. 
    You are practicing with a student.
    The student gave you the following sentence: "{sample}" to be reviewed.
    Respond to the user message with a completely different sentence that contains simmilar grammar issues. 
    Your sentence must contain at least one grammar issue.
    The student will review the sentence you generated and submit it back to you.
    """

    prompt = [
        {
            "role": "system",
            "content": f"""{text_system}""",
        },
    ]

    print(prompt)

    result = await completion(prompt, model="gpt-3.5-turbo", temperature=0.3, max_tokens=200)
    return [result, result]

async def generate_practice_feedback(sample, answer):
        
    text_system = f"""
    You are an excellent English teacher. 
    
    You are practicing with a student. You gave him the following sentence: "{sample}" to be reviewed.

    The student replied: "{answer}".

    Analyze the student's review and provide feedback.

    """

    prompt = [
        {
            "role": "system",
            "content": f"""{text_system}""",
        },
    ]

    print(prompt)
    return await completion(prompt, model="gpt-3.5-turbo", temperature=0, max_tokens=700)
    
async def generate_practice_hint(sample):
        
    text_system = f"""
    You are an excellent English teacher. 
    
    You are practicing with a student. He provided you the following sentence: "{sample}" to be reviewed.

    Analyze the student's sentence and provide hints about the grammatical mistakes. Do not provide the corrected version of the sentence.

    """

    prompt = [
        {
            "role": "system",
            "content": f"""{text_system}""",
        },
    ]

    print(prompt)

    return await completion(prompt, model="gpt-3.5-turbo", temperature=0, max_tokens=500)
    
async def completion(prompt, model="gpt-3.5-turbo", temperature=0, max_tokens=200):
    try:
        creation = openai.ChatCompletion.create(
            model=model,
            messages=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        print(creation)
        return creation.choices[0].message.content
    except Exception as e:
        gr.Warning(e)
        return "Error"
    
async def update_api_key(api_key):
    openai.api_key = api_key
    try:
        openai.api_key = api_key
        openai.Model.list()
        return "API key set"
    except Exception as e:
        print(e)
        return "Invalid API key"


with gr.Blocks(title="Grammarly") as demo:
    with gr.Row():
        with gr.Column(scale=6):
            gr.Markdown(
                value="# English Grammar Correction and Practice",
            )
        with gr.Column():
            tb_api_key = gr.Textbox(
                interactive=True,
                type="password",
                placeholder="Enter your OpenAI API key and press Enter",
                container=False,
            )
        with gr.Column():
            lb_api_status = gr.Markdown(
                value=""" API key not set """,
            )
            
    with gr.Row():
        with gr.Column(variant="panel"):
            with gr.Row():
                gr.Markdown(
                    value="## Input Text",
                )
            with gr.Tab("Initial Text"):
                with gr.Row():
                    input = gr.Textbox(
                        info="The text you want to be corrected",
                        lines=28,
                        placeholder="Hi. I am really sorry about what happened with you. I will try to help you as much as I can.",
                        show_label=False,
                    )
            with gr.Tab("Context"):
                with gr.Row():
                    context = gr.Textbox(
                        info="Provide some context for the text",
                        lines=28,
                        placeholder="This is an email message in response to a customer complaint.",
                        show_label=False,
                    )
            with gr.Row():
                btn_submit = gr.Button(
                    value="Send",
                    variant="primary",
                    elem_id="submit",
                )
                
                reset = gr.Button(
                    value="Reset",
                )
        with gr.Column(variant="panel", scale=2):
            with gr.Row():
                with gr.Column():
                    gr.Markdown(
                        value="## Text Analysis",
                    )
                with gr.Column():
                    score = gr.Markdown()
                    
            with gr.Row():
                with gr.Tab("Reviewed"):
                    reviewed = gr.Textbox(
                        label="Reviewed text",
                        lines=12,
                        interactive=False,
                    )
                with gr.Tab("Difference"):
                    diff = gr.HighlightedText(
                        label="Diff",
                        combine_adjacent=True,
                        show_legend=True,
                        color_map={"+": "green", "-": "red"} 
                    )
                with gr.Tab("Explanation"):
                    explanation = gr.Textbox(
                        label="Explanation",
                        lines=10,
                        interactive=False,
                    )
                with gr.Tab("Sentences"):
                    gr.Label(
                        value="Select a sentence to practice",
                        container=False,
                    )
                    sentences = gr.DataFrame(
                        headers=["Sentence", "Review", "Comments", "Score"],
                        datatype=["str", "str", "str", "number"],
                        wrap=True,
                        height=500,
                    )
                    
                with gr.Tab("Practice"):
                    with gr.Row():
                        statement = gr.Label(
                            value="No sentence selected",
                            container=False,
                            scale=6,
                        )
                    with gr.Row():
                        btn_practice = gr.Button(
                            value="Start practicing",
                        )

                    with gr.Row():
                        practice_sentence = gr.Label(
                            container=True,
                            scale=3,
                            label="Practice sentence",
                        )
                        answer = gr.Textbox(
                            lines=3,
                            placeholder="Improve the sentence",
                            show_label=False,
                            scale=3,
                            label="Improved sentence",
                        )

                    with gr.Row():
                        btn_submit_practice = gr.Button(
                            value="Submit",
                            variant="primary",
                        )
                        btn_hint_practice = gr.Button(
                            value="Hint",
                        )
                        btn_next_practice = gr.Button(
                            value="Generate new sentence",
                        )

                    with gr.Row():
                        practice_feedback = gr.Label(
                            value="Feedback",
                            label="Feedback",
                        )


                
                
                
        
    btn_submit.click(fn=diff_texts, inputs=[input, context], outputs=[diff, reviewed, explanation, score, sentences])  
    
    def on_select(evt: gr.SelectData):  # SelectData is a subclass of EventData
        return f"{evt.value}"

    sentences.select(on_select, None, statement)

    btn_practice.click(fn=generate_practice, inputs=[statement], outputs=[practice_sentence, answer])
    btn_submit_practice.click(fn=generate_practice_feedback, inputs=[practice_sentence, answer], outputs=[practice_feedback])
    btn_next_practice.click(fn=generate_practice, inputs=[practice_sentence], outputs=[practice_sentence, answer])
    btn_hint_practice.click(fn=generate_practice_hint, inputs=[practice_sentence], outputs=[practice_feedback])

    tb_api_key.submit(fn=update_api_key, inputs=[tb_api_key], outputs=[lb_api_status])

if __name__ == "__main__":
    demo.launch()

