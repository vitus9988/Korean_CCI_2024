import json
from openai import OpenAI
from tqdm import tqdm

# This script generates answer sheets using the GPT API to self-evaluate the performance of the fine-tuned model.

key = "OpenAI API KEY"
client = OpenAI(api_key=key)

with open("../resource/data/test.json", 'r') as file:
    data = json.load(file)

ans_data = []

for i in tqdm(data):
    convers = ""
    conv = i['input']['conversation']
    convers += "먼저 아래 <대화내용>의 내용을 파악하고 숙고하라. 그리고 <문제>를 파악하고 판단하여 <대화내용>의 대화맥락으로 가장 일치한다고 판단되는 보기를 'inference_1', 'inference_2', 'inference_3' 중 하나를 골라 대답하라.\n\n<대화내용>\n"
    
    for k in conv:
        convers += f"화자{k['speaker']}: {k['utterance']}"
        convers += '\n'
        
    category = i['input']['category']
    inf_1 = i['input']['inference_1']
    inf_2 = i['input']['inference_2']
    inf_3 = i['input']['inference_3']
    
    mat = f"\n<문제>\n대화맥락: {category}\ninference_1: {inf_1}\ninference_2: {inf_2}\ninference_3: {inf_3}\n"
    
    convers += mat
    
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "당신은 대화맥락추론용 ai입니다. 질문에 대한 응답은 'inference_1', 'inference_2', 'inference_3' 중에 하나로만 대답하세요"},
            {
                "role": "user",
                "content": f"{convers}"
            }
        ]
    )
    gpt_ans = completion.choices[0].message.content
    ans_data.append(gpt_ans)

with open("../resource/data/gpt4_answer.json", 'w', encoding='utf-8') as f:
    json.dump(ans_data, f, ensure_ascii=False, indent=4)