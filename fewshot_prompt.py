from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
import torch
import os
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)

def loadtyphoon2(modelconfig):
    load_path = os.path.join("model_trained", modelconfig)
    #load_path = "scb10x/llama3.2-typhoon2-1b-instruct"
    # Reload the saved quantized model
    typhoon2_model = AutoModelForCausalLM.from_pretrained(
        load_path,
        device_map="auto",
        local_files_only=True
    )
    typhoon2_tokenizer = AutoTokenizer.from_pretrained(load_path, local_files_only=True)
    return typhoon2_model, typhoon2_tokenizer

def typhoon2chat(query, typhoon2_model, typhoon2_tokenizer):
    messages = [
        {"role": "system", "content": "You are a male AI assistant named Typhoon created by SCB 10X to be helpful, harmless, and honest. Typhoon is happy to help with analysis, question answering, math, coding, creative writing, teaching, role-play, general discussion, and all sorts of other tasks. Typhoon responds directly to all human messages without unnecessary affirmations or filler phrases like “Certainly!”, “Of course!”, “Absolutely!”, “Great!”, “Sure!”, etc. Specifically, Typhoon avoids starting responses with the word “Certainly” in any way. Typhoon follows this information in all languages, and always responds to the user in the language they use or request. Typhoon is now being connected with a human. Write in fluid, conversational prose, Show genuine interest in understanding requests, Express appropriate emotions and empathy. Also showing information in term that is easy to understand and visualized."},
        {"role": "user", "content": query},
    ]

    input_ids = typhoon2_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)

    terminators = [
        typhoon2_tokenizer.eos_token_id,
        typhoon2_tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    ## Typhoon 1b need low temperature to inference.
    outputs = typhoon2_model.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.4,
        top_p=0.9,
    )

    response = outputs[0][input_ids.shape[-1]:]
    #print("Token decoded",typhoon2_tokenizer.decode(response, skip_special_tokens=True))
    return typhoon2_tokenizer.decode(response, skip_special_tokens=True)

"""
examples = [
    ("วัสดุที่นิยมใช้ทำตัวกีตาร์คืออะไร", "ตัวกีตาร์มักทำจากไม้ที่มีคุณภาพสูง เช่น ไม้เมเปิ้ล (Maple), ไม้โรสวูด (Rosewood), ไม้มะฮอกกานี (Mahogany) ซึ่งมีคุณสมบัติให้เสียงที่แตกต่างกันตามประเภทของไม้ที่เลือกใช้", "หนังสือที่เกี่ยวข้องกับการสร้างกีตาร์"),
    ("ขั้นตอนการขึ้นรูปตัวกีตาร์ทำอย่างไร","การขึ้นรูปตัวกีตาร์เริ่มจากการเลือกไม้คุณภาพดี ตัดไม้ให้ได้รูปทรงตามแบบ จากนั้นใช้เครื่องมืออย่างมือกบหรือเครื่องขัดไม้ในการทำให้พื้นผิวเรียบเนียนเพื่อประกอบเป็นตัวกีตาร์", "บทความจากเว็บการสร้างกีตาร์"),
    ("ทำไมสายกีตาร์ถึงต้องใช้วัสดุต่างชนิดกัน","สายกีตาร์ทำจากวัสดุหลากหลาย เช่น นิกเกิล, สแตนเลส หรือไนลอน เพื่อให้ได้เสียงที่แตกต่างกัน โดยสายโลหะมักให้เสียงคมชัด ส่วนสายไนลอนให้เสียงนุ่มและเหมาะกับคลาสสิก", "คู่มือการเลือกสายกีตาร์"),
    ("ส่วนประกอบหลักของกีตาร์มีกี่ส่วน","กีตาร์มีส่วนประกอบหลัก 3 ส่วน คือ คอกีตาร์ (Neck), ตัวกีตาร์ (Body) และหัวกีตาร์ (Headstock) โดยแต่ละส่วนมีฟังก์ชันเฉพาะเพื่อสร้างเสียงและความสะดวกสบายในการเล่น", "www.google.com"),
    ("การเคลือบผิวกีตาร์สำคัญอย่างไร","การเคลือบผิวช่วยป้องกันไม้จากความชื้นและรอยขีดข่วน รวมถึงเพิ่มความเงางามและความทนทาน การเคลือบยังส่งผลต่อเสียงสะท้อนที่เกิดจากตัวกีตาร์อีกด้วย", "https://www.facebook.com/TheGuitarMag/?locale=th_TH")
]
"""
examples = [
    ("",""),
    ("","")
]

def construct_prompt(query, examples):
    """if not isinstance(examples, (list, tuple)):
        raise TypeError("Expected 'examples' to be a list or tuple of (question, answer) pairs.")

    for item in examples:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"Invalid example format: {item}. Expected (question, answer) pair.")
"""
    prompt = query
    for q, a in examples:
        prompt += f"\nQuestion: {q}\nAnswer: {a}\n\n"

    prompt += f"Q: {query}\nA:"
    return prompt

def cal_perplexity(response_text, model, tokenizer):
    # |  perplexity  |   Confident  |
    # |   1.0 - 10   |    Best      |
    # |   10 - 100   |    Middle    |
    # |     100 +    |    Worst     |
    inputs = tokenizer(response_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    perplexity = torch.exp(loss)
    return perplexity

def cal_bleu_rouge(reference, generated):
    rouge = Rouge()
    bleu_score = sentence_bleu([reference.split()], generated.split())
    # 0 = bad match,   1.0 = Perfect match
    rouge_score = rouge.get_scores(generated, reference)[0]
    # 0 = bad match,   1.0 = Perfect match
    return bleu_score, rouge_score


"""
user_prompt = "อธิบายส่วนประกอบของกีตาร์"
Correct_answer = ""

generated_prompt = construct_prompt(user_prompt, examples)

print(f"Generate Few-Show Prompt: {generated_prompt}")
modelconfig = "local_typhoon2_1b"

typhoon2_model, typhoon2_tokenizer = loadtyphoon2(modelconfig)
response = typhoon2chat(generated_prompt, typhoon2_model, typhoon2_tokenizer)

perplexity_scored = cal_perplexity(response, typhoon2_model, typhoon2_tokenizer)
bleu, rouge = cal_bleu_rouge(Correct_answer, response)

print(f"Model Response : {response}")
print(f"Few Shot Perplexity Scores : {perplexity_scored}")
print(f"BLEU Score : {bleu}")
print(f"ROUGE Score: {rouge}")
"""
