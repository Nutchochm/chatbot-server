from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

def load_chkans_model(model_name):
    print(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModel.from_pretrained(model_name, local_files_only=True)
    model.eval()

    return model, tokenizer

def tokenize_input(question, answer, tokenizer):
    # Combine question and answer to give context
    combined_input = question + " [SEP] " + answer
    return tokenizer(answer, return_tensors='pt', padding=True, truncation=True)

def get_embedding(input_encodings, model):
    with torch.no_grad():
        outputs = model(**input_encodings)
        # Use the [CLS] token representation for sentence embedding
        embeddings = outputs.pooler_output
        return embeddings


def compute_similarity(embedding1, embedding2):
    # Compute cosine similarity
    similarity = F.cosine_similarity(embedding1, embedding2)
    return similarity.item()

def rating_scores(similarity_score):
    # Compute rating with >= 0.80 score_max is 10
    # rating = (((similarity_score - 0.8) / 0.2) * 9) + 1
    if similarity_score < 0.5:   
        return 0
    elif 0.5 <= similarity_score < 0.8:
        rating = (((similarity_score - 0.5) / 0.29) * 4) + 1   
    else:            
        rating = (((similarity_score - 0.8) / 0.2) * 4) + 6 
    
    return round(min(rating, 10))

"""
    input_question = "อํานาจหน้าที่ของคณะกรรมการวิธีปฏิบัติราชการทางปกครอง ข้อใดกล่าวไม่ถูกต้อง ?"
    correct_answer = "จัดทํารายงานเกี่ยวกับการปฏิบัติตามพระราชบัญญัตินี้เสนอนายกรัฐมนตรี"
    input_answer   = "จัดทํารายงานเกี่ยวกับการปฏิบัติตามพระราชบัญญัตินี้เสนอนายกรัฐมนตรี"

    input_encodings = tokenize_input(input_question, input_answer)
    correct_encodings = tokenize_input(input_question, correct_answer)
    
    input_embedding = get_embedding(input_encodings)
    correct_embedding = get_embedding(correct_encodings)

    print(correct_answer)
    print(input_answer)
    
    similarity_score = compute_similarity(input_embedding, correct_embedding)
    print(f"Similarity Score: {similarity_score:.2f}")

    ratings = rating_scores(similarity_score)

    # Threshold for human-acceptable similarity
    if similarity_score >= 0.8:        
        print(f"Answer is correct or highly similar. given score: {ratings}")
    elif 0.5 <= similarity_score < 0.8:
        print("Answer is partially correct.", ratings)
    else:
        print("Answer is incorrect.", ratings)

    return similarity_score


model_name = "model/wangchanberta"
check_ans(model_name)"""