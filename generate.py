import torch
from model.gpt2 import miniGPT2
from config import GPT2Config
from tokenizer import tokenizer

def generate(model, tokenizer, prompt, max_length=50):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    input_ids = input_ids.to(next(model.parameters()).device)
    
    for _ in range(max_length):
        outputs = model(input_ids)
        next_token_logits = outputs[:,-1,:]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        input_ids = torch.cat((input_ids, next_token), dim=1)
        
    return tokenizer.decode(input_ids[0])

#Example usage

if __name__=='__main__': 
    model = miniGPT2(GPT2Config())
    model.load_state_dict(torch.load('./minigpt2.pth'))
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    prompt = input("Enter prompt: ")
    generated_text = generate(model, tokenizer, prompt)
    print("miniGPT2:",generated_text)