from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoModel, AutoTokenizer

# Khởi tạo FastAPI
app = FastAPI()
device = torch.device("cpu") 

# Load PhoBERT model
class PhoBERTClassifier(torch.nn.Module):
    def __init__(self, phobert_model):
        super(PhoBERTClassifier, self).__init__()
        self.phobert = phobert_model
        self.fc = torch.nn.Linear(768, 3)

    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0][:, 0, :]
        return self.fc(pooled_output)

phobert = AutoModel.from_pretrained("vinai/phobert-base").to(device) 
model = PhoBERTClassifier(phobert).to(device)  # Chạy trên CPU
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

model.load_state_dict(torch.load("phobert_sentiment_model.pth", map_location=device))
model.eval()

class TextRequest(BaseModel):
    text: str

# Hàm dự đoán cảm xúc
def predict_sentiment(text):
    # Tokenize dữ liệu
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    
    input_ids = tokens["input_ids"].to(device)  # Chuyển sang CPU
    attention_mask = tokens["attention_mask"].to(device)  # Chuyển sang CPU

    with torch.no_grad():
        output = model(input_ids, attention_mask)  # Model chạy trên CPU
        pred = torch.argmax(output, dim=1).numpy()[0]  # Không cần .cpu() vì đã ở CPU

    return ["negative", "neutral", "positive"][pred]



print(f"Model device: {next(model.parameters()).device}")
@app.get("/")
def health_check():
    return {"status": "API is running"}
# Endpoint API
@app.post("/predict")
def predict(request: TextRequest):
    sentiment = predict_sentiment(request.text)
    return {"sentiment": sentiment}
