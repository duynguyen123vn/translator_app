
import gradio as gr
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# --- Cấu hình Mô hình ---
MODEL_PATH = "./mt-en-vi-finetuned-transformer-fast"
MAX_LENGTH = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu")

# --- Khởi tạo Model và Tokenizer ---
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(DEVICE)
    print(f"Model loaded on: {DEVICE}")
except Exception as e:
    print(f"Error loading fine-tuned model: {e}")
    MODEL_CHECKPOINT = "Helsinki-NLP/opus-mt-en-vi"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT).to(DEVICE)
    print("Loaded default pre-trained model as backup.")


def translate_english_to_vietnamese(text: str) -> str:
    # 1. Tokenize Input
    input_ids = tokenizer(
        text, 
        return_tensors="pt", 
        max_length=MAX_LENGTH, 
        truncation=True
    ).input_ids.to(DEVICE)
    
    # 2. Generate Translation
    translated_tokens = model.generate(
        input_ids,
        max_length=MAX_LENGTH,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True,
    )
    
    # 3. Decode Output
    translation = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translation

# --- Xây dựng Giao diện Gradio ---
iface = gr.Interface(
    fn=translate_english_to_vietnamese,
    inputs=gr.Textbox(
        lines=2, 
        placeholder="Enter English sentence here...", 
        label="English Input"
    ),
    outputs=gr.Textbox(
        label="Vietnamese Output"
    ),
    title="Vietnamese-English Translator (MarianMT Fine-tuned)",
    description="A sequence-to-sequence model fine-tuned for high-speed English-Vietnamese translation."
)

if __name__ == "__main__":
    iface.launch()
