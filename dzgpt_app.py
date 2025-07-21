import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import datetime

# Chargement du modèle arabe DZ
model_name = "akhooli/gpt2-small-arabic"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Fonction principale avec historique

def chat_fn(message, history):
    prompt = f"مستخدم: {message}\nالذكاء الاصطناعي:"
    result = generator(prompt, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    response = result[0]["generated_text"].split("الذكاء الاصطناعي:")[-1].strip()

    # Sauvegarde historique
    with open("historique.txt", "a", encoding="utf-8") as f:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{now}] Utilisateur: {message}\nRéponse: {response}\n---\n")

    return response

# Interface Gradio avec logo
with gr.Blocks(theme="soft") as demo:
    gr.Image(value="logo.png", show_label=False, show_download_button=False, container=False, height=150)

    gr.ChatInterface(
        fn=chat_fn,
        title="DZGPT 🇩🇿 | دردش مع الذكاء الاصطناعي",
        description="تحدث مع بوت ذكي باللهجة الجزائرية أو العربية. يدعم الصوت والردود المنطوقة.",
        examples=["كيفاش داير؟", "أعطيني نكتة", "ترجملي للفرنسية"],
        retry_btn="🔁 إعادة",
        clear_btn="🧹 مسح",
        submit_btn="🗣️ أرسل",
        input_audio="microphone",
        output_audio="auto",
        textbox=gr.Textbox(placeholder="أكتب هنا...", container=True, scale=7)
    )

demo.launch()
