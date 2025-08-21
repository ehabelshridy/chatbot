# file: chat_cli.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "microsoft/DialoGPT-small"  # Open-source lightweight model

def load_model(model_name=MODEL_NAME, device=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device

def generate_reply(user_text, tokenizer, model, device, chat_history_ids=None,
                   max_new_tokens=200, top_p=0.92, top_k=50, temperature=0.7):
    # Prepare user input and append EOS token
    new_user_input_ids = tokenizer.encode(
        user_text + tokenizer.eos_token, return_tensors="pt"
    ).to(device)

    # Concatenate with chat history if available
    if chat_history_ids is not None:
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
    else:
        bot_input_ids = new_user_input_ids

    # Generate bot reply
    output_ids = model.generate(
        bot_input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Extract the newly generated tokens (bot's reply)
    reply_ids = output_ids[0, bot_input_ids.shape[-1]:]
    reply_text = tokenizer.decode(reply_ids, skip_special_tokens=True)
    return reply_text, output_ids

def main():
    print("💬 Simple HF Chatbot (DialoGPT) — type 'exit' to quit")
    tokenizer, model, device = load_model()
    chat_history_ids = None

    while True:
        user_text = input("👤 You: ").strip()
        if user_text.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break

        reply, chat_history_ids = generate_reply(
            user_text, tokenizer, model, device, chat_history_ids
        )
        print(f"🤖 Bot: {reply}")

if __name__ == "__main__":
    main()
