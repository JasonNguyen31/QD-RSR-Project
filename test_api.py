import os
from dotenv import load_dotenv
from openai import OpenAI
from google import genai

load_dotenv()
Q = "Tinh 17 * 23. Chi tra loi so, khong giai thich."

orc = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

TEACHERS = {
    "DeepSeek-V3":    "deepseek/deepseek-chat",
    "Qwen2.5-72B":    "qwen/qwen-2.5-72b-instruct",
    "Llama-3.3-70B":  "meta-llama/llama-3.3-70b-instruct",
}

for ten, model_id in TEACHERS.items():
    try:
        r = orc.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": Q}],
            max_tokens=50,
        )
        print(f"[OK]  {ten:16s} -> {r.choices[0].message.content.strip()[:40]}")
    except Exception as e:
        print(f"[LOI] {ten:16s} -> {type(e).__name__}: {str(e)[:120]}")

try:
    g = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    r = g.models.generate_content(model="gemini-3.7-flash", contents=Q)
    print(f"[OK]  {'Gemini Flash':16s} -> {r.text.strip()[:40]}")
except Exception as e:
    print(f"[LOI] {'Gemini Flash':16s} -> {type(e).__name__}: {str(e)[:120]}")
