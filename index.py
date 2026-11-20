import os
from dotenv import load_dotenv
from transformers import pipeline
import torch

load_dotenv (override=True)
api_key = os.getenv("OPENAI_API_KEY")


device = 0 if torch.cuda.is_available() else -1
dtype = torch.float16 if torch.cuda.is_available() else torch.float32


transcriber = pipeline(
    "automatic-speech-recognition", 
    model="openai/whisper-base",     
    chunk_length_s=30,               
    device=device,                 
    torch_dtype=dtype,              
)


generator = pipeline(
    "text-generation",               
    model="Qwen/Qwen2.5-1.5B-Instruct",
    device=device,
    torch_dtype=dtype,
)


def generate_meeting_minutes(audio_path: str) -> dict:
    asr_result = transcriber(audio_path)
    transcript = asr_result["text"].strip()


    messages = [
        {
            "role": "system",
            "content": (
                "You are an executive assistant. Convert the provided meeting transcript "
                "into clean, structured meeting minutes using these exact bold headers:\n"
                "**Executive Summary**\n"
                "**Key Discussion Points**\n"
                "**Decisions Made**\n"
                "**Action Items** (include owner and task if mentioned)"
            ),
        },
        {
            "role": "user",
            "content": f"Here is the meeting transcript:\n\n{transcript}",
        },
    ]


    output = generator(
        messages,
        max_new_tokens=512,
        temperature=0.3,
        do_sample=True,
    )

    

    minutes_markdown = output[0]["generated_text"][-1]["content"]

if __name__ == "__main__":
    result = generate_meeting_minutes("sample_meeting.wav")
    print("\n--- TRANSCRIPT ---\n", result["transcript"])
    print("\n--- MEETING MINUTES ---\n", result["minutes"])