# Deepgram Voice Agent with AI Private Layer

This repository demonstrates how to build a voice agent that is intrinsically safe for highly regulated industries (e.g., healthcare, finance, insurance) utilizing **AI Private Layer Ops** for real-time PII masking.

In environments governed by strong data protection regulations like GPA (Global Privacy Assembly standards) or GDPR, sending raw customer audio transcripts directly to a cloud LLM is unacceptable. This project shows how to intercept and mask Sensitive Data / Personally Identifiable Information (PII) **before** it ever leaves your secure perimeter.

The flow is:

1. Capture microphone audio locally.
2. Send each utterance to [Deepgram STT](https://developers.deepgram.com/docs/listen) via REST.
3. **Data Protection:** Run the transcript through the **AI Private Layer** (`ai_private_api`) to detect and mask PII, ensuring that only anonymized text is exposed.
4. Send the *masked* text to OpenAI for reasoning with the domain-specific prompt.
5. Generate the spoken reply through [Deepgram TTS](https://developers.deepgram.com/docs/text-to-speech) and play it back.
6. When the LLM emits a business payload like `POLICY_UPDATE: {...}`, extract it and securely map back the required entities if needed.

## Why this matters (GPA & GDPR Compliance)
- **Data Minimization:** Only necessary, non-identifiable data reaches the LLM.
- **Privacy by Design:** Security is built into the architecture. The **AI Private Layer** acts as a reliable filter.
- **Auditability:** You can log the exact state of the masked payloads sent out of your network.

## Features
- Local capture + Deepgram STT (REST) means you can run the AI Private Layer **before** sending data to the LLM.
- Responses come from OpenAI using the insurance-specific prompt in `insurance_prompt.py`.
- Deepgram TTS REST API returns natural-sounding speech that we play back immediately.
- When the assistant outputs a `POLICY_UPDATE:` line, we persist it to `data/policy_updates.json` via `policy_storage.py`.
- Each session writes a detailed log (raw transcript, sanitized text, OpenAI payloads, bundles) under `data/logs/`.

## Requirements
- Python 3.10+
- A running instance of the **AI Private Layer** (`ai_private_api` folder) on `http://127.0.0.1:9000` or a custom URL.
- Deepgram API key with access to `listen` (STT) and `speak` (TTS) endpoints.
- OpenAI API key for the Think step.
- Dependencies from `requirements.txt` (`sounddevice`, `requests`, `openai`, `webrtcvad`, etc.).

## Quick start
1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. In a separate terminal start your AI Private Layer instance (from `~/Desktop/ai_private_api` or wherever you cloned the ops repo):
   ```bash
   cd ~/Desktop/ai_private_api
   source .venv/bin/activate  # if you created one
   uvicorn detect_app:app --host 0.0.0.0 --port 9000
   ```
3. Copy `.env.example` to `.env` and set the required keys:
   ```ini
   DEEPGRAM_API_KEY=dg_xxx
   OPENAI_API_KEY=sk-...
   PRIVATE_LAYER_URL=http://127.0.0.1:9000
   PRIVATE_LAYER_API_KEY=dev-secret-demo  # match tenants.yaml
   PRIVATE_LAYER_TENANT=default            # or qic, etc.
   PRIVATE_LAYER_TIMEOUT=90                # optional: give GLiNER time to warm up
   ```
   Adjust `DEEPGRAM_LISTEN_MODEL`, `DEEPGRAM_SPEAK_MODEL`, or `OPENAI_MODEL` if needed.
   On the very first request the AI Private Layer downloads the GLiNER model, so keep the
   server running between calls or set `PRIVATE_LAYER_TIMEOUT` higher to avoid client timeouts.
4. Run the agent:
   ```bash
   python run_insurance_agent.py
   ```
5. Speak with the agent about a policy change. Each turn is transcribed, sanitized, and passed to OpenAI.
6. When the customer confirms, the LLM emits a `POLICY_UPDATE:` message which is saved to `data/policy_updates.json`.
7. Inspect the conversation log in `data/logs/` for raw text, sanitized text, bundles, and assistant replies.

## Output format
Each saved record looks like this:
```json
{
  "id": 1,
  "policy_number": "ABC12345",
  "old_name": "Ivan Petrov",
  "new_name": "Natalia Petrova",
  "date_of_changes": "2025-02-15",
  "phone_number": "+1-555-123-4567",
  "details": "Add Natalia as an additional driver on the auto policy",
  "raw": "POLICY_UPDATE: ...",
  "created_at": "2025-02-19T10:45:21Z"
}
```

## Related Projects
- **AI Private Layer Ops** (The core PII proxy server toolkit): [https://github.com/Azizbek-Analyst/ai-private-layer-ops](https://github.com/Azizbek-Analyst/ai-private-layer-ops)
- **Chat Order Demo** (Example using text messages for ordering): [https://github.com/Azizbek-Analyst/ai_private_layer_chat_demo](https://github.com/Azizbek-Analyst/ai_private_layer_chat_demo)
