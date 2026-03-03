# Deepgram Voice Agent – Insurance Policy Updates

This sample shows how to build an insurance servicing voice agent that keeps customer PII private before it ever reaches the LLM. The flow is:

1. Capture microphone audio locally.
2. Send each utterance to [Deepgram STT](https://developers.deepgram.com/docs/listen) via REST.
3. Run the transcript through the **AI Private Layer** (`ai_private_api`) to mask PII and produce encryption bundles.
4. Send the masked text to OpenAI for reasoning with the domain-specific prompt in `insurance_prompt.py`.
5. Generate the spoken reply through [Deepgram TTS](https://developers.deepgram.com/docs/text-to-speech) and play it back.
6. When the LLM emits `POLICY_UPDATE: {...}`, parse and persist it to `data/policy_updates.json` for auditing.

## Features
- Local capture + Deepgram STT (REST) means you can run the AI Private Layer **before** sending data to the LLM.
- Responses come from OpenAI using the insurance-specific prompt in `insurance_prompt.py`.
- Deepgram TTS REST API returns natural-sounding speech that we play back immediately.
- When the assistant outputs a `POLICY_UPDATE:` line, we persist it to `data/policy_updates.json` via `policy_storage.py`.
- Each session writes a detailed log (raw transcript, sanitized text, OpenAI payloads, bundles) under `data/logs/`.

## Requirements
- Python 3.10+
- A running instance of the **AI Private Layer** (`ai_private_api` folder on Desktop) on `http://127.0.0.1:9000` or a custom URL.
- Deepgram API key with access to `listen` (STT) and `speak` (TTS) endpoints.
- OpenAI API key for the Think step.
- Dependencies from `requirements.txt` (`sounddevice`, `requests`, `openai`, `webrtcvad`, etc.).

## Quick start
1. Create a virtual environment and install dependencies:
   ```bash
   cd Desktop/deepgram_test
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. In a separate terminal start the AI Private Layer (from `~/Desktop/ai_private_api`):
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
7. Inspect the conversation log in `data/logs/session_<timestamp>.log` for raw text, sanitized text, bundles, and assistant replies.

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
- **AI Private Layer Ops** (The core PII proxy server): [https://github.com/Azizbek-Analyst/ai-private-layer-ops](https://github.com/Azizbek-Analyst/ai-private-layer-ops)
- **Chat Order Demo** (Example using text messages for ordering): [https://github.com/Azizbek-Analyst/ai_private_layer_chat_demo](https://github.com/Azizbek-Analyst/ai_private_layer_chat_demo)

