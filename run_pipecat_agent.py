"""Insurance voice agent: Deepgram STT/TTS + Private Layer + OpenAI via Pipecat."""
from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime
from typing import Optional

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import EndFrame, Frame, StartFrame, TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.aggregators.sentence import SentenceAggregator
from pipecat.services.deepgram import DeepgramSTTService, DeepgramTTSService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.local.audio import LocalAudioTransport

from config import (
    DEEPGRAM_API_KEY,
    DEEPGRAM_LISTEN_MODEL,
    DEEPGRAM_SPEAK_MODEL,
    LOGS_DIR,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    PRIVATE_LAYER_API_KEY,
    PRIVATE_LAYER_DETECT_PATH,
    PRIVATE_LAYER_TENANT,
    PRIVATE_LAYER_URL,
)
from insurance_prompt import build_system_prompt, get_greeting
from pipecat_privacy_filters import (
    PrivateLayerDecryptFilter,
    PrivateLayerEncryptFilter,
    PrivateLayerSession,
)
from policy_storage import PolicyUpdate, save_policy_update
from private_layer_client import PrivateLayerClient

import openai

import httpx

async def _extract_policy_update_from_context(messages: list, log_filename: str) -> Optional[PolicyUpdate]:
    """Offline extraction using OpenAI structured JSON output after chat ends."""
    try:
        prompt = (
            "Review the following conversation and provide a brief summary of the client's request.\n"
            "Extract the details of the policy update required by the client.\n"
            "Also, list any bundle IDs (the values inside the placeholders like [PHONE_abc] -> 'abc', [PERSON_def] -> 'def') provided by the user.\n"
            "Return JSON matching exactly this structure: {\"summary\": \"...\", \"policy_updates\": \"...\", \"bundles\": [\"id1\", \"id2\"]}"
        )
        # Assuming last few turns contain the final context
        convo_text = "\n".join(f"{m['role']}: {m['content']}" for m in messages if m['role'] in ('user', 'assistant'))
        
        # Instantiate a clean httpx async client without proxies to avoid the init exception
        # and manage its lifecycle directly to avoid the 'AsyncHttpxClientWrapper' teardown error.
        async with httpx.AsyncClient() as http_client:
            client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY, http_client=http_client)
            response = await client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": convo_text}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
        
        content = response.choices[0].message.content
        data = json.loads(content)
        
        # Build a flexible PolicyUpdate object focusing on summary and bundles
        update = PolicyUpdate(
            policy_number="N/A", # We are no longer extracting a strict policy number
            old_name="N/A",
            new_name="N/A",
            date_of_changes="N/A",
            details=str(data.get("summary", "")).strip(),
            phone_number="N/A",
            raw=content,
            log_filename=log_filename
        )
        
        # Log the newly extracted bundles
        logger.bind(dialogue=True).info(f"Extracted Bundles: {data.get('bundles', [])}")
        
        return update
    except Exception as e:
        logger.error(f"Failed to extract JSON offline: {e}")
        return None


def _ensure_prereqs() -> None:
    missing = []
    if not DEEPGRAM_API_KEY:
        missing.append("DEEPGRAM_API_KEY")
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not PRIVATE_LAYER_API_KEY:
        missing.append("PRIVATE_LAYER_API_KEY")
    if missing:
        logger.error(f"Missing required env vars: {', '.join(missing)}")
        sys.exit(1)


async def main() -> None:
    _ensure_prereqs()

    # Session logs
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    session_ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    
    logger.remove()
    
    # 1. Add dialogue log exclusively for cleanly formatted chat events
    logger.add(
        LOGS_DIR / f"session_{session_ts}.log",
        format="[{time:YYYY-MM-DDTHH:mm:ss.SSSSSS}Z] {message}",
        filter=lambda record: record["extra"].get("dialogue", False),
        level="INFO"
    )
    
    # 2. Add pipecat debug log, filtering out dialogue events & noisy pipecat transports
    def pipecat_filter(record):
        if record["extra"].get("dialogue", False):
            return False
        if record["name"].startswith("pipecat.transports.") and record["level"].name == "DEBUG":
            return False
        return True

    logger.add(
        LOGS_DIR / f"pipecat_session_{session_ts}.log",
        level="DEBUG",
        filter=pipecat_filter
    )
    
    # 3. Add console logger with INFO level
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO",
        filter=pipecat_filter
    )
    
    # 4. Add specialized console output for dialogue events specifically for readability
    logger.add(
        sys.stderr,
        format="<magenta>[{time:YYYY-MM-DD HH:mm:ss}]</magenta> <white>{message}</white>",
        level="INFO",
        filter=lambda record: record["extra"].get("dialogue", False)
    )

    logger.info("Initializing Pipecat Insurance Agent components...")

    transport = LocalAudioTransport(
        TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=2.0)),
            vad_audio_passthrough=True,
        )
    )

    llm = OpenAILLMService(
        api_key=OPENAI_API_KEY,
        model=OPENAI_MODEL,
        params=OpenAILLMService.InputParams(temperature=0.4)
    )

    stt = DeepgramSTTService(
        api_key=DEEPGRAM_API_KEY,
        model=DEEPGRAM_LISTEN_MODEL,
    )

    tts = DeepgramTTSService(
        api_key=DEEPGRAM_API_KEY,
        voice=DEEPGRAM_SPEAK_MODEL,
    )

    system_prompt = build_system_prompt()
    greeting = get_greeting()
    
    dl = logger.bind(dialogue=True)
    dl.info(f"Session started (stt_model={DEEPGRAM_LISTEN_MODEL} speak_model={DEEPGRAM_SPEAK_MODEL} openai_model={OPENAI_MODEL})")
    dl.info(f"System prompt:\n{system_prompt}")
    dl.info(f"Greeting: {greeting}")

    context = OpenAILLMContext(
        [{"role": "system", "content": system_prompt}]
    )

    private_layer_client = PrivateLayerClient(
        base_url=PRIVATE_LAYER_URL,
        api_key=PRIVATE_LAYER_API_KEY,
        tenant_id=PRIVATE_LAYER_TENANT,
        detect_path=PRIVATE_LAYER_DETECT_PATH,
    )

    # State container for encrypt/decrypt bounds
    session = PrivateLayerSession()
    
    encrypt_filter = PrivateLayerEncryptFilter(private_layer_client, session)
    decrypt_filter = PrivateLayerDecryptFilter(private_layer_client, session)

    log_filename = f"session_{session_ts}.log"

    # Aggregates the streamed llm chunks into full sentences for correct decryption
    sentence_aggregator = SentenceAggregator()

    context_aggregator = llm.create_context_aggregator(context)

    # class GreetingProcessor(FrameProcessor):
    #     async def process_frame(self, frame: Frame, direction: FrameDirection):
    #         await super().process_frame(frame, direction)
    #         if isinstance(frame, StartFrame):
    #             # Forward the StartFrame first to initialize all downstream components (TTS, Transport)
    #             await self.push_frame(frame, direction)
    #             
    #             logger.info("Pipeline started. Agent preparing to speak greeting...")
    #             context.add_message({"role": "assistant", "content": greeting})
    #             # Yield the greeting as a TextFrame to TTS downstream
    #             await self.push_frame(TextFrame(text=greeting), FrameDirection.DOWNSTREAM)
    #             return
    #         await self.push_frame(frame, direction)

    # greeting_processor = GreetingProcessor()

    # Note: pipecat pipelines run top-to-bottom for audio/STT -> LLM,
    # and bottom-to-top (or rather LLM to TTS flows) depending on frame direction.
    pipeline = Pipeline(
        [
            transport.input(),          # 1. Microphone Input (yields audio frames, STT consumes)
            stt,                        # 2. STT (yields TranscriptionFrames)
            encrypt_filter,             # 3. Encrypts TranscriptionFrames before going to LLM
            context_aggregator.user(),  # 4. Aggregates User Text into Context messages
            llm,                        # 5. LLM generates TextFrames and sends downstream
            sentence_aggregator,        # 6. Wait for full sentence from LLM
            decrypt_filter,             # 7. Decrypts TextFrames
            tts,                        # 8. TTS synthesizes audio from decrypted TextFrames
            transport.output(),         # 9. Play Audio
            context_aggregator.assistant(), # 10. Store LLM generated final message back in Context
        ]
    )

    task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))

    runner = PipelineRunner()

    logger.info("Agent pipeline constructed. Launching runner...")
    
    # Injecting a dummy "hello" from the user so the LLM responds with a greeting natively.
    # We must actually push an LLMMessagesFrame into the task queue so the LLM triggers.
    from pipecat.frames.frames import LLMMessagesFrame
    context.add_message({"role": "user", "content": "Hello. Start the conversation by introducing yourself and asking how you can help."})
    task.queue_frames([LLMMessagesFrame(messages=context.get_messages())])

    try:
        await runner.run(task)
    except KeyboardInterrupt:
        logger.info("Shutdown requested via KeyboardInterrupt.")
    except Exception as exc:
        logger.exception(f"Pipeline error: {exc}")
    finally:
        logger.info("Pipecat session finished. Evaluating final policy update...")
        messages = context.get_messages()
        # Ensure we don't try if there was virtually no chatting
        if len(messages) > 2:
            update = await _extract_policy_update_from_context(messages, log_filename)
            if update and update.policy_number:
                logger.bind(dialogue=True).info(f"POLICY_UPDATE JSON Captured: {update.policy_number}")
                try:
                    path = save_policy_update(update)
                    logger.info(f"[Storage] Policy update saved to {path}")
                except Exception as e:
                    logger.error(f"Failed to save policy update: {e}")
            else:
                logger.info("No policy update detected or extracted from conversation.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    print("Done.")
