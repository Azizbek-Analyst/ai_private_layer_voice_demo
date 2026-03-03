"""Custom Pipecat filters for communicating with AI Private Layer."""
import json
import re
from typing import Any, Dict, List

from loguru import logger
from pipecat.frames.frames import Frame, TextFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from private_layer_client import PrivateLayerClient, PrivateLayerError


class PrivateLayerSession:
    """Holds state across the conversation for PrivateLayer decryption."""

    def __init__(self) -> None:
        self.bundles: List[Dict[str, Any]] = []


class PrivateLayerEncryptFilter(FrameProcessor):
    """Encrypts final user transcriptions before sending them to the LLM."""

    def __init__(self, client: PrivateLayerClient, session: PrivateLayerSession):
        super().__init__()
        self._client = client
        self._session = session

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            text = frame.text.strip()
            if text:
                try:
                    logger.bind(dialogue=True).info(f"USER_RAW: {text}")
                    result = self._client.sanitize(text)
                    
                    # Accumulate bundles
                    self._session.bundles.extend(result.bundles)
                    
                    logger.bind(dialogue=True).info(f"USER_SANITIZED: {result.text_with_placeholders}")
                    if result.bundles:
                        logger.bind(dialogue=True).info(f"PRIVATE_LAYER_BUNDLES: {json.dumps(result.bundles)}")
                    else:
                        logger.bind(dialogue=True).info("PRIVATE_LAYER_BUNDLES: []")
                    
                    # Reconstruct a new frame with the sanitized text
                    frame = TranscriptionFrame(
                        text=result.text_with_placeholders,
                        user_id=frame.user_id,
                        timestamp=frame.timestamp,
                        language=frame.language
                    )
                except PrivateLayerError as err:
                    logger.error(f"PrivateLayer sanitize error: {err}")

        await self.push_frame(frame, direction)


class PrivateLayerDecryptFilter(FrameProcessor):
    """Decrypts synthesized LLM text before sending it to TTS."""

    def __init__(self, client: PrivateLayerClient, session: PrivateLayerSession):
        super().__init__()
        self._client = client
        self._session = session

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Assuming this is preceded by a SentenceAggregator
        # so TextFrames contain complete chunks, not fragments
        if isinstance(frame, TextFrame):
            text = frame.text.strip()
            if text and self._session.bundles:
                try:
                    # The LLM sometimes hallucinates the placeholder format as <PHONE_xxx> instead of [PHONE_xxx].
                    # We normalize any angle-bracket placeholders back to square brackets to fix decryption.
                    normalized_text = re.sub(r'<([A-Z0-9_]+_[A-Za-z0-9_-]+)>', r'[\1]', text)
                    if normalized_text != text:
                        logger.bind(dialogue=True).info(f"LLM_NORMALIZED: {normalized_text}")
                    else:
                        logger.bind(dialogue=True).info(f"LLM_OUTPUT: {text}")
                        
                    decrypted_text, _ = self._client.decrypt(normalized_text, self._session.bundles)
                    if normalized_text != decrypted_text:
                        logger.bind(dialogue=True).info(f"LLM_DECRYPTED: {decrypted_text}")
                    
                    frame = TextFrame(text=decrypted_text)
                except PrivateLayerError as err:
                    logger.error(f"PrivateLayer decrypt error: {err}")
            elif text:
                # If there are no bundles to decrypt, just log the text
                logger.bind(dialogue=True).info(f"LLM_OUTPUT: {text}")

        await self.push_frame(frame, direction)
