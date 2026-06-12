# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Communication helper utilities for DexControl using DexComm.

This module provides simple helper functions for DexControl's communication
needs using the DexComm library's Raw API.
"""

import json
import time

from dexcomm import call_service
from dexcomm.codecs import JsonDataCodec
from loguru import logger

from dexcontrol.utils.os_utils import resolve_key_name


def query_json_service(
    topic: str,
    timeout: float = 2.0,
    max_retries: int = 1,
    retry_delay: float = 0.5,
) -> dict | None:
    """Query for JSON information using DexComm with retry logic.

    Args:
        topic: Topic to query (will be resolved with robot namespace).
        timeout: Maximum time to wait for a response in seconds.
        max_retries: Maximum number of retry attempts.
        retry_delay: Initial delay between retries (doubles each retry).

    Returns:
        Dictionary containing the parsed JSON response if successful, None otherwise.
    """
    resolved_topic = resolve_key_name(topic)
    logger.debug(f"Querying topic: {resolved_topic}")

    current_delay = retry_delay
    for attempt in range(max_retries + 1):
        try:
            data = call_service(
                resolved_topic,
                timeout=timeout,
                request_encoder=None,
                response_decoder=JsonDataCodec.decode,
            )

            if data:
                logger.debug(f"Successfully received JSON data from {resolved_topic}")
                return data

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response from {resolved_topic}: {e}")
        except Exception as e:
            logger.warning(
                f"Query failed for {resolved_topic} (attempt {attempt + 1}/{max_retries + 1}): {e}"
            )

        if attempt < max_retries:
            logger.debug(f"Retrying in {current_delay:.1f} seconds...")
            time.sleep(current_delay)
            current_delay *= 2  # Exponential backoff

    logger.error(f"Failed to query {resolved_topic} after {max_retries + 1} attempts")
    return None
