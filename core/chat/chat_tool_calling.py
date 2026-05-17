import json
import logging
import time
import re
import uuid
from typing import Dict, Any, Optional, List

from .llm_providers import LLMResponse
from .llm_providers.base import BaseProvider

logger = logging.getLogger(__name__)


def filter_to_thinking_only(content: str) -> str:
    """
    Extract only <think> tags, removing all other content.
    Used for assistant messages with tool_calls to prevent premature responses.
    """
    if not content:
        return ""
    
    think_pattern = r'<(?:seed:)?think[^>]*>.*?</(?:seed:)?think[^>]*>'
    think_matches = re.findall(think_pattern, content, re.DOTALL | re.IGNORECASE)
    
    filtered = "\n\n".join(think_matches) if think_matches else ""
    
    # If no think tags but there's content, wrap FULL content (don't truncate!)
    if not filtered and content:
        filtered = f"<think>{content.strip()}</think>"
        logger.info(f"No think tags found, wrapped full content in think block")
    
    if filtered != content:
        logger.info(f"Stripped prose: {len(content)} -> {len(filtered)} chars")
    
    return filtered

def strip_ui_markers(content: str) -> str:
    """
    Strip UI-only markers from content before sending to LLM.
    
    Removes patterns like:
    - <<IMG::image_id>>
    - <<FILE::file_id>>
    - Any other <<MARKER::data>> patterns
    
    These markers are kept in history for UI parsing but removed from LLM context.
    """
    if not content:
        return content
    
    # Pattern to match <<TYPE::data>> markers
    marker_pattern = r'<<[A-Z]+::[^>]+>>\s*'
    
    # Remove all markers
    clean = re.sub(marker_pattern, '', content)
    
    # Log if we stripped anything
    if clean != content:
        markers_found = re.findall(marker_pattern, content)
        logger.info(f"[CLEANUP] Stripped {len(markers_found)} UI markers from tool result for LLM context")
        for marker in markers_found:
            logger.debug(f"   - {marker.strip()}")
    
    return clean.strip()


def wrap_tool_result(tool_call_id: str, function_name: str, result: str) -> Dict[str, Any]:
    """
    Wrap tool results in standard OpenAI tool format.
    
    Args:
        tool_call_id: The tool call ID
        function_name: Name of the function that was called
        result: The result string from the function (ALREADY STRIPPED of UI markers)
    
    Returns:
        Properly formatted message dict for the LLM
    """
    clean_result = strip_ui_markers(result) if '<<' in result else result
    
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": function_name,
        "content": clean_result
    }


def wire_to_canonical(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert a wire-format message list to canonical persistence shape.

    Some providers (Claude, Anthropic-compat) emit tool results as
    `{role:"user", content:[{type:"tool_result", tool_use_id, content}]}`
    so the next API call carries them in the shape the model expects. The
    canonical persistence form, used by `format_messages_for_display` and
    every OpenAI-style reader, is `{role:"tool", tool_call_id, name, content}`.

    The foreground chat path avoids this mismatch by writing canonical
    directly through `history.add_tool_result()` during execution. The
    heartbeat / ExecutionContext path doesn't have a history object, so its
    `messages` slice arrives at persistence time still in wire form. Without
    converting, every Claude-backed scheduled task ends up with
    invisible-bubble corruption in its named chat.

    This walks the input and:
      - Splits any `role=user` list-content `tool_result` blocks into
        separate canonical `role=tool` messages (one per block) via the
        existing `wrap_tool_result` shape, looking up `name` from the most
        recent preceding assistant `tool_calls` within the slice.
      - Preserves any non-`tool_result` blocks (text/image/file) on a
        stripped user message at the original position. In practice
        `format_tool_result` only emits a single `tool_result` block per
        message — the mixed branch is defensive.
      - Passes everything else through unchanged. Image-injection user
        messages from `_inject_tool_images` (text + image blocks) are
        already a shape the display formatter handles, so they pass.

    Pure function — does not mutate the input list.
    """
    result: List[Dict[str, Any]] = []
    name_by_tool_use_id: Dict[str, str] = {}

    for msg in messages:
        role = msg.get("role")

        # Build the rolling name lookup from assistant tool_calls so we can
        # populate canonical `name` on the converted tool messages below.
        if role == "assistant":
            tcs = msg.get("tool_calls")
            if isinstance(tcs, list):
                for tc in tcs:
                    if not isinstance(tc, dict):
                        continue
                    tc_id = tc.get("id")
                    fn_name = (tc.get("function") or {}).get("name", "tool")
                    if tc_id:
                        name_by_tool_use_id[tc_id] = fn_name
            result.append(msg)
            continue

        if role == "user" and isinstance(msg.get("content"), list):
            content = msg["content"]
            tool_result_blocks = [
                b for b in content
                if isinstance(b, dict) and b.get("type") == "tool_result"
            ]
            if tool_result_blocks:
                for block in tool_result_blocks:
                    tc_id = block.get("tool_use_id") or block.get("tool_call_id", "")
                    name = name_by_tool_use_id.get(tc_id, "tool")
                    block_content = block.get("content", "")
                    # tool_result content can itself be a list of typed blocks
                    # (Claude allows nested text/image inside a tool_result);
                    # flatten the text parts to a single string for canonical
                    # storage. Image blocks inside tool results aren't a path
                    # we exercise today — fall through to str() if needed.
                    if isinstance(block_content, list):
                        text_parts = [
                            b.get("text", "") for b in block_content
                            if isinstance(b, dict) and b.get("type") == "text"
                        ]
                        block_content = "\n".join(t for t in text_parts if t)
                    result.append(wrap_tool_result(tc_id, name, str(block_content)))

                # Preserve any non-tool_result blocks at the original position
                # so a defensive mixed message doesn't lose its text/image data.
                non_tool_blocks = [
                    b for b in content
                    if not (isinstance(b, dict) and b.get("type") == "tool_result")
                ]
                if non_tool_blocks:
                    result.append({**msg, "content": non_tool_blocks})
                continue

        result.append(msg)

    return result


def _extract_tool_images(result, history=None, provider=None):
    """Extract images from a tool result if it returned structured data.

    Tools can return {"text": "...", "images": [{"data": base64, "media_type": "image/..."}]}
    to pass images back to the LLM on the next turn.

    Images are saved to the chat history DB and <<IMG::tool:id>> markers are embedded
    in the text so they persist in history and render via existing image infrastructure.

    If the provider doesn't support vision, images are still saved (UI + history
    still render them) but the LLM-bound images list is empty, and the tool result
    text gets a "vision unavailable" suffix. This keeps the conversation a clean
    tool-call → tool-result → assistant-response pair instead of injecting a fake
    user message after the tool result. 2026-05-15.

    Returns (text_str, images_list). images_list contains the raw image dicts
    for injection into the next LLM turn.
    """
    if isinstance(result, dict) and "images" in result and isinstance(result["images"], list):
        text = str(result.get("text", ""))
        images = [
            img for img in result["images"]
            if isinstance(img, dict) and img.get("data")
        ]
        supports_vision = bool(provider and getattr(provider, 'supports_images', False))
        # Save images to DB and embed markers in text
        # Images with display_only=True are saved for user gallery but not sent to LLM
        llm_images = []
        for img in images:
            img_id = _save_tool_image(img, history)
            if img_id:
                text = f"<<IMG::tool:{img_id}>>\n{text}"
            if not img.get("display_only") and supports_vision:
                llm_images.append(img)
        if images and not supports_vision:
            # Non-vision model: image goes to disk + UI history, but the LLM
            # can't see it. Generate a CLIP-based atmospheric description so
            # the model still gets the "vibe" of what was captured.
            vibe_parts = []
            for img in images:
                try:
                    import base64 as _b64
                    from core import vibes as _vibes
                    raw = _b64.b64decode(img["data"])
                    vibe_parts.append(_vibes.describe(raw))
                except Exception as e:
                    logger.warning(f"[VIBES] failed to describe image: {e}")
            if vibe_parts:
                text = (text + "\n\n" + "\n\n".join(vibe_parts)).strip()
            else:
                text = (text + "\n\n[Note: current model does not support image inputs — "
                        f"image(s) were captured and saved but cannot be analyzed this turn.]").strip()
        return text, llm_images
    return str(result), []


def _save_tool_image(img, history=None):
    """Save a base64 tool image to the chat history database. Returns image ID or None."""
    import base64

    try:
        img_id = uuid.uuid4().hex[:12]
        media_type = img.get("media_type", "image/jpeg")
        ext = "png" if "png" in media_type else "jpg"
        full_id = f"{img_id}.{ext}"

        img_bytes = base64.b64decode(img["data"])

        if history and hasattr(history, 'save_tool_image'):
            history.save_tool_image(full_id, img_bytes, media_type)
            logger.info(f"[TOOL] Saved tool image to DB: {full_id}")
        else:
            # Fallback to disk if no history available (isolated tool calls)
            from pathlib import Path
            img_dir = Path(__file__).parent.parent.parent / "user" / "tool_images"
            img_dir.mkdir(parents=True, exist_ok=True)
            (img_dir / full_id).write_bytes(img_bytes)
            logger.info(f"[TOOL] Saved tool image to disk (no history): {full_id}")

        return full_id
    except Exception as e:
        logger.error(f"[TOOL] Failed to save tool image: {e}")
        return None


class ToolCallingEngine:
    def __init__(self, function_manager):
        self.function_manager = function_manager

    def call_llm_with_metrics(self, provider: BaseProvider, messages: List, gen_params: Dict, tools: List = None) -> LLMResponse:
        """Call LLM with performance metrics via provider abstraction."""
        logger.info(f"[TOOL] LLM CALL [{provider.provider_name}]: {len(messages)} messages, {sum(len(str(m)) for m in messages)} chars")
        
        start_time = time.time()
        
        response = provider.chat_completion(messages, tools=tools, generation_params=gen_params)
        
        elapsed = time.time() - start_time
        
        # Log performance + diagnostic trinity (finish_reason, reasoning tokens).
        # Without these, "short response + fast t/s" is ambiguous — could be
        # the model legitimately stopping, could be hitting max_tokens, could
        # be reasoning tokens eating budget invisibly. 2026-04-24.
        try:
            usage = response.usage or {}
            finish = getattr(response, 'finish_reason', None) or '?'
            reasoning_tok = usage.get('reasoning_tokens', 0)
            content_chars = len(str(response.content)) if response.content else 0
            if usage.get('completion_tokens'):
                tps = usage['completion_tokens'] / elapsed
                logger.info(
                    f"LLM ({provider.model}): {elapsed:.2f}s, {content_chars} chars, "
                    f"{tps:.1f} t/s, finish={finish}, reasoning_tok={reasoning_tok}"
                )
            else:
                logger.info(
                    f"LLM ({provider.model}): {elapsed:.2f}s, {content_chars} chars, "
                    f"finish={finish}"
                )
        except (AttributeError, ZeroDivisionError, TypeError):
            pass
        
        return response

    def extract_function_call_from_text(self, text: str) -> Optional[Dict]:
        """Extract function calls from text (LM Studio, Qwen3, GLM, etc compatibility)."""
        if not text:
            return None

        # Format 1: {"function_call": {"name": "...", "arguments": {...}}}
        pattern = r'(\{"function_call":\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)

        if match:
            try:
                parsed = json.loads(match.group(1))
                if "function_call" in parsed and "name" in parsed["function_call"]:
                    return parsed
            except json.JSONDecodeError as e:
                logger.debug(f"JSON parse failed for format 1: {e}")

        # Format 2: <function_call>{"name": "...", "arguments": {...}}</function_call> (Qwen3, etc)
        xml_pattern = r'<function_call>\s*(\{.*?\})\s*</function_call>'
        xml_match = re.search(xml_pattern, text, re.IGNORECASE | re.DOTALL)

        if xml_match:
            try:
                inner = json.loads(xml_match.group(1))
                if "name" in inner:
                    return {"function_call": inner}
            except json.JSONDecodeError as e:
                logger.debug(f"JSON parse failed for XML format: {e}")

        # Format 3: <tool_call>{"name": "...", "arguments": {...}}</tool_call> (GLM, etc)
        tool_call_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        tool_call_match = re.search(tool_call_pattern, text, re.IGNORECASE | re.DOTALL)

        if tool_call_match:
            try:
                inner = json.loads(tool_call_match.group(1))
                if "name" in inner:
                    return {"function_call": inner}
            except json.JSONDecodeError as e:
                logger.debug(f"JSON parse failed for tool_call format: {e}")

        # Format 4: Raw JSON with name field (fallback) - {"name": "tool_name", "arguments": {...}}
        # Only match if it looks like a standalone tool call (not embedded in prose)
        raw_pattern = r'^\s*(\{"name":\s*"[^"]+",\s*"arguments":\s*\{[^{}]*\}\s*\})\s*$'
        raw_match = re.search(raw_pattern, text, re.MULTILINE)

        if raw_match:
            try:
                inner = json.loads(raw_match.group(1))
                if "name" in inner:
                    return {"function_call": inner}
            except json.JSONDecodeError as e:
                logger.debug(f"JSON parse failed for raw format: {e}")

        return None

    def format_tool_calls_for_conversation(self, tool_calls):
        """Convert tool_calls to proper format."""
        tool_calls_formatted = []
        for tool_call in tool_calls:
            tool_calls_formatted.append({
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments
                }
            })
        return tool_calls_formatted

    def execute_tool_calls(self, tool_calls, messages, history, provider: BaseProvider = None, scopes=None, allowed_tools=None, executor_snapshot=None):
        """
        Execute tool calls and add results to messages array AND history.

        Key behaviors:
        - Tool results sent to LLM have UI markers STRIPPED (clean context)
        - Tool results saved to history contain raw content (structured JSON handles display)
        - Reset function results are NOT saved to history
        - Uses provider.format_tool_result() if provider given (for Claude compatibility)
        - Tools returning {"text": "...", "images": [...]} have images accumulated

        Returns (tools_executed, tool_images) where tool_images is a list of
        {"data": base64, "media_type": "image/..."} dicts from tool results.

        Note: Caller should slice tool_calls to MAX_PARALLEL_TOOLS before calling.
        """
        tools_executed = 0
        tool_images = []

        for tool_call in tool_calls:
            function_name = tool_call["function"]["name"]

            try:
                function_args = json.loads(tool_call["function"]["arguments"])
            except json.JSONDecodeError:
                logger.error(f"Failed to parse tool arguments: {tool_call['function']['arguments']}")
                error_result = "Error: Invalid JSON arguments."

                if provider:
                    wrapped_msg = provider.format_tool_result(tool_call["id"], function_name, error_result)
                else:
                    wrapped_msg = wrap_tool_result(tool_call["id"], function_name, error_result)
                messages.append(wrapped_msg)

                if history:
                    history.add_tool_result(tool_call["id"], function_name, error_result)
                continue

            try:
                function_result = self.function_manager.execute_function(function_name, function_args, scopes=scopes, allowed_tools=allowed_tools, executor_snapshot=executor_snapshot)
            except Exception as tool_error:
                logger.error(f"Tool execution failed for {function_name}: {tool_error}", exc_info=True)
                function_result = f"Tool '{function_name}' failed: {str(tool_error)}"

            # Extract images if tool returned structured result
            result_str, images = _extract_tool_images(function_result, history, provider)
            if images:
                tool_images.extend(images)
                logger.info(f"[TOOL] {function_name} returned {len(images)} image(s)")

            clean_result = strip_ui_markers(result_str)

            if provider:
                wrapped_msg = provider.format_tool_result(tool_call["id"], function_name, clean_result)
            else:
                wrapped_msg = wrap_tool_result(tool_call["id"], function_name, clean_result)
            messages.append(wrapped_msg)

            logger.info(f"[OK] Tool result added to messages")
            logger.debug(f"   Message role: {wrapped_msg['role']}")
            logger.debug(f"   Content preview: {str(wrapped_msg.get('content', ''))[:100]}")

            if history:
                logger.info(f"[SAVE] Saving tool result for: {function_name}")
                history.add_tool_result(tool_call["id"], function_name, result_str, inputs=function_args)
            else:
                logger.debug(f"[ISOLATED] No history manager, skipping save for: {function_name}")

            tools_executed += 1
            logger.info(f"[OK] Executed tool: {function_name}")

        return tools_executed, tool_images

    def execute_text_based_tool_call(self, function_call_data, filtered_content, messages, history, provider: BaseProvider = None, scopes=None, allowed_tools=None, executor_snapshot=None):
        """
        Execute text-based function call (LM Studio compatibility).
        
        Args:
            function_call_data: The parsed function call data
            filtered_content: Pre-filtered content (thinking only) from caller
            messages: Messages array to append to
            history: History manager to save to
            provider: Optional provider for format_tool_result (Claude compatibility)
        
        Returns tool call ID.
        """
        tool_call_id = f"call_{uuid.uuid4().hex[:8]}"
        function_name = function_call_data["function_call"]["name"]
        function_args = function_call_data["function_call"]["arguments"]

        # Some text-based-tool-call providers deliver `arguments` as an already-
        # JSON-encoded string; others as a dict. json.dumps on a string would
        # double-encode (writes `"\"{...}\""` to history), bricking the chat
        # for strict providers on the next turn. Honor whatever the LLM gave us.
        if isinstance(function_args, str):
            args_json = function_args
        else:
            args_json = json.dumps(function_args)

        tool_calls_formatted = [{
            "id": tool_call_id,
            "type": "function",
            "function": {
                "name": function_name,
                "arguments": args_json
            }
        }]
        
        messages.append({
            "role": "assistant",
            "content": filtered_content,
            "tool_calls": tool_calls_formatted
        })
        
        if history:
            history.add_assistant_with_tool_calls(filtered_content, tool_calls_formatted)

        try:
            function_result = self.function_manager.execute_function(function_name, function_args, scopes=scopes, allowed_tools=allowed_tools, executor_snapshot=executor_snapshot)
        except Exception as tool_error:
            logger.error(f"Text-based tool failed for {function_name}: {tool_error}")
            function_result = f"Tool '{function_name}' failed: {str(tool_error)}"

        result_str, tool_images = _extract_tool_images(function_result, history, provider)
        if tool_images:
            logger.info(f"[TOOL] {function_name} returned {len(tool_images)} image(s) (text-based)")
        clean_result = strip_ui_markers(result_str)

        if provider:
            wrapped_msg = provider.format_tool_result(tool_call_id, function_name, clean_result)
        else:
            wrapped_msg = wrap_tool_result(tool_call_id, function_name, clean_result)
        messages.append(wrapped_msg)

        if history:
            history.add_tool_result(tool_call_id, function_name, result_str, inputs=function_args)
        else:
            logger.debug(f"[ISOLATED] No history manager, skipping save for: {function_name}")

        logger.info(f"[OK] Executed text-based tool: {function_name}")
        return tool_call_id, tool_images