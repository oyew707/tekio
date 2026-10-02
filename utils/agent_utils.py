"""
-------------------------------------------------------
Utility module for agents
-------------------------------------------------------
Author:  Einstein Oyewole
Email:   eo2233@nyu.edu
-------------------------------------------------------
"""

# Imports
import asyncio
import re
import base64
from io import BytesIO
from typing import Dict, List, Tuple
from PIL import Image, ImageDraw
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, BaseMessage
from .logger import get_logger
from dataclasses import dataclass

# Constants
logger = get_logger(__name__, "debug")


@dataclass
class ActiveElement:
    """
    -------------------------------------------------------
    Information about the currently active element in the browser.
    -------------------------------------------------------
    Parameters:
       role - ARIA role of the active element (str)
       name - accessible name of the active element (str)
       refid - playwright unique reference IDs to interactive components (str)
       bbox - bounding box of the element (x,y,width,height)
    -------------------------------------------------------
    """

    role: str | None = None
    name: str | None = None
    refid: str | None = None
    bbox: list[int] | None = None


@dataclass
class BrowserState:
    """
    -------------------------------------------------------
    Snapshot of the current browser state including URL, title, and active element.
    -------------------------------------------------------
    Parameters:
       url - url of the current page (str)
       title - Title of the current page (str)
       active_element - The active element on the page
    -------------------------------------------------------
    """

    url: str | None = None
    title: str | None = None
    active_element: ActiveElement | None = None


def get_screenshot_dimensions(screenshot: Image.Image) -> Tuple[int, int]:
    """
    -------------------------------------------------------
    Return validated screenshot/observation dimensions
    -------------------------------------------------------
    Parameters:
       screenshot [PIL.Image.Image - The screenshot image object]
    Returns:
       dimensions [tuple[int, int] - (width, height)]
    -------------------------------------------------------
    """
    width, height = screenshot.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid screenshot size: {width}x{height}")
    return int(width), int(height)


def build_screen_message(
    playwright_screencapture_resp: List[Dict],
    browser_state: BrowserState,
) -> Tuple[str, str, str]:
    """
    -------------------------------------------------------
    Build the text block, base64 string, and mime type
    paired with a screenshot before a model call.
    -------------------------------------------------------
    Parameters:
       playwright_screencapture_resp [list[dict] - The response list containing text and image data]
    Returns:
       tuple[str, str, str] - (base64_string, mime_type)
    -------------------------------------------------------
    """
    image_obj = None
    base64_str = None
    mime_type = None

    # Extract text and image from the response list
    img_item = next(
        item for item in playwright_screencapture_resp if item.get("type") == "image"
    )
    base64_str = img_item.get("base64", "")
    mime_type = img_item.get("mime_type", "image/png")
    if base64_str:
        image_bytes = base64.b64decode(base64_str)
        image_obj = Image.open(BytesIO(image_bytes))

    if not image_obj:
        raise ValueError("No image found in playwright_screencapture_resp")

    # width, height = get_screenshot_dimensions(image_obj)

    parts = [
        "### Browser State",
        f"Page Title: {browser_state.title}",
        f"Current URL: {browser_state.url.split('?')[0][:100]}",
        # f"Screenshot resolution: {width}x{height}",
    ]
    if browser_state.active_element:
        parts.extend(
            [
                "**Active element:**",
                f"Name: {browser_state.active_element.name}",
                f"Role: {browser_state.active_element.role}",
            ]
        )
        if browser_state.active_element.bbox:
            # Draw a box to show active element
            draw = ImageDraw.Draw(image_obj)
            bbox = browser_state.active_element.bbox
            left, top, right, bottom = (
                bbox[0],
                bbox[1],
                bbox[0] + bbox[2],
                bbox[1] + bbox[3],
            )
            box_color: tuple = (255, 0, 0)
            line_width: int = 1
            draw.rectangle(
                [left, top, right, bottom], outline=box_color, width=line_width
            )

            # Encode to base64
            img_byte_arr = BytesIO()
            image_obj.save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)
            base64_str = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")
            mime_type = "image/png"
    return "\n".join(parts), base64_str, mime_type


async def get_browser_state(
    browser_snapshot_tool: Tool,
) -> HumanMessage:
    """
    -------------------------------------------------------
    Captures the current browser state including URL, Title,
    and details about the currently active element.
    -------------------------------------------------------
    Parameters:
       browser_snapshot_tool - captures a structured accessibility
           tree (ARIA tree) of the current web pag
    Returns:
       BrowserState
    -------------------------------------------------------
    """
    logger.info("Getting Browser State")
    state = BrowserState()
    try:
        snap_resp = await browser_snapshot_tool.ainvoke({"boxes": True})
        raw_text = snap_resp[0]["text"]
        logger.debug(f"Snapshot Response: {raw_text[:100]}")

        # 1. Extract URL and Title
        url_match = re.search(r"- Page URL: (https?://\S+)", raw_text)
        title_match = re.search(r"- Page Title: (.*?)(?:\n|$)", raw_text)

        state.url = url_match.group(1) if url_match else None
        state.title = title_match.group(1) if title_match else None

        # 2. Find the line with [active] on it
        target = "[active]"
        pattern = rf"^.*{re.escape(target)}.*$"
        active_lines = re.findall(pattern, raw_text, flags=re.MULTILINE)

        assert (
            len(active_lines) <= 1
        ), f"Expected at most one active element, found {len(active_lines)}"
        active_line = active_lines[0]
        logger.debug(f"{active_line=}")
        pattern = r'^\s*-\s+(?P<role>\w+)(?:\s+"(?P<name>[^"]+)")?'
        role_name_match = re.search(pattern, active_line)
        ref_match = re.search(r"\[ref=([^\]]+)\]", active_line)
        box_match = re.search(r"\[box=([^\]]+)\]", active_line)

        if role_name_match:
            logger.info(f"Found role and name {str(role_name_match)}")
            match_data = role_name_match.groupdict()
            refid = ref_match.group(1) if ref_match else None
            bbox = None
            if box_match:
                bbox_str = box_match.group(1)
                bbox = [int(val) for val in bbox_str.split(",")]

            state.active_element = ActiveElement(
                role=match_data.get("role"),
                name=match_data.get("name"),
                refid=refid,
                bbox=bbox,
            )

    except Exception as e:
        logger.error(f"Failed to find active elements: {e}")

    logger.info(f"Returning State {str(state)}")
    return state


async def capture_and_build_screen_message(
    playwright_tools: List,
) -> HumanMessage:
    """
    -------------------------------------------------------
    Capture screenshot with active element highlight,
    build the multimodal screen message, and return as HumanMessage.
    -------------------------------------------------------
    Parameters:
       playwright_tools [dict - Available Playwright tools]
    Returns:
       HumanMessage
    -------------------------------------------------------
    """
    browser_state = await get_browser_state(
        next(tool for tool in playwright_tools if tool.name == "browser_snapshot")
    )
    logger.debug(f"1. Browser State: {browser_state=}")

    # Take Screenshot
    screenshot_tool = next(
        tool for tool in playwright_tools if tool.name == "browser_take_screenshot"
    )
    screenshot_resp = await screenshot_tool.ainvoke({})

    # Build multimodal message
    text, image_enc, mime_type = build_screen_message(screenshot_resp, browser_state)

    return HumanMessage(
        content=[
            {"type": "text", "text": text},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{image_enc}"},
            },
        ],
        additional_kwargs={"type": "observation"},
    )


def contains_target_string(message: BaseMessage, target_string: str) -> bool:
    """
    -------------------------------------------------------
    Checks if target_string exists within the message content,
    handling both list of dicts and strings.
    -------------------------------------------------------
    Parameters:
       message - message to search the text content for (BaseMessage)
       target_string - string that should be in message (str)
    Returns:
       Bool - True if in message False otherwise
    -------------------------------------------------------
    """
    content = getattr(message, "content", "")
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                if target_string in item.get("text", ""):
                    logger.debug(f"Found {target_string} in {item.get('text')}")
                    return True
    elif isinstance(content, str):
        if target_string in content:
            logger.debug(f"Found {target_string} in {item.get('text')}")
            return True
    return False
