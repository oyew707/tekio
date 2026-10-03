"""
-------------------------------------------------------
bridges high-level computer-use actions to Playwright browser
tools, enabling mouse, keyboard, navigation, and utility
operations for LLM agents.
-------------------------------------------------------
Author:  Einstein O
Email:   eo2233@nyu.edu
-------------------------------------------------------
"""

# Imports
import os
import string
from typing import List, Literal, Tuple
from langchain_core.tools import ToolException, StructuredTool
from utils.logger import get_logger
from urllib.parse import quote_plus
from dotenv import load_dotenv
from utils.agent_utils import get_browser_state, BrowserState

# Constants
load_dotenv()
logger = get_logger(__name__, "debug")
ALPHA_NUM = string.digits + string.ascii_lowercase
CUA_KEY_TO_PLAYWRIGHT_KEY = {
    "/": "Divide",
    "\\": "Backslash",
    "alt": "Alt",
    "arrowdown": "ArrowDown",
    "arrowleft": "ArrowLeft",
    "arrowright": "ArrowRight",
    "arrowup": "ArrowUp",
    "down": "ArrowDown",
    "left": "ArrowLeft",
    "right": "ArrowRight",
    "up": "ArrowUp",
    "backspace": "Backspace",
    "capslock": "CapsLock",
    "cmd": "Meta",
    "ctrl": "Control",
    "delete": "Delete",
    "end": "End",
    "enter": "Enter",
    "esc": "Escape",
    "escape": "Escape",
    "home": "Home",
    "insert": "Insert",
    "option": "Alt",
    "pagedown": "PageDown",
    "pageup": "PageUp",
    "shift": "Shift",
    "space": " ",
    "super": "Super",
    "tab": "Tab",
    "win": "Meta",
}
# https://developer.mozilla.org/en-US/docs/Web/API/UI_Events/Keyboard_event_key_values
VALID_KEYBOARD_KEYS = (
    list(CUA_KEY_TO_PLAYWRIGHT_KEY.values())
    + [
        "AltGraph",
        "Fn",
        "FnLock",
        "Hyper",
        "NumLock",
        "ScrollLock",
        "Symbol",
        "SymbolLock",
        "Clear",
        "Copy",
        "CrSel",
        "Cut",
        "EraseEof",
        "ExSel",
        "Paste",
        "Redo",
        "Undo",
        "Pause",
        "Play",
        "Select",
        "ZoomIn",
        "ZoomOut",
        "PrintScreen",
        "F1",
        "F2",
        "F3",
        "F4",
        "F5",
        "F6",
        "F7",
        "F8",
        "F9",
        "F10",
        "F11",
        "F12",
        "BrowserBack",
        "BrowserFavorites",
        "BrowserForward",
        "BrowserRefresh",
        "BrowserStop",
        "Decimal",
        "Multiply",
        "Add",
        "Divide",
        "Subtract",
        "Separator",
    ]
    + list(ALPHA_NUM)
)


class BrowserTools:
    """
    -------------------------------------------------------
    Mapping Playwright browser use tools to Fara Computer Use
    -------------------------------------------------------
    """

    DISPLAY_SIZE = 1000
    VIEWPORT_SIZE = os.getenv("VIEWPORT_SIZE")

    def __init__(self, playwright_tools: List):
        """
        -------------------------------------------------------
        Initializes browser tools
        -------------------------------------------------------
        Parameters:
            playwright_tools
        -------------------------------------------------------
        """
        self.tools_map = {tool.name: tool for tool in playwright_tools}
        self.viewport_height = int(self.VIEWPORT_SIZE.split("x")[1])
        self.viewport_width = int(self.VIEWPORT_SIZE.split("x")[0])
        self.logger = get_logger(self.__class__.__name__, "debug")

    def _normalized_scale_to_viewport(self, x: int, y: int) -> tuple[int, int]:
        """
        -------------------------------------------------------
        Scales Normalized Coordinates from Model to viewport coordinates
        -------------------------------------------------------
        Parameters:
            x - x coordinate in 1000 pixel (int)
            y - y coordinate in 1000 pixel (int)
        Returns:
            new_x - actual coordinate (int)
            new_y - actual coordinate in browser (int)
        -------------------------------------------------------
        """
        pixel_x = round((x / self.DISPLAY_SIZE) * self.viewport_width)
        pixel_y = round((y / self.DISPLAY_SIZE) * self.viewport_height)
        return pixel_x, pixel_y

    async def left_click(self, coordinates: tuple[int, int]):
        """
        -------------------------------------------------------
        Click the left mouse button.
        -------------------------------------------------------
        Parameters:
            coordinates - (x, y): The x (pixels from the left edge) and
                y (pixels from the top edge) coordinates to move the mouse to click.
        Returns:
            tool response
        -------------------------------------------------------
        """
        tgt_x, tgt_y = coordinates
        self.logger.info(f"Left click requested at ({tgt_x}, {tgt_y})")
        if tgt_x < 0 or tgt_x > self.DISPLAY_SIZE:
            raise ToolException(
                f"x coordinate should be between 0 and {self.DISPLAY_SIZE}"
            )
        if tgt_y < 0 or tgt_y > self.DISPLAY_SIZE:
            raise ToolException(
                f"y coordinate should be between 0 and {self.DISPLAY_SIZE}"
            )
        # Implement the click
        try:
            n_tgt_x, n_tgt_y = self._normalized_scale_to_viewport(tgt_x, tgt_y)
            resp = await self.tools_map["browser_mouse_click_xy"].ainvoke(
                {"x": n_tgt_x, "y": n_tgt_y, "button": "left"}
            )
            self.logger.debug(f"Browser click response: {resp}")
            if len(resp) > 0 and "### Error" in resp[0].get("text", ""):
                raise Exception("Some error arised")
        except Exception as e:
            self.logger.error(f"Error during browser click: {e}")
            raise ToolException(f"Failed to click at {coordinates}")
        return f"I clicked at coordinates ({tgt_x}, {tgt_y})."

    async def right_click(self, coordinates: tuple[int, int]):
        """
        -------------------------------------------------------
        Click the right mouse button.
        -------------------------------------------------------
        Parameters:
            coordinates - (x, y): The x (pixels from the left edge) and
                y (pixels from the top edge) coordinates to move the mouse to right click.
        Returns:
            tool response
        -------------------------------------------------------
        """
        tgt_x, tgt_y = coordinates
        self.logger.info(f"Right click requested at ({tgt_x}, {tgt_y})")
        if tgt_x < 0 or tgt_x > self.DISPLAY_SIZE:
            raise ToolException(
                f"x coordinate should be between 0 and {self.DISPLAY_SIZE}"
            )
        if tgt_y < 0 or tgt_y > self.DISPLAY_SIZE:
            raise ToolException(
                f"y coordinate should be between 0 and {self.DISPLAY_SIZE}"
            )
        # Implement
        try:
            n_tgt_x, n_tgt_y = self._normalized_scale_to_viewport(tgt_x, tgt_y)
            response = await self.tools_map["browser_mouse_click_xy"].ainvoke(
                {"x": n_tgt_x, "y": n_tgt_y, "button": "right"}
            )
            self.logger.debug(f"Browser click response: {response}")
            if len(response) > 0 and "### Error" in response[0].get("text", ""):
                raise Exception("Some error arised")
        except Exception as e:
            self.logger.error(f"Error during browser click: {e}")
            raise ToolException(f"Failed to click at {coordinates}")

        self.logger.info(f"Right click completed at ({tgt_x}, {tgt_y}).")
        return f"I right-clicked at coordinates ({tgt_x}, {tgt_y})."

    async def key(self, keys: list[str] = []):
        """
        -------------------------------------------------------
        Performs key down presses on the arguments passed in order, then
        performs key releases in reverse order.
        -------------------------------------------------------
        Parameters:
            keys - the keys to press down (List[str])
        Returns:
            tool response
        -------------------------------------------------------
        """

        def _normalize_keys(keys: list[str]) -> frozenset:
            mapped = [CUA_KEY_TO_PLAYWRIGHT_KEY.get(k.lower(), k) for k in keys]
            valid_keys = [k if len(k) > 1 else k.lower() for k in mapped]
            assert all(
                map(lambda k: k in VALID_KEYBOARD_KEYS, valid_keys)
            ), f"Found some invalid Keys in input. These are the valid keys {VALID_KEYBOARD_KEYS}"
            return valid_keys

        normalized_keys = _normalize_keys(keys)
        key = "+".join(normalized_keys)
        try:
            resp = await self.tools_map["browser_press_key"].ainvoke({"key": key})
            self.logger.debug(f"Response after pressing {key}: {str(resp)}")
        except AssertionError as e:
            logger.error(f"Received an assertion error {str(e)}")
            return str(e)
        except Exception as e:
            logger.error(f"Could not press the following keys {keys[len(key):]}")
        return f"I pressed the following keys: {key}"

    async def mouse_move(self, coordinates: tuple[int, int]):
        """
        -------------------------------------------------------
        Move the cursor to a specified (x, y) pixel coordinate on the screen.
        -------------------------------------------------------
        Parameters:
            coordinates - (x, y): The x (pixels from the left edge) and
                y (pixels from the top edge) coordinates to move the mouse to.
        Returns:
            tool response
        -------------------------------------------------------
        """

        tgt_x, tgt_y = coordinates
        self.logger.info(f"Mouse move requested to ({tgt_x}, {tgt_y})")
        if tgt_x < 0 or tgt_x > self.DISPLAY_SIZE:
            raise ToolException(
                f"x coordinate should be between 0 and {self.DISPLAY_SIZE}"
            )
        if tgt_y < 0 or tgt_y > self.DISPLAY_SIZE:
            raise ToolException(
                f"y coordinate should be between 0 and {self.DISPLAY_SIZE}"
            )
        # Implement
        try:
            n_tgt_x, n_tgt_y = self._normalized_scale_to_viewport(tgt_x, tgt_y)
            response = await self.tools_map["browser_mouse_move_xy"].ainvoke(
                {"x": n_tgt_x, "y": n_tgt_y}
            )
            self.logger.debug(f"Browser mouse move response: {response}")
            if len(response) > 0 and "### Error" in response[0].get("text", ""):
                raise Exception("Some error arised")
        except Exception as e:
            raise ToolException(f"Failed to move mouse due to {coordinates}")

        self.logger.info(f"Mouse moved to ({tgt_x}, {tgt_y}).")
        return f"I moved the cursor to ({tgt_x}, {tgt_y})."

    async def type(self, text: str = "") -> str:
        """
        -------------------------------------------------------
        Type a string of text on the keyboard.
        -------------------------------------------------------
        Parameters:
            text - text to type (str)
        Returns:
            tool response (str)
        -------------------------------------------------------
        """
        self.logger.info(f"Type requested for: {text}")
        # Find active element
        try:
            browser_state: BrowserState = await get_browser_state(
                self.tools_map["browser_snapshot"]
            )
            self.logger.debug(f"Current Browser State: {str(browser_state)}")
            target_element = browser_state.active_element.refid
            self.logger.debug(f"Typing {text} into {str(browser_state.active_element)}")
            response = await self.tools_map["browser_type"].ainvoke(
                {
                    "text": text,
                    "submit": True,
                    "target": target_element,
                }
            )
            self.logger.debug(f"I typed {text} on keyboard : {str(response)}")
            if len(response) > 0 and "### Error" in response[0].get("text", ""):
                raise Exception("Some error arised")
        except Exception as e:
            self.logger.error(f"Failed to type into active field", str(e))
            raise ToolException(f"Failed to type {text} into active field")
        return f"I typed '{text}'."

    async def scroll(self, pixels: int = 0) -> str:
        """
        -------------------------------------------------------
        Performs a Vertical scroll of the mouse scroll wheel.
        -------------------------------------------------------
        Parameters:
            pixels - The amount of scrolling to perform. Positive values
                scroll up, negative values scroll down. (int)
        Returns:
            tool response
        -------------------------------------------------------
        """
        self.logger.info(f"Scroll requested by {pixels} pixels")
        delta = int(pixels * self.viewport_height / self.DISPLAY_SIZE)
        try:
            response = await self.tools_map["browser_mouse_wheel"].ainvoke(
                {"deltaX": 0, "deltaY": -1 * delta}
            )
            self.logger.debug(f"Browser wheel response: {response}")
            if len(response) > 0 and "### Error" in response[0].get("text", ""):
                raise Exception("Some error arised")
        except Exception as e:
            self.logger.error(f"Error during scroll: {e}")
            raise ToolException(f"Failed to scroll")
        direction = "up" if pixels > 0 else "down"
        self.logger.info(f"Scrolled {direction}.")
        return f"I scrolled {direction}."

    async def hscroll(self, pixels: int = 0) -> str:
        """
        -------------------------------------------------------
        Performs a horizontal scroll (mapped to regular scroll).
        -------------------------------------------------------
        Parameters:
            pixels - The amount of scrolling to perform. Positive values scroll left,
                negative values scroll right. (int)
        Returns:
            tool response
        -------------------------------------------------------
        """
        self.logger.info(f"Horizontal scroll requested by {pixels} pixels")
        delta = int(pixels * self.viewport_width / self.DISPLAY_SIZE)
        try:
            response = await self.tools_map["browser_mouse_wheel"].ainvoke(
                {"deltaX": -1 * delta, "deltaY": 0}
            )
            self.logger.debug(f"Browser wheel response: {response}")
            if len(response) > 0 and "### Error" in response[0].get("text", ""):
                raise Exception("Some error arised")
        except Exception as e:
            self.logger.error(f"Error during horizontal scroll: {e}")
            raise ToolException(f"Failed to scroll horizontally")

        self.logger.info(f"Horizontally scrolled by {pixels} pixels.")
        return f"I scrolled horizontally by {pixels} pixels."

    async def double_click(self, coordinates: tuple[int, int]) -> str:
        """
        -------------------------------------------------------
        Double-click the left mouse button.
        -------------------------------------------------------
        Parameters:
            coordinates - (x, y): The x (pixels from the left edge) and
                y (pixels from the top edge) coordinates to move the mouse to click.
        Returns:
            tool response
        -------------------------------------------------------
        """
        tgt_x, tgt_y = coordinates
        self.logger.info(f"Double click requested at ({tgt_x}, {tgt_y})")
        if tgt_x < 0 or tgt_x > self.DISPLAY_SIZE:
            raise ToolException(
                f"x coordinate should be between 0 and {self.DISPLAY_SIZE}"
            )
        if tgt_y < 0 or tgt_y > self.DISPLAY_SIZE:
            raise ToolException(
                f"y coordinate should be between 0 and {self.DISPLAY_SIZE}"
            )
        # Implement the click
        try:
            response = await self.tools_map["browser_mouse_click_xy"].ainvoke(
                {"x": tgt_x, "y": tgt_y, "button": "left", "clickCount": 2}
            )
            self.logger.debug(f"Browser double click response: {response}")
            if len(response) > 0 and "### Error" in response[0].get("text", ""):
                raise Exception("Some error arised")
        except Exception as e:
            self.logger.error(f"Error during double click: {e}")
            raise ToolException(f"Failed to double-click at {coordinates}")
        self.logger.info(f"Double clicked at ({tgt_x}, {tgt_y}).")
        return f"I double-clicked at coordinates ({tgt_x}, {tgt_y})."

    async def triple_click(self, coordinates: tuple[int, int]) -> str:
        """
        -------------------------------------------------------
        Triple-click the left mouse button (e.g. to select a line of text).
        -------------------------------------------------------
        Parameters:
            coordinates - (x, y): The x (pixels from the left edge) and
                y (pixels from the top edge) coordinates to move the mouse to right click.
        Returns:
            tool response
        -------------------------------------------------------
        """
        tgt_x, tgt_y = coordinates
        self.logger.info(f"Triple click requested at ({tgt_x}, {tgt_y})")

        if tgt_x < 0 or tgt_x > self.DISPLAY_SIZE:
            raise ToolException(
                f"x coordinate should be between 0 and {self.DISPLAY_SIZE}"
            )
        if tgt_y < 0 or tgt_y > self.DISPLAY_SIZE:
            raise ToolException(
                f"y coordinate should be between 0 and {self.DISPLAY_SIZE}"
            )
        # Implement
        try:
            response = await self.tools_map["browser_mouse_click_xy"].ainvoke(
                {"x": tgt_x, "y": tgt_y, "button": "left", "clickCount": 3}
            )
            self.logger.debug(f"Browser triple click response: {response}")
            if len(response) > 0 and "### Error" in response[0].get("text", ""):
                raise Exception("Some error arised")
        except Exception as e:
            self.logger.error(f"Error during triple click: {e}")
            raise ToolException(f"Failed to triple-click at {coordinates}")

        self.logger.info(f"Triple clicked at ({tgt_x}, {tgt_y}).")
        return f"I triple-clicked at coordinates ({tgt_x}, {tgt_y})."

    async def left_click_drag(
        self, start_coordinates: tuple[int, int], end_coordinates: tuple[int, int]
    ) -> str:
        """
        -------------------------------------------------------
        Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.
        -------------------------------------------------------
        Parameters:
            start_coordinates - (x, y): The x (pixels from the left edge) and
                y (pixels from the top edge) coordinates to move the mouse to click.
            send_coordinates - (x, y): The x (pixels from the left edge) and
                    y (pixels from the top edge) coordinates to move the mouse to click.
        Returns:
            tool response
        -------------------------------------------------------
        """
        tgt_x_stt, tgt_y_stt = start_coordinates
        tgt_x_end, tgt_y_end = end_coordinates
        self.logger.info(
            f"Drag requested from ({tgt_x_stt}, {tgt_y_stt}) to ({tgt_x_end}, {tgt_y_end})"
        )
        try:
            response = await self.tools_map["browser_mouse_drag_xy"].ainvoke(
                {
                    "startX": tgt_x_stt,
                    "startY": tgt_y_stt,
                    "endX": tgt_x_end,
                    "endY": tgt_y_end,
                }
            )
            self.logger.debug(f"Browser drag response: {response}")
            if len(response) > 0 and "### Error" in response[0].get("text", ""):
                raise Exception("Some error arised")
        except Exception as e:
            self.logger.error(f"Error during drag: {e}")
            raise ToolException(
                f"Failed to Left click drag from {start_coordinates} to {end_coordinates}"
            )

        self.logger.info(
            f"Dragged from ({tgt_x_stt}, {tgt_y_stt}) to ({tgt_x_end}, {tgt_y_end})."
        )
        return (
            f"I dragged from ({tgt_x_stt}, {tgt_y_stt}) to ({tgt_x_end}, {tgt_y_end})."
        )

    async def visit_url(self, url: str = "") -> str:
        """
        -------------------------------------------------------
        Visit a specified URL.
        -------------------------------------------------------
        Parameters:
            url - The URL to visit. (string)
        Returns:
            tool response
        -------------------------------------------------------
        """
        self.logger.info(f"URL visit requested for: {url}")
        if url.startswith(("https://", "http://", "file://", "about:")):
            target = url
        elif " " in url:
            target = f"https://www.duckduckgo.com/search?q={quote_plus(url)}"
        else:
            target = "https://" + url
        try:
            response = await self.tools_map["browser_navigate"].ainvoke({"url": target})
            if len(response) > 0 and "### Error" in response[0].get("text", ""):
                raise Exception("Some error arised")
        except Exception as e:
            self.logger.error(f"Error during navigation: {e}")
            raise ToolException(f"I tried to navigate to {url} but it failed")
        self.logger.info(f"Navigated to {target}.")
        return f"I navigated to {url}."

    async def history_back(self) -> str:
        """
        -------------------------------------------------------
        Go back to the previous page in the browser history.
        -------------------------------------------------------
        Returns:
            tool response
        -------------------------------------------------------
        """
        self.logger.info(f"Navigate to previous page requested")
        try:
            response = await self.tools_map["browser_navigate_back"].ainvoke({})
            if len(response) > 0 and "### Error" in response[0].get("text", ""):
                raise Exception("Some error arised")
        except Exception as e:
            self.logger.error(f"Error during history back: {e}")
            raise ToolException(f"I tried to navigate back but it failed")

        return "I clicked the browser back button."

    async def web_search(self, query: str = "") -> str:
        """
        -------------------------------------------------------
        Perform a web search with a specified query.
        -------------------------------------------------------
        Parameters:
            query - The query to search for.  (str)
        Returns:
            tool response
        -------------------------------------------------------
        """
        self.logger.info(f"Web search requested for: {query}")
        try:
            response = await self.tools_map["browser_navigate"].ainvoke(
                {"url": f"https://www.duckduckgo.com/search?q={quote_plus(query)}"}
            )
            if len(response) > 0 and "### Error" in response[0].get("text", ""):
                raise Exception("Some error arised")
        except Exception as e:
            self.logger.error(f"Error during web search: {e}")
            raise ToolException(f"I tried to search for {query} but it failed: {e}")

        self.logger.info(f"Web search completed for: {query}.")
        return f"I searched for '{query}'."

    async def ask_user_question(self, question: str = "") -> str:
        """
        -------------------------------------------------------
        Ask the user a clarifying question and wait for a response.
        -------------------------------------------------------
        Parameters:
            question - The question to ask.
        Returns:
            tool response
        -------------------------------------------------------
        """
        self.logger.info(f"User question requested: {question}")
        return f"I asked the user: {question}"

    async def wait(self, time: int = 3) -> str:
        """
        -------------------------------------------------------
        Wait specified seconds for the change to happen.
        -------------------------------------------------------
        Parameters:
            time - The seconds to wait. (int)
        Returns:
            tool response
        -------------------------------------------------------
        """
        self.logger.info(f"Wait requested for {time}s")
        await self.tools_map["browser_wait_for"].ainvoke({"time": time})
        self.logger.debug(f"Wait completed for {time}s.")
        return f"I waited {time}s."

    async def pause_and_memorize_fact(self, fact: str = "") -> Tuple[str, dict]:
        """
        -------------------------------------------------------
        Pause and memorize a fact for future reference.
        -------------------------------------------------------
        Parameters:
            fact - The fact to remember for the future. (str)
        Returns:
            tool response
        -------------------------------------------------------
        """
        self.logger.info(f"Fact memorized: {fact}")
        return f"I memorized the following fact: {fact}", {"facts": [fact]}

    async def terminate(self, answer) -> str:
        """
        -------------------------------------------------------
        Terminate the current task and provide the final answer.
        -------------------------------------------------------
        Parameters:
            answer - The final answer for the task.
        Returns:
            tool response
        -------------------------------------------------------
        """
        self.logger.info(f"Termination requested with answer: {answer}")
        return answer

    def get_tools(self) -> list[StructuredTool]:
        """
        -------------------------------------------------------
        All the tools
        -------------------------------------------------------
        Returns:
            tools - list of all browser tools (list)
        -------------------------------------------------------
        """
        tool_methods = [
            "left_click",
            "right_click",
            "key",
            "mouse_move",
            "type",
            "scroll",
            "hscroll",
            "double_click",
            "triple_click",
            "left_click_drag",
            "visit_url",
            "history_back",
            "web_search",
            "ask_user_question",
            "wait",
            "pause_and_memorize_fact",
            "terminate",
        ]

        return [
            StructuredTool.from_function(
                coroutine=getattr(self, method),
                name=method,
                parse_docstring=True,
                response_format=(
                    "content_and_artifact"
                    if method == "pause_and_memorize_fact"
                    else "content"
                ),
            )
            for method in tool_methods
        ]
