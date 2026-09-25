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
import asyncio
from typing import List
from langchain_core.tools import ToolException, StructuredTool
from utils.logger import get_logger
from urllib.parse import quote_plus

# Constants
logger = get_logger(__name__, "info")

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
    "super": "Meta",
    "tab": "Tab",
    "win": "Meta",
}


class BrowserTools:
    """
    -------------------------------------------------------
    Mapping Playwright browser use tools to Fara Computer Use
    -------------------------------------------------------
    """

    DISPLAY_SIZE = 1000

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
        self.viewport_height = 1444
        self.viewport_width = 1000
        self.logger = get_logger(self.__class__.__name__, "debug")

    def left_click(self, coordinates: tuple[int, int]):
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
            resp = asyncio.run(
                self.tools_map["browser_mouse_click_xy"].ainvoke(
                    {"x": tgt_x, "y": tgt_y, "button": "left"}
                )
            )
            self.logger.debug(f"Browser click response: {resp}")
        except Exception as e:
            self.logger.error(f"Error during browser click: {e}")
            raise ToolException(f"Failed to click at {e}")
        return f"I clicked at coordinates ({tgt_x}, {tgt_y})."

    def right_click(self, coordinates: tuple[int, int]):
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
            response = asyncio.run(
                self.tools_map["browser_mouse_click_xy"].ainvoke(
                    {"x": tgt_x, "y": tgt_y, "button": "right"}
                )
            )
            self.logger.debug(f"Browser click response: {response}")
        except Exception as e:
            self.logger.error(f"Error during browser click: {e}")
            raise ToolException(f"Failed to click at {e}")

        self.logger.info(f"Right click completed at ({tgt_x}, {tgt_y}).")
        return f"I right-clicked at coordinates ({tgt_x}, {tgt_y})."

    def key(self, keys: List = []):
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
        pressed = []
        try:
            for key in keys:
                assert (
                    key in CUA_KEY_TO_PLAYWRIGHT_KEY.keys()
                ), f"{key} is not valid, Use one of the following keys: { CUA_KEY_TO_PLAYWRIGHT_KEY.keys()}"
                playwright_key = CUA_KEY_TO_PLAYWRIGHT_KEY[key]
                resp = asyncio.run(
                    self.tools_map["browser_press_key"].ainvoke({"key": playwright_key})
                )
                self.logger.debug(f"Response after pressing {key}: {str(resp)}")
                pressed.append(key)
        except Exception as e:
            logger.error(f"Could not press the following keys {keys[len(pressed):]}")
        return f"I pressed the following keys: {pressed}"

    def mouse_move(self, coordinates: tuple[int, int]):
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
            response = asyncio.run(
                self.tools_map["browser_mouse_move_xy"].ainvoke(
                    {"x": tgt_x, "y": tgt_y}
                )
            )
            self.logger.debug(f"Browser mouse move response: {response}")
        except Exception as e:
            raise ToolException(f"Failed to move mouse due to {e}")

        self.logger.info(f"Mouse moved to ({tgt_x}, {tgt_y}).")
        return f"I moved the cursor to ({tgt_x}, {tgt_y})."

    def type(self, text: str) -> str:
        """
        -------------------------------------------------------
        Type a string of text on the keyboard.
        -------------------------------------------------------
        Parameters:
            text - text to type
        Returns:
            tool response
        -------------------------------------------------------
        """
        self.logger.info(f"Type requested for: {text}")
        # TODO - figure out how to get target and or element
        raise ToolException("Not Implemented yet")
        return f"I typed '{text}'."

    def scroll(self, pixels):
        """
        -------------------------------------------------------
        Performs a scroll of the mouse scroll wheel.
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
            response = asyncio.run(
                self.tools_map["browser_mouse_wheel"].ainvoke(
                    {"deltaX": delta, "deltaY": 0}
                )
            )
            self.logger.debug(f"Browser wheel response: {response}")
        except Exception as e:
            self.logger.error(f"Error during scroll: {e}")
            raise ToolException(f"Failed to move mouse due to {e}")
        direction = "up" if pixels > 0 else "down"
        self.logger.info(f"Scrolled {direction}.")
        return f"I scrolled {direction}."

    def hscroll(self, pixels):
        """
        -------------------------------------------------------
        Performs a horizontal scroll (mapped to regular scroll).
        -------------------------------------------------------
        Parameters:
            pixels - The amount of scrolling to perform. Positive values scroll up,
                negative values scroll down. (int)
        Returns:
            tool response
        -------------------------------------------------------
        """
        self.logger.info(f"Horizontal scroll requested by {pixels} pixels")
        delta = int(pixels * self.viewport_width / self.DISPLAY_SIZE)
        try:
            response = asyncio.run(
                self.tools_map["browser_mouse_wheel"].ainvoke(
                    {"deltaX": 0, "deltaY": delta}
                )
            )
            self.logger.debug(f"Browser wheel response: {response}")
        except Exception as e:
            self.logger.error(f"Error during horizontal scroll: {e}")
            raise ToolException(f"Failed to move mouse due to {e}")

        self.logger.info(f"Horizontally scrolled by {pixels} pixels.")
        return f"I scrolled horizontally by {pixels} pixels."

    def double_click(self, coordinates: tuple[int, int]) -> str:
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
            response = asyncio.run(
                self.tools_map["browser_mouse_click_xy"].ainvoke(
                    {"x": tgt_x, "y": tgt_y, "button": "left", "clickCount": 2}
                )
            )
            self.logger.debug(f"Browser double click response: {response}")
        except Exception as e:
            self.logger.error(f"Error during double click: {e}")
            raise ToolException(f"Failed to double-click at {e}")
        self.logger.info(f"Double clicked at ({tgt_x}, {tgt_y}).")
        return f"I double-clicked at coordinates ({tgt_x}, {tgt_y})."

    def triple_click(self, coordinates: tuple[int, int]):
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
            response = asyncio.run(
                self.tools_map["browser_mouse_click_xy"].ainvoke(
                    {"x": tgt_x, "y": tgt_y, "button": "left", "clickCount": 3}
                )
            )
            self.logger.debug(f"Browser triple click response: {response}")
        except Exception as e:
            self.logger.error(f"Error during triple click: {e}")
            raise ToolException(f"Failed to triple-click at {e}")

        self.logger.info(f"Triple clicked at ({tgt_x}, {tgt_y}).")
        return f"I triple-clicked at coordinates ({tgt_x}, {tgt_y})."

    def left_click_drag(
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
            response = asyncio.run(
                self.tools_map["browser_mouse_drag_xy"].ainvoke(
                    {
                        "startX": tgt_x_stt,
                        "startY": tgt_y_stt,
                        "endX": tgt_x_end,
                        "endY": tgt_y_end,
                    }
                )
            )
            self.logger.debug(f"Browser drag response: {response}")
        except Exception as e:
            self.logger.error(f"Error during drag: {e}")
            raise ToolException(f"Failed to Left click drag due to {e}")

        self.logger.info(
            f"Dragged from ({tgt_x_stt}, {tgt_y_stt}) to ({tgt_x_end}, {tgt_y_end})."
        )
        return (
            f"I dragged from ({tgt_x_stt}, {tgt_y_stt}) to ({tgt_x_end}, {tgt_y_end})."
        )

    def visit_url(self, url: str = "") -> str:
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
            target = f"https://www.google.com/search?q={quote_plus(url)}"
        else:
            target = "https://" + url
        try:
            asyncio.run(self.tools_map["browser_navigate"].ainvoke({"url": target}))
        except Exception as e:
            self.logger.error(f"Error during navigation: {e}")
            raise ToolException(f"I tried to navigate to {url} but it failed: {e}")
        self.logger.info(f"Navigated to {target}.")
        return f"I navigated to {url}."

    def history_back(self) -> str:
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
            asyncio.run(self.tools_map["browser_navigate_back"].ainvoke({}))
        except Exception as e:
            self.logger.error(f"Error during history back: {e}")
            raise ToolException(f"I tried to navigate back but it failed: {e}")

        return "I clicked the browser back button."

    def web_search(self, query: str) -> str:
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
            response = asyncio.run(
                self.tools_map["browser_navigate"].ainvoke(
                    {"url": f"https://www.google.com/search?q={quote_plus(query)}"}
                )
            )
        except Exception as e:
            self.logger.error(f"Error during web search: {e}")
            raise ToolException(f"I tried to search for {query} but it failed: {e}")

        self.logger.info(f"Web search completed for: {query}.")
        return f"I searched for '{query}'."

    def ask_user_question(self, question: str) -> str:
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

    def wait(self, time: int) -> str:
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
        asyncio.run(self.tools_map["browser_wait_for"].ainvoke({"time": time}))
        self.logger.debug(f"Wait completed for {time}s.")
        return f"I waited {time}s."

    def pause_and_memorize_fact(self, fact: str) -> str:
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
        # TODO: Implement
        return f"I memorized the following fact: {fact}"

    def terminate(self, answer) -> str:
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
        ]

        return [
            StructuredTool.from_function(
                func=getattr(self, method),
                parse_docstring=True,
            )
            for method in tool_methods
        ]
