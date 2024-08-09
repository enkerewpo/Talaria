import talaria.functional as functional
from loguru import logger
from PIL import ImageGrab


def capture():
    logger.info("called capture()")
    screenshot = ImageGrab.grab()
    screenshot.save("screenshot.png")
    screenshot.close()


def init():
    functional.Function.new().register("capture", capture)
