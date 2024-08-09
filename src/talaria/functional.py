from loguru import logger


class Function:
    """
    Function class is a singleton class that holds all the functions that can be used in Talaria.
    """

    __instance = None

    @staticmethod
    def new():
        if Function.__instance is None:
            Function()
        return Function.__instance

    def __init__(self):
        if Function.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            Function.__instance = self
            self.functions = {}

    def register(self, name, function):
        self.functions[name] = function
        logger.debug(f"Function: {name} registered")

    def get(self, name):
        return self.functions[name]

    def get_all(self):
        return self.functions
