class CollisionException(Exception):
    """Raised when agents collide"""

    # https://stackoverflow.com/questions/1319615
    def __init__(self, message, payload=None):
        self.message = message
        self.payload = payload

    def __str__(self):
        return str(self.message)
