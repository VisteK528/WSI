class Player:
    """A class that represents a player in a game"""
    def __init__(self, char: str) -> None:
        """
        Initializes a player.

        Parameters:
            char: a single-character string to represent the player in textual representations of game state
        """
        if len(char) != 1:
            raise ValueError('Character that represents player should be of length 1')

        self.char = char
