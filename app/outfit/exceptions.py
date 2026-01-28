class OutfitError(Exception):
    pass


class LLMError(OutfitError):
    pass


class ParseError(OutfitError):
    pass


class InsufficientItemsError(OutfitError):
    pass


class EmptyClosetError(OutfitError):
    pass
