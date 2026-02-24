class ShopError(Exception):
    pass


class ShopLLMError(ShopError):
    pass


class ShopParseError(ShopError):
    pass


class InsufficientProductsError(ShopError):
    pass
