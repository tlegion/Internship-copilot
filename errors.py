class HarnessError(Exception):
    pass


class ProviderError(HarnessError):
    pass


class StorageError(HarnessError):
    pass


class ToolError(HarnessError):
    pass
