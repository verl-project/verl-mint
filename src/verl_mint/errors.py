class BackendError(Exception):
    pass


class UnknownBackendError(BackendError, KeyError):
    pass


class UnknownSessionError(BackendError, KeyError):
    pass


class BackendKindMismatchError(BackendError, TypeError):
    pass


class UnsupportedOperationError(BackendError, RuntimeError):
    pass
