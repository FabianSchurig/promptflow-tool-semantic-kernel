import promptflow.tracing


class TracingDisabler:

    def __enter__(self):
        self.original_is_tracing_disabled = promptflow.tracing._utils.is_tracing_disabled
        promptflow.tracing._utils.is_tracing_disabled = lambda: True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        promptflow.tracing._utils.is_tracing_disabled = self.original_is_tracing_disabled
