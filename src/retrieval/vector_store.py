class VectorStore:
    """Vector store adapter."""

    def upsert(self) -> None:
        raise NotImplementedError
