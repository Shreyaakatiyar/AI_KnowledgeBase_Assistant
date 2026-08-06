class KnowledgeBaseError(Exception):
    pass


class DocumentIngestionError(KnowledgeBaseError):
    pass


class EmptyDocumentError(DocumentIngestionError):
    pass


class VectorStoreError(KnowledgeBaseError):
    pass


class RetrievalError(KnowledgeBaseError):
    pass


class LLMGenerationError(KnowledgeBaseError):
    pass