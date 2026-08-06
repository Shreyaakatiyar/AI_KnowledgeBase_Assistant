
from app.core.config import get_settings
s = get_settings()
print(s.llm_model, s.chunk_size)