---
layout: default
title: "PydanticAI Advanced Error Handling and Testing (Python)"
description: "Robust error handling, validation recovery, and testing patterns for PydanticAI."
---

# PydanticAI Advanced Error Handling and Testing (Python)

Last verified: 2025-11

## Validation-First Design
- Model outputs with Pydantic; retry generation on validation errors

```python
from pydantic import BaseModel, ValidationError

class Answer(BaseModel):
    bullets: list[str]

def parse_or_retry(raw: str) -> Answer:
    try:
        return Answer.model_validate_json(raw)
    except ValidationError:
        # re-prompt with error hints or stricter schema
        raise
```

## Testing Strategy
- Unit: model schemas, validators
- Integration: agent runs with mocked model
- Snapshot: golden outputs for prompts
