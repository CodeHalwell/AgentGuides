---
title: "Class deep dives — volume 22 (credential exchange pipeline, workflow errors, replay sequencing, conformance testing, tool-connection analysis & GoogleSearchAgentTool)"
description: "Source-verified 2.3.0 deep dives: GoogleSearchAgentTool/create_google_search_agent (google_search workaround for multi-tool agents), BaseCredentialExchanger/ExchangeResult/CredentialExchangeError (pluggable credential exchange interface), CredentialExchangerRegistry (type-keyed exchanger registry), OAuth2CredentialExchanger (client credentials + auth code + PKCE flow), SessionStateCredentialService (session-state credential persistence), NodeInterruptedError/NodeTimeoutError/DynamicNodeFailError (workflow error taxonomy), ReplaySequenceBarrier (asyncio chronological replay ordering with 15s timeout), TestSpec/UserMessage/TestCase (conformance test DSL), AdkWebServerClient (conformance test HTTP client with SSE streaming), ToolConnectionAnalyzer/ToolConnectionMap/StatefulParameter (LLM-driven tool dependency analysis for environment simulation)."
framework: google-adk
language: python
sidebar:
  label: "Class deep dives — vol. 22"
  order: 91
---

Source-verified against **google-adk==2.3.0** (installed from PyPI, June 2026). Every field name, signature, constant, and code example is drawn from the installed package source at `/usr/local/lib/python3.11/dist-packages/google/adk/`.

| # | Class / group | Module | Status |
|---|---|---|---|
| 1 | `GoogleSearchAgentTool` + `create_google_search_agent` | `google.adk.tools.google_search_agent_tool` | Stable (internal workaround) |
| 2 | `BaseCredentialExchanger` + `ExchangeResult` + `CredentialExchangeError` | `google.adk.auth.exchanger.base_credential_exchanger` | `@experimental` |
| 3 | `CredentialExchangerRegistry` | `google.adk.auth.exchanger.credential_exchanger_registry` | `@experimental` |
| 4 | `OAuth2CredentialExchanger` | `google.adk.auth.exchanger.oauth2_credential_exchanger` | `@experimental` |
| 5 | `SessionStateCredentialService` | `google.adk.auth.credential_service.session_state_credential_service` | `@experimental` |
| 6 | `NodeInterruptedError` + `NodeTimeoutError` + `DynamicNodeFailError` | `google.adk.workflow._errors` | Stable (internal) |
| 7 | `ReplaySequenceBarrier` | `google.adk.workflow.utils._replay_sequence_barrier` | Stable (internal) |
| 8 | `TestSpec` + `UserMessage` + `TestCase` | `google.adk.cli.conformance.test_case` | Stable |
| 9 | `AdkWebServerClient` | `google.adk.cli.conformance.adk_web_server_client` | Stable |
| 10 | `ToolConnectionAnalyzer` + `ToolConnectionMap` + `StatefulParameter` | `google.adk.tools.environment_simulation.tool_connection_analyzer` + `.tool_connection_map` | `@experimental(ENVIRONMENT_SIMULATION)` |

---

## 1 · `GoogleSearchAgentTool` + `create_google_search_agent` — google_search workaround for multi-tool agents

**Source:** `google/adk/tools/google_search_agent_tool.py`

`GoogleSearchAgentTool` is a concrete `AgentTool` subclass that solves a specific Gemini API constraint: the `google_search` built-in tool cannot be combined with other tools in a single `LlmAgent` tool list because Gemini rejects requests that mix native built-in tools with custom function calls. The workaround wraps the search capability in a dedicated sub-agent that the parent agent delegates to via `AgentTool`, sidestepping the restriction entirely.

`create_google_search_agent(model)` is the companion factory function that builds the correctly-named and instructed sub-agent. Both the function and the class are intended to be used together: call the factory, then pass the returned `LlmAgent` to `GoogleSearchAgentTool.__init__`. `GoogleSearchAgentTool` calls `super().__init__(agent=self.agent, propagate_grounding_metadata=True)` — the `propagate_grounding_metadata=True` flag ensures that Gemini's grounding metadata (search attribution, rendered content, web sources) bubbles up from the sub-agent's response into the parent agent's event stream.

The class carries a source-level `TODO(b/448114567)` marker noting it should be removed once the API restriction is lifted.

### Key behaviours

| Behaviour | Detail |
|---|---|
| Fixed sub-agent name | `create_google_search_agent` always creates an agent named `'google_search_agent'` — do not rename it or sub-agent routing will break. |
| Grounding metadata propagation | `propagate_grounding_metadata=True` is hardcoded in `GoogleSearchAgentTool.__init__`, ensuring search attribution reaches the parent event stream. |
| Model delegation | The model for the sub-agent is supplied by the caller; the parent can use a different model (e.g., cheaper model for search, more expensive for reasoning). |
| Parallel search requests | Because this is an `AgentTool`, the parent agent can call it in parallel with other tools — multiple search intents in one LLM turn are each dispatched to a fresh sub-invocation of the search agent. |
| Tool list constraint | The parent `LlmAgent` must **not** also include `google_search` directly — only `GoogleSearchAgentTool` should appear. Including both causes the original API error. |

### Example 1 — combining google_search with a custom function tool

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.google_search_agent_tool import (
    GoogleSearchAgentTool,
    create_google_search_agent,
)

# 1. Build the dedicated search sub-agent.
search_agent = create_google_search_agent(model="gemini-2.5-flash")
search_tool = GoogleSearchAgentTool(agent=search_agent)


# 2. Define a custom tool that cannot share a tool list with google_search.
def get_weather(city: str) -> dict:
    """Return current weather for a city."""
    return {"city": city, "temp_c": 22, "condition": "sunny"}


# 3. Build the parent agent — search tool and custom tool coexist here.
agent = LlmAgent(
    name="research_assistant",
    model="gemini-2.5-flash",
    instruction=(
        "You can search the web via the google_search_agent tool and also "
        "check weather. Combine both when the user asks about travel."
    ),
    tools=[search_tool, get_weather],
)


async def main():
    runner = InMemoryRunner(agent=agent, app_name="multi_tool_demo")
    session = await runner.session_service.create_session(
        app_name="multi_tool_demo", user_id="u1"
    )
    async for event in runner.run_async(
        user_id="u1",
        session_id=session.id,
        new_message="What's the weather in Paris and what are the top tourist spots there?",
    ):
        if event.is_final_response():
            print(event.content.parts[0].text)


asyncio.run(main())
```

### Example 2 — using a more capable model for search delegation

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.google_search_agent_tool import (
    GoogleSearchAgentTool,
    create_google_search_agent,
)

# Use a lightweight model for the search sub-agent to minimise cost.
# The parent uses a more capable model for synthesis.
search_agent = create_google_search_agent(model="gemini-2.0-flash")
search_tool = GoogleSearchAgentTool(agent=search_agent)


def save_report(title: str, content: str) -> dict:
    """Persist a research report."""
    return {"saved": True, "title": title, "length": len(content)}


parent = LlmAgent(
    name="research_writer",
    model="gemini-2.5-pro",
    instruction="Research topics via google_search_agent and then save reports.",
    tools=[search_tool, save_report],
)


async def main():
    runner = InMemoryRunner(agent=parent, app_name="research_writer")
    session = await runner.session_service.create_session(
        app_name="research_writer", user_id="u1"
    )
    async for event in runner.run_async(
        user_id="u1",
        session_id=session.id,
        new_message="Research the history of the internet and save a report titled 'Internet History'",
    ):
        if event.is_final_response():
            print(event.content.parts[0].text)


asyncio.run(main())
```

### Example 3 — inspecting grounding metadata from a search sub-agent

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.google_search_agent_tool import (
    GoogleSearchAgentTool,
    create_google_search_agent,
)

search_tool = GoogleSearchAgentTool(agent=create_google_search_agent("gemini-2.5-flash"))

agent = LlmAgent(
    name="grounding_demo",
    model="gemini-2.5-flash",
    instruction="Answer questions by searching the web.",
    tools=[search_tool],
)


async def main():
    runner = InMemoryRunner(agent=agent, app_name="grounding_demo")
    session = await runner.session_service.create_session(
        app_name="grounding_demo", user_id="u1"
    )
    async for event in runner.run_async(
        user_id="u1",
        session_id=session.id,
        new_message="What is the current version of Python?",
    ):
        # Grounding metadata is surfaced via event.grounding_metadata
        # because propagate_grounding_metadata=True is set on the tool.
        if event.grounding_metadata:
            print("Search grounding sources:")
            for chunk in event.grounding_metadata.grounding_chunks or []:
                if chunk.web:
                    print(f"  - {chunk.web.title}: {chunk.web.uri}")
        if event.is_final_response():
            print("\nFinal answer:", event.content.parts[0].text)


asyncio.run(main())
```

---

## 2 · `BaseCredentialExchanger` + `ExchangeResult` + `CredentialExchangeError` — credential exchange interface

**Source:** `google/adk/auth/exchanger/base_credential_exchanger.py`

This module defines the three building blocks of ADK's credential exchange pipeline, introduced in 2.3.0 as `@experimental`:

- **`CredentialExchangeError`** — base exception for all exchange failures; raise it from custom exchangers to signal unrecoverable errors (as opposed to returning the original credential unchanged).
- **`ExchangeResult`** — a `NamedTuple` with two fields: `credential: AuthCredential` (the resulting credential, possibly modified) and `was_exchanged: bool` (whether the exchange actually occurred). Callers inspect `was_exchanged` to detect no-ops.
- **`BaseCredentialExchanger`** — abstract base class with a single `@abstractmethod`: `async def exchange(auth_credential, auth_scheme=None) -> ExchangeResult`. Implementations may return the original credential with `was_exchanged=False` when exchange is not applicable or not possible.

### Key behaviours

| Behaviour | Detail |
|---|---|
| `ExchangeResult` is a `NamedTuple` | Immutable, unpacks as `(credential, was_exchanged)` — safe to destructure in callers. |
| No-op return pattern | Return `ExchangeResult(original_credential, False)` rather than raising when exchange simply isn't needed (e.g., token already present). |
| `auth_scheme` is optional | Some exchangers (e.g., API key rotators) don't need the scheme; others (e.g., `OAuth2CredentialExchanger`) require it and raise `CredentialExchangeError` if absent. |
| `@experimental` decorator | Emits a `UserWarning` on first import; may change without notice. |

### Example 1 — implementing a custom API key rotating exchanger

```python
from typing import Optional
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_schemes import AuthScheme
from google.adk.auth.exchanger.base_credential_exchanger import (
    BaseCredentialExchanger,
    CredentialExchangeError,
    ExchangeResult,
)


class ApiKeyRotatingExchanger(BaseCredentialExchanger):
    """Exchanges a placeholder API key for a fresh rotated key."""

    def __init__(self, key_vault_url: str):
        self._vault_url = key_vault_url

    async def exchange(
        self,
        auth_credential: AuthCredential,
        auth_scheme: Optional[AuthScheme] = None,
    ) -> ExchangeResult:
        if auth_credential.api_key is None:
            return ExchangeResult(auth_credential, False)

        # Fetch a fresh key from the vault (simplified).
        fresh_key = await self._fetch_key_from_vault(auth_credential.api_key)
        if fresh_key is None:
            raise CredentialExchangeError(
                f"Vault returned no key for {self._vault_url}"
            )
        updated = auth_credential.model_copy(update={"api_key": fresh_key})
        return ExchangeResult(updated, True)

    async def _fetch_key_from_vault(self, current_key: str) -> Optional[str]:
        # In production: call Secret Manager or Vault API.
        return "rotated-" + current_key
```

### Example 2 — inspecting ExchangeResult

```python
import asyncio
from google.adk.auth.auth_credential import AuthCredential, HttpCredentials
from google.adk.auth.exchanger.base_credential_exchanger import ExchangeResult


async def demonstrate_exchange_result():
    # ExchangeResult is a NamedTuple — can unpack or use named attributes.
    fake_cred = AuthCredential(http=HttpCredentials(token="old-token"))
    result = ExchangeResult(credential=fake_cred, was_exchanged=False)

    credential, was_exchanged = result  # NamedTuple unpacking
    print(f"was_exchanged={was_exchanged}")  # False

    if not result.was_exchanged:
        print("Credential was returned unchanged — skip saving to store.")
    else:
        print("Credential was exchanged — persist updated credential.")


asyncio.run(demonstrate_exchange_result())
```

### Example 3 — error handling patterns for custom exchangers

```python
import asyncio
from typing import Optional
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_schemes import AuthScheme
from google.adk.auth.exchanger.base_credential_exchanger import (
    BaseCredentialExchanger,
    CredentialExchangeError,
    ExchangeResult,
)


class ServiceAccountTokenExchanger(BaseCredentialExchanger):
    """Exchanges a service account JSON key for a short-lived bearer token."""

    async def exchange(
        self,
        auth_credential: AuthCredential,
        auth_scheme: Optional[AuthScheme] = None,
    ) -> ExchangeResult:
        if not auth_credential.service_account:
            # Not applicable to this credential type — signal no-op.
            return ExchangeResult(auth_credential, False)

        try:
            token = await self._mint_token(auth_credential.service_account)
        except ConnectionError as exc:
            # Re-raise domain-specific errors as CredentialExchangeError
            # so callers can catch a single exception type.
            raise CredentialExchangeError(
                f"Failed to mint service account token: {exc}"
            ) from exc

        updated = auth_credential.model_copy(
            update={"http": {"token": token}, "service_account": None}
        )
        return ExchangeResult(updated, True)

    async def _mint_token(self, service_account) -> str:
        return "ya29.minted-token"


async def main():
    exchanger = ServiceAccountTokenExchanger()
    cred = AuthCredential(api_key="placeholder")
    try:
        result = await exchanger.exchange(cred)
        print("Exchanged:", result.was_exchanged)
    except CredentialExchangeError as e:
        print("Exchange failed:", e)


asyncio.run(main())
```

---

## 3 · `CredentialExchangerRegistry` — type-keyed exchanger registry

**Source:** `google/adk/auth/exchanger/credential_exchanger_registry.py`

`CredentialExchangerRegistry` is an `@experimental` registry that maps `AuthCredentialTypes` enum values to `BaseCredentialExchanger` instances. Its internal store is a plain `dict[AuthCredentialTypes, BaseCredentialExchanger]` (`_exchangers`). The two public methods are `register(credential_type, exchanger_instance)` and `get_exchanger(credential_type) -> Optional[BaseCredentialExchanger]`.

The registry pattern decouples exchange logic selection from the auth pipeline: the pipeline asks the registry for an exchanger by credential type and, if one is found, delegates to it — without needing to know which concrete exchanger handles each type. Build one registry per application and inject it into the credential manager.

### Key behaviours

| Behaviour | Detail |
|---|---|
| Dict-backed | `_exchangers` is a plain `dict` — last `register()` call for a given type wins; no duplicate protection. |
| Returns `None` for missing types | `get_exchanger()` returns `None` rather than raising for unregistered types — callers should check before calling `.exchange()`. |
| Instance-based | One exchanger *instance* per type — the instance may carry per-type configuration (e.g., token endpoint URL). |
| `@experimental` | Emits a `UserWarning` on import; API may change. |

### Example 1 — building and populating a registry

```python
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.exchanger.credential_exchanger_registry import CredentialExchangerRegistry
from google.adk.auth.exchanger.oauth2_credential_exchanger import OAuth2CredentialExchanger

registry = CredentialExchangerRegistry()

# Register the built-in OAuth2 exchanger for OAuth2 credential types.
oauth2_exchanger = OAuth2CredentialExchanger()
registry.register(AuthCredentialTypes.OAUTH2, oauth2_exchanger)

# Register for OIDC as well.
registry.register(AuthCredentialTypes.OPEN_ID_CONNECT, oauth2_exchanger)

# Look up by type.
exchanger = registry.get_exchanger(AuthCredentialTypes.OAUTH2)
if exchanger:
    print("Found exchanger:", type(exchanger).__name__)  # OAuth2CredentialExchanger
else:
    print("No exchanger registered for OAuth2")
```

### Example 2 — custom exchanger in a registry

```python
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import asyncio
from typing import Optional
from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes
from google.adk.auth.auth_schemes import AuthScheme
from google.adk.auth.exchanger.base_credential_exchanger import (
    BaseCredentialExchanger,
    ExchangeResult,
)
from google.adk.auth.exchanger.credential_exchanger_registry import CredentialExchangerRegistry


class ServiceTokenExchanger(BaseCredentialExchanger):
    def __init__(self, service_url: str):
        self._url = service_url

    async def exchange(
        self, auth_credential: AuthCredential, auth_scheme: Optional[AuthScheme] = None
    ) -> ExchangeResult:
        # Exchange service account for bearer token (simplified).
        return ExchangeResult(auth_credential, False)


registry = CredentialExchangerRegistry()
registry.register(
    AuthCredentialTypes.SERVICE_ACCOUNT,
    ServiceTokenExchanger(service_url="https://iam.googleapis.com/token"),
)


async def run_exchange(cred: AuthCredential, cred_type: AuthCredentialTypes):
    exchanger = registry.get_exchanger(cred_type)
    if not exchanger:
        print(f"No exchanger for {cred_type}")
        return
    result = await exchanger.exchange(cred)
    print("was_exchanged:", result.was_exchanged)


asyncio.run(run_exchange(AuthCredential(api_key="key"), AuthCredentialTypes.SERVICE_ACCOUNT))
```

### Example 3 — overriding a registered exchanger at runtime

```python
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.exchanger.credential_exchanger_registry import CredentialExchangerRegistry
from google.adk.auth.exchanger.oauth2_credential_exchanger import OAuth2CredentialExchanger


class MockOAuth2Exchanger(OAuth2CredentialExchanger):
    """Test double that always reports exchange as skipped."""

    async def exchange(self, auth_credential, auth_scheme=None):
        result = await super().exchange(auth_credential, auth_scheme)
        return result._replace(was_exchanged=False)


registry = CredentialExchangerRegistry()
registry.register(AuthCredentialTypes.OAUTH2, OAuth2CredentialExchanger())

# In tests: swap in a mock without modifying the pipeline.
registry.register(AuthCredentialTypes.OAUTH2, MockOAuth2Exchanger())

exchanger = registry.get_exchanger(AuthCredentialTypes.OAUTH2)
print(type(exchanger).__name__)  # MockOAuth2Exchanger — the latest register() wins
```

---

## 4 · `OAuth2CredentialExchanger` — OAuth2 credential exchange (client credentials, auth code, PKCE)

**Source:** `google/adk/auth/exchanger/oauth2_credential_exchanger.py`

`OAuth2CredentialExchanger` is the built-in `@experimental` implementation of `BaseCredentialExchanger` for OAuth2 and OIDC credentials. It handles two grant types — `client_credentials` and `authorization_code` — and requires `authlib` to be installed (gracefully skips with a warning if not available).

The `exchange()` method performs these steps in order:
1. **No-op check** — if `auth_credential.oauth2.access_token` is already set, return immediately with `was_exchanged=False`.
2. **Grant type detection** — inspects the `auth_scheme.flows` (for `OAuth2` schemes) or `auth_scheme.grant_types_supported` (for `OpenIdConnectWithConfig`); defaults OIDC to `authorization_code` unless `client_credentials` is explicitly supported.
3. **Client credentials exchange** — calls `client.fetch_token(token_endpoint, grant_type=OAuthGrantType.CLIENT_CREDENTIALS)` via `authlib`.
4. **Auth code exchange** — calls `client.fetch_token(token_endpoint, authorization_response=..., code=..., grant_type=...)` with optional `code_verifier` for PKCE; strips a trailing `#` from `auth_response_uri` before passing to `authlib` (which can add extraneous trailing hashes).

### Key behaviours

| Behaviour | Detail |
|---|---|
| authlib guard | If `authlib` is not installed, returns `ExchangeResult(credential, False)` after a warning — never raises. |
| Short-circuit on existing token | Checks `auth_credential.oauth2.access_token` presence first — avoids redundant token fetches on repeat invocations. |
| PKCE via `code_verifier` | If `auth_credential.oauth2.code_verifier` is set, it is forwarded to `fetch_token()` as `code_verifier=` keyword arg. |
| Trailing `#` strip | `_normalize_auth_uri()` drops a trailing `#` from `auth_response_uri` — works around an `authlib` edge case. |
| Error handling | Logs errors but returns `ExchangeResult(credential, False)` on exchange failure — callers can still use the original credential. |

### Example 1 — client credentials grant exchange

```python
import asyncio
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from fastapi.openapi.models import OAuth2, OAuthFlows, OAuthFlowClientCredentials
from google.adk.auth.auth_credential import AuthCredential, OAuth2Auth
from google.adk.auth.exchanger.oauth2_credential_exchanger import OAuth2CredentialExchanger


async def demo_client_credentials():
    # Build an OAuth2 scheme with a clientCredentials flow.
    scheme = OAuth2(
        flows=OAuthFlows(
            clientCredentials=OAuthFlowClientCredentials(
                tokenUrl="https://auth.example.com/oauth2/token",
                scopes={"read:data": "Read access"},
            )
        )
    )

    # Credential carries client_id and client_secret.
    cred = AuthCredential(
        oauth2=OAuth2Auth(
            client_id="my-service-client",
            client_secret="secret",
        )
    )

    exchanger = OAuth2CredentialExchanger()
    result = await exchanger.exchange(cred, auth_scheme=scheme)
    print("Exchanged:", result.was_exchanged)
    if result.was_exchanged:
        print("Access token:", result.credential.oauth2.access_token)


asyncio.run(demo_client_credentials())
```

### Example 2 — authorization code grant with PKCE

```python
import asyncio
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from fastapi.openapi.models import OAuth2, OAuthFlows, OAuthFlowAuthorizationCode
from google.adk.auth.auth_credential import AuthCredential, OAuth2Auth
from google.adk.auth.exchanger.oauth2_credential_exchanger import OAuth2CredentialExchanger


async def demo_auth_code_with_pkce():
    scheme = OAuth2(
        flows=OAuthFlows(
            authorizationCode=OAuthFlowAuthorizationCode(
                authorizationUrl="https://auth.example.com/authorize",
                tokenUrl="https://auth.example.com/oauth2/token",
                scopes={"openid": "OpenID", "profile": "Profile"},
            )
        )
    )

    # Credential carries the auth code returned from the redirect callback
    # and the PKCE code_verifier generated before the authorization redirect.
    cred = AuthCredential(
        oauth2=OAuth2Auth(
            client_id="my-app-client",
            client_secret="secret",
            auth_code="AUTH_CODE_FROM_REDIRECT",
            auth_response_uri="https://app.example.com/callback?code=AUTH_CODE_FROM_REDIRECT#",
            code_verifier="random-verifier-string-64-chars",
        )
    )

    exchanger = OAuth2CredentialExchanger()
    result = await exchanger.exchange(cred, auth_scheme=scheme)
    print("Exchanged:", result.was_exchanged)


asyncio.run(demo_auth_code_with_pkce())
```

### Example 3 — short-circuit on already-valid token

```python
import asyncio
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from fastapi.openapi.models import OAuth2, OAuthFlows, OAuthFlowClientCredentials
from google.adk.auth.auth_credential import AuthCredential, OAuth2Auth
from google.adk.auth.exchanger.oauth2_credential_exchanger import OAuth2CredentialExchanger


async def demo_skip_exchange_when_token_present():
    scheme = OAuth2(
        flows=OAuthFlows(
            clientCredentials=OAuthFlowClientCredentials(
                tokenUrl="https://auth.example.com/oauth2/token",
                scopes={},
            )
        )
    )

    # Credential already has an access_token — exchange should be skipped.
    cred = AuthCredential(
        oauth2=OAuth2Auth(
            client_id="client",
            client_secret="secret",
            access_token="already-valid-token",
        )
    )

    exchanger = OAuth2CredentialExchanger()
    result = await exchanger.exchange(cred, auth_scheme=scheme)
    print("Exchanged:", result.was_exchanged)  # False — no network call made
    print("Token unchanged:", result.credential.oauth2.access_token)  # already-valid-token


asyncio.run(demo_skip_exchange_when_token_present())
```

---

## 5 · `SessionStateCredentialService` — session-state credential persistence

**Source:** `google/adk/auth/credential_service/session_state_credential_service.py`

`SessionStateCredentialService` is an `@experimental` `BaseCredentialService` that stores and retrieves `AuthCredential` objects directly in the session's mutable state dictionary. It is the simplest possible persistence backend — no external storage, no encryption — suitable for development and low-sensitivity workloads.

`load_credential(auth_config, callback_context)` performs a `callback_context.state.get(auth_config.credential_key)` and returns the value (or `None` if absent). `save_credential(auth_config, callback_context)` writes `auth_config.exchanged_auth_credential` to `callback_context.state[auth_config.credential_key]`.

The source carries a prominent docstring caveat: **storing credentials in session state may not be secure** — use at your own risk. For production, prefer `InMemoryCredentialService` (volatile, in-process) or a custom `BaseCredentialService` backed by a secrets manager.

### Key behaviours

| Behaviour | Detail |
|---|---|
| State key | `auth_config.credential_key` is used as the dict key — derives from the tool's auth scheme configuration. |
| Scope tied to session | Credentials expire when the session is deleted or garbage-collected; there is no cross-session sharing. |
| No serialisation | Stores the `AuthCredential` object directly in session state — relies on the session service's serialisation for persistence across restarts. |
| `@experimental` | Emits a `UserWarning` on import. |

### Example 1 — registering SessionStateCredentialService with an agent

```python
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import asyncio
from google.adk.agents import LlmAgent
from google.adk.apps.app import App
from google.adk.auth.auth_credential import AuthCredential, OAuth2Auth
from google.adk.auth.auth_schemes import OAuthGrantType
from google.adk.auth.credential_service.session_state_credential_service import (
    SessionStateCredentialService,
)
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.authenticated_function_tool import AuthenticatedFunctionTool


def get_calendar_events(auth_token: str) -> list:
    """Fetch calendar events using the provided auth token."""
    # In production: call Google Calendar API with auth_token.
    return [{"title": "Team meeting", "time": "10:00"}]


agent = LlmAgent(
    name="calendar_agent",
    model="gemini-2.5-flash",
    instruction="Help manage calendar events.",
    tools=[get_calendar_events],
)

session_service = InMemorySessionService()
# Attach SessionStateCredentialService to the App-level credential service.
app = App(
    name="calendar_app",
    agent=agent,
    credential_service=SessionStateCredentialService(),
)
runner = Runner(
    agent=agent,
    app_name="calendar_app",
    session_service=session_service,
)
print("App configured with SessionStateCredentialService")
```

### Example 2 — manually testing load/save via a mock CallbackContext

```python
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import asyncio
from unittest.mock import MagicMock
from google.adk.auth.auth_credential import AuthCredential, OAuth2Auth
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.credential_service.session_state_credential_service import (
    SessionStateCredentialService,
)


async def test_session_state_round_trip():
    service = SessionStateCredentialService()

    # Build a mock auth_config whose credential_key is "oauth2_creds".
    auth_config = MagicMock()
    auth_config.credential_key = "oauth2_creds"
    auth_config.exchanged_auth_credential = AuthCredential(
        oauth2=OAuth2Auth(access_token="test-token-123")
    )

    # Build a mock callback_context with a real mutable state dict.
    state = {}
    callback_context = MagicMock()
    callback_context.state = state

    # Test save.
    await service.save_credential(auth_config, callback_context)
    print("State keys after save:", list(state.keys()))  # ['oauth2_creds']

    # Test load.
    loaded = await service.load_credential(auth_config, callback_context)
    print("Loaded access_token:", loaded.oauth2.access_token)  # test-token-123

    # Test missing credential.
    auth_config.credential_key = "nonexistent"
    missing = await service.load_credential(auth_config, callback_context)
    print("Missing credential:", missing)  # None


asyncio.run(test_session_state_round_trip())
```

### Example 3 — custom credential service for comparison (Redis-backed)

```python
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import asyncio
import json
from typing import Optional
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.credential_service.base_credential_service import BaseCredentialService


class RedisCredentialService(BaseCredentialService):
    """Production credential service backed by Redis for cross-session sharing.

    Contrast with SessionStateCredentialService: credentials survive
    session deletion and are accessible across multiple sessions for the
    same user.
    """

    def __init__(self, redis_client, ttl_seconds: int = 3600):
        self._redis = redis_client
        self._ttl = ttl_seconds

    async def load_credential(
        self, auth_config: AuthConfig, callback_context
    ) -> Optional[AuthCredential]:
        key = f"creds:{auth_config.credential_key}:{callback_context.user_id}"
        raw = await self._redis.get(key)
        if raw is None:
            return None
        return AuthCredential.model_validate(json.loads(raw))

    async def save_credential(
        self, auth_config: AuthConfig, callback_context
    ) -> None:
        key = f"creds:{auth_config.credential_key}:{callback_context.user_id}"
        cred = auth_config.exchanged_auth_credential
        await self._redis.setex(key, self._ttl, cred.model_dump_json())


# Usage:
# import redis.asyncio as aioredis
# redis_client = aioredis.from_url("redis://localhost:6379")
# service = RedisCredentialService(redis_client, ttl_seconds=7200)
print("RedisCredentialService defined — use as drop-in for SessionStateCredentialService")
```

---

## 6 · `NodeInterruptedError` + `NodeTimeoutError` + `DynamicNodeFailError` — workflow error taxonomy

**Source:** `google/adk/workflow/_errors.py`

Three distinct exception types form the workflow framework's internal error taxonomy, each with a specific base class that governs how it interacts with user code and retry logic:

- **`NodeInterruptedError(BaseException)`** — raised exclusively by `ctx.run_node()` when a dynamic child node has unresolved HITL interrupt IDs. Extends `BaseException` deliberately — `except Exception` will **not** catch it, ensuring the interrupt propagates up to the framework's `NodeRunner` rather than being accidentally swallowed by user-level error handling.
- **`NodeTimeoutError(Exception)`** — raised when a node exceeds its configured `timeout` seconds. Extends `Exception` (not `BaseException`) so it **is** catchable by `retry_config` — a timed-out node can be automatically retried. Constructor: `NodeTimeoutError(node_name: str, timeout: float)` → message `"Node 'X' timed out after Y seconds."`.
- **`DynamicNodeFailError(Exception)`** — raised when a dynamic node fails; wraps the underlying exception and carries `error: Exception` and `error_node_path: str` attributes. Caught by the parent node's `NodeRunner` to propagate the child's failure upward.

### Key behaviours

| Exception | Base class | `except Exception` catches it? | Retryable? | Detail |
|---|---|---|---|---|
| `NodeInterruptedError` | `BaseException` | No | No | HITL signal; caught only by the framework's NodeRunner |
| `NodeTimeoutError` | `Exception` | Yes | Yes | Works with `retry_config`; carries `node_name` and `timeout` attrs |
| `DynamicNodeFailError` | `Exception` | Yes | No | Carries `error` (original exception) and `error_node_path` attrs |

### Example 1 — understanding NodeInterruptedError cannot be caught by user code

```python
from google.adk.workflow._errors import NodeInterruptedError

# NodeInterruptedError extends BaseException, NOT Exception.
# The following user-level try/except will NOT intercept it.
try:
    raise NodeInterruptedError("HITL interrupt occurred")
except Exception as e:
    print("Caught by 'except Exception' — this line NEVER runs")
except BaseException as e:
    print(f"Caught by 'except BaseException': {type(e).__name__}")
    # NodeInterruptedError — the framework's NodeRunner catches it here.
```

### Example 2 — NodeTimeoutError in a retryable workflow node

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.workflow._workflow import Workflow
from google.adk.workflow._node import node
from google.adk.workflow._retry_config import RetryConfig
from google.adk.workflow._errors import NodeTimeoutError


@node(
    timeout=5.0,  # Node must complete within 5 seconds.
    retry_config=RetryConfig(max_retries=3, initial_delay=1.0, backoff_factor=2.0),
)
async def slow_api_call(ctx):
    """Calls an external API that may time out."""
    import asyncio
    # Simulate a slow API.  In production, a real HTTP call goes here.
    await asyncio.sleep(10)  # Will trigger NodeTimeoutError after 5s.
    return {"result": "ok"}


async def main():
    workflow = Workflow(
        name="timeout_demo",
        nodes=[slow_api_call],
    )
    # NodeTimeoutError is raised internally, caught by retry_config, and
    # the node is retried up to 3 times before propagating as
    # DynamicNodeFailError to the parent.
    print("Workflow defined — NodeTimeoutError triggers retry_config automatically")


asyncio.run(main())
```

### Example 3 — inspecting DynamicNodeFailError in error handling

```python
from google.adk.workflow._errors import DynamicNodeFailError

# DynamicNodeFailError carries the original error and the node path.
original_error = ValueError("API returned 500")
fail_error = DynamicNodeFailError(
    message="Dynamic node failed at /root/fetch_data",
    error=original_error,
    error_node_path="/root/fetch_data",
)

print("error_node_path:", fail_error.error_node_path)  # /root/fetch_data
print("wrapped error:", fail_error.error)  # ValueError: API returned 500
print("str(fail_error):", str(fail_error))  # Dynamic node failed at /root/fetch_data

# Pattern: catch DynamicNodeFailError to identify which child failed.
try:
    raise fail_error
except DynamicNodeFailError as e:
    print(f"Node '{e.error_node_path}' failed with: {type(e.error).__name__}: {e.error}")
```

---

## 7 · `ReplaySequenceBarrier` — asyncio chronological replay ordering

**Source:** `google/adk/workflow/utils/_replay_sequence_barrier.py`

`ReplaySequenceBarrier` enforces deterministic replay order across concurrently-replaying workflow nodes. When a workflow is rehydrated from session history, multiple nodes may resume simultaneously but their side effects must reproduce in the original chronological order. The barrier solves this using a pre-built dictionary of `asyncio.Event` objects — one per sequence key — with a chain-unlock mechanism.

**Construction:** `ReplaySequenceBarrier(sequence: list[str], timeout_sec: float = 15.0)` — the `sequence` list defines the intended execution order. An `asyncio.Event` is created for each key; the first key's event is pre-set (`.set()`) so the first node can proceed immediately.

**`wait(key)`** — if `key` is in the sequence, awaits its event with `timeout_sec` seconds. If `key` is not in the sequence ("silent" nodes that only wrote state but produced no output), it returns immediately (fast-forward). A timeout raises `RuntimeError("Replay divergence detected: Timed out waiting for sequence key '...' to be unblocked.")`.

**`check_and_advance(key)`** — called after a node completes. If `key` matches `sequence[current_index]`, increments `current_index` and sets the next event, unblocking the next waiting node.

### Key behaviours

| Behaviour | Detail |
|---|---|
| First key pre-set | `sequence[0]`'s event is `set()` in `__init__` — the first node does not wait. |
| "Silent" node fast-forward | Nodes not in `sequence` skip the `wait()` call and execute immediately, without blocking the chain. |
| 15-second default timeout | `asyncio.wait_for(event.wait(), timeout=15.0)` — raises `RuntimeError` with "Replay divergence detected" message if exceeded. |
| Chain-unlock pattern | Each `check_and_advance()` call unlocks exactly one subsequent event — strictly serial unblocking. |
| asyncio-native | Uses `asyncio.Event` — must be created and used within the same event loop. |

### Example 1 — basic barrier for three sequential nodes

```python
import asyncio
from google.adk.workflow.utils._replay_sequence_barrier import ReplaySequenceBarrier


async def demo_sequence_barrier():
    # Three nodes in chronological order from session history.
    sequence = ["fetch_data", "transform_data", "save_result"]
    barrier = ReplaySequenceBarrier(sequence, timeout_sec=5.0)

    async def replay_node(name: str, delay: float = 0):
        """Simulates a replaying node."""
        await asyncio.sleep(delay)  # Simulate concurrent replay startup.
        await barrier.wait(name)    # Block until it's this node's turn.
        print(f"Node '{name}' replaying...")
        await asyncio.sleep(0.05)  # Simulate replay work.
        barrier.check_and_advance(name)  # Unlock the next node.

    # All three nodes start concurrently but replay in chronological order.
    await asyncio.gather(
        replay_node("save_result", delay=0),     # Starts first but waits.
        replay_node("transform_data", delay=0),  # Waits for fetch_data.
        replay_node("fetch_data", delay=0),      # Proceeds immediately.
    )
    print("All nodes replayed in correct order: fetch → transform → save")


asyncio.run(demo_sequence_barrier())
```

### Example 2 — silent node fast-forward

```python
import asyncio
from google.adk.workflow.utils._replay_sequence_barrier import ReplaySequenceBarrier


async def demo_silent_node():
    # 'log_event' produced only state updates — not in the sequence.
    sequence = ["fetch_data", "save_result"]
    barrier = ReplaySequenceBarrier(sequence, timeout_sec=5.0)

    async def replay_with_silent(name: str):
        await barrier.wait(name)  # 'log_event' is not in sequence — fast-forwards.
        print(f"Node '{name}' replaying")
        barrier.check_and_advance(name)

    await asyncio.gather(
        replay_with_silent("fetch_data"),
        replay_with_silent("save_result"),
        replay_with_silent("log_event"),   # Not in sequence; executes immediately.
    )
    print("Done — log_event ran without blocking the sequence chain")


asyncio.run(demo_silent_node())
```

### Example 3 — timeout detection for diverged replay

```python
import asyncio
from google.adk.workflow.utils._replay_sequence_barrier import ReplaySequenceBarrier


async def demo_timeout():
    # Sequence expects 'node_a' before 'node_b', but 'node_a' never advances.
    barrier = ReplaySequenceBarrier(["node_a", "node_b"], timeout_sec=0.5)

    # Manually set 'node_a' as the first, but never call check_and_advance("node_a").
    # 'node_b' will time out trying to wait.
    try:
        await barrier.wait("node_b")
    except RuntimeError as e:
        # RuntimeError: Replay divergence detected: Timed out waiting for
        # sequence key 'node_b' to be unblocked.
        assert "Replay divergence detected" in str(e)
        print("Caught expected divergence error:", e)


asyncio.run(demo_timeout())
```

---

## 8 · `TestSpec` + `UserMessage` + `TestCase` — conformance test DSL

**Source:** `google/adk/cli/conformance/test_case.py`

ADK 2.3.0 adds a YAML-driven conformance test DSL for exercising agents through the `adk` web server. The three types form the data layer of the `adk conformance` CLI command.

- **`UserMessage(BaseModel)`** — represents one turn of user input. Exactly one of `text: Optional[str]` or `content: Optional[types.UserContent]` should be set (oneof semantics, not enforced by Pydantic). An optional `state_delta: Optional[dict[str, Any]]` applies state changes before the turn is sent.
- **`TestSpec(BaseModel)`** — the human-authored test specification loaded from `spec.yaml`. Configures `extra="forbid"` so unknown YAML keys are rejected. Fields: `description: str`, `agent: str` (ADK agent name), `initial_state: dict[str, Any]` (session creation state), `user_messages: list[UserMessage]`.
- **`TestCase`** — a `@dataclass` (not Pydantic) composed after filesystem discovery. Fields: `category: str` (parent folder name), `name: str` (test folder name), `dir: Path` (absolute path to the test folder), `test_spec: TestSpec`.

### Key behaviours

| Behaviour | Detail |
|---|---|
| `extra="forbid"` on `TestSpec` | Unknown YAML keys cause `ValidationError` at load time — typos in spec files are caught early. |
| `TestCase` is a `@dataclass` | Discovered at runtime by `ConformanceTestRunner._discover_test_cases()` — not loaded from YAML directly. |
| `state_delta` per message | Each `UserMessage` can carry a `state_delta` that is applied to the session before sending the message — enables stateful multi-turn test scenarios. |
| Category/name from path | `category = test_case_dir.parent.name`, `name = test_case_dir.name` — folder structure `tests/{category}/{name}/spec.yaml`. |
| Recordings files | Replay mode looks for `generated-recordings.yaml` (non-streaming) or `generated-recordings-sse.yaml` (SSE). |

### Example 1 — writing a spec.yaml and loading it in Python

```python
# spec.yaml (lives at tests/search/basic_search/spec.yaml):
# description: "Verify the agent uses google_search for factual questions"
# agent: research_agent
# initial_state:
#   user_tier: premium
# user_messages:
#   - text: "What is the capital of France?"
#   - text: "And what is its population?"
#     state_delta:
#       follow_up: true

import yaml
from pathlib import Path
from google.adk.cli.conformance.test_case import TestSpec, UserMessage, TestCase

spec_yaml = """
description: "Verify the agent uses google_search for factual questions"
agent: research_agent
initial_state:
  user_tier: premium
user_messages:
  - text: "What is the capital of France?"
  - text: "And what is its population?"
    state_delta:
      follow_up: true
"""

spec_dict = yaml.safe_load(spec_yaml)
test_spec = TestSpec.model_validate(spec_dict)
print("Agent under test:", test_spec.agent)            # research_agent
print("Initial state:", test_spec.initial_state)       # {'user_tier': 'premium'}
print("Turns:", len(test_spec.user_messages))          # 2
print("Turn 2 state_delta:", test_spec.user_messages[1].state_delta)  # {'follow_up': True}

# Build a TestCase (normally done by ConformanceTestRunner).
test_case = TestCase(
    category="search",
    name="basic_search",
    dir=Path("tests/search/basic_search"),
    test_spec=test_spec,
)
print("TestCase path:", test_case.dir)
```

### Example 2 — unknown keys are rejected by TestSpec

```python
import yaml
from google.adk.cli.conformance.test_case import TestSpec
from pydantic import ValidationError

bad_spec_yaml = """
description: "Test with unknown key"
agent: my_agent
unknown_field: this_should_fail
user_messages: []
"""

try:
    TestSpec.model_validate(yaml.safe_load(bad_spec_yaml))
except ValidationError as e:
    # extra='forbid' catches the typo immediately.
    print("Validation error:", e.error_count(), "error(s)")
    for error in e.errors():
        print(f"  - {error['loc']}: {error['msg']}")
```

### Example 3 — UserMessage with typed content (not just text)

```python
from google.genai import types
from google.adk.cli.conformance.test_case import UserMessage

# Text-based user message (most common).
text_msg = UserMessage(text="Show me a recipe for pasta.")
print("text_msg.text:", text_msg.text)

# Content-based message with an image part.
image_part = types.Part(
    inline_data=types.Blob(mime_type="image/jpeg", data=b"\xff\xd8\xff")  # minimal JPEG header
)
content_msg = UserMessage(
    content=types.UserContent(parts=[image_part]),
    state_delta={"has_image": True},
)
print("content_msg.content type:", type(content_msg.content).__name__)
print("content_msg.state_delta:", content_msg.state_delta)
```

---

## 9 · `AdkWebServerClient` — conformance test HTTP client with SSE streaming

**Source:** `google/adk/cli/conformance/adk_web_server_client.py`

`AdkWebServerClient` is an async HTTP client purpose-built for ADK conformance testing. It wraps `httpx.AsyncClient` and communicates with the `adk web` FastAPI server (`/apps/...`, `/run_sse`, `/version`). The client supports both manual lifecycle management and async context manager usage.

Key methods:
- `create_session(app_name, user_id, state=None)` → `Session`
- `run_agent(request, mode=None, test_case_dir=None, user_message_index=None)` → `AsyncGenerator[Event, None]` — streams SSE events; injects `_adk_recordings_config` or `_adk_replay_config` into `request.state_delta` when `mode` is `"record"` or `"replay"`.
- `update_session(app_name, user_id, session_id, state_delta)` → `Session` — applies state changes without running the agent (via `PATCH`).
- `get_artifact_version_metadata(...)` and `list_artifact_versions_metadata(...)` — artifact introspection.

The client has a single internal `httpx.AsyncClient` instance (`_client`) created lazily on first `_get_client()` call and reused across calls. `close()` / `__aexit__` dispose it via `aclose()`.

### Key behaviours

| Behaviour | Detail |
|---|---|
| Default `base_url` | `http://127.0.0.1:8000` — matches the default port of `adk web`. |
| Default `timeout` | `30.0` seconds — wrap with `httpx.Timeout(...)` internally. |
| `_adk_replay_config` injection | Sets `streaming_mode: "sse"` or `"none"` based on `request.streaming` — must match the mode used during recording. |
| SSE parse | Reads lines starting with `"data:"`, strips the prefix, parses JSON, validates as `Event`. Lines with `"error"` key in the JSON raise `RuntimeError`. |
| `state_delta` mutation | `run_agent()` mutates `request.state_delta` in place when `mode` is set — initialises the dict if `None`. |
| Context manager lifecycle | `async with AdkWebServerClient() as client: ...` — `aclose()` is called on `__aexit__`. |

### Example 1 — running an agent and streaming events

```python
import asyncio
from pathlib import Path
from google.adk.cli.conformance.adk_web_server_client import AdkWebServerClient
from google.adk.cli.adk_web_server import RunAgentRequest


async def run_conformance_replay(test_case_dir: str, user_message_index: int):
    """Run a single conformance replay turn and print collected events."""
    async with AdkWebServerClient(base_url="http://127.0.0.1:8000") as client:
        # Create a session for the test.
        session = await client.create_session(
            app_name="research_app",
            user_id="conformance_tester",
            state={"user_tier": "premium"},
        )
        print("Created session:", session.id)

        request = RunAgentRequest(
            app_name="research_app",
            user_id="conformance_tester",
            session_id=session.id,
            new_message="What is the capital of France?",
            streaming=False,
        )

        events = []
        async for event in client.run_agent(
            request,
            mode="replay",
            test_case_dir=test_case_dir,
            user_message_index=user_message_index,
        ):
            events.append(event)
            if event.is_final_response():
                print("Final response:", event.content.parts[0].text)

        print(f"Total events received: {len(events)}")

        # Cleanup.
        await client.delete_session(
            app_name="research_app",
            user_id="conformance_tester",
            session_id=session.id,
        )


# asyncio.run(run_conformance_replay("tests/search/basic_search", user_message_index=0))
print("AdkWebServerClient example defined — start 'adk web' before running")
```

### Example 2 — record mode to generate a golden YAML fixture

```python
import asyncio
from google.adk.cli.conformance.adk_web_server_client import AdkWebServerClient
from google.adk.cli.adk_web_server import RunAgentRequest


async def record_golden_run(output_dir: str):
    """Record agent responses as a golden fixture for future conformance tests."""
    async with AdkWebServerClient() as client:
        session = await client.create_session(
            app_name="research_app",
            user_id="golden_recorder",
        )

        user_messages = [
            "What is the capital of France?",
            "What is its population?",
        ]

        for index, message in enumerate(user_messages):
            request = RunAgentRequest(
                app_name="research_app",
                user_id="golden_recorder",
                session_id=session.id,
                new_message=message,
                streaming=False,
            )
            # mode="record" injects _adk_recordings_config into state_delta.
            async for event in client.run_agent(
                request,
                mode="record",
                test_case_dir=output_dir,
                user_message_index=index,
            ):
                if event.is_final_response():
                    print(f"Turn {index}: recorded response")

        print(f"Golden recording saved to {output_dir}/generated-recordings.yaml")


# asyncio.run(record_golden_run("tests/search/basic_search"))
print("Record example defined — requires running 'adk web' server")
```

### Example 3 — applying state_delta between turns via update_session

```python
import asyncio
from google.adk.cli.conformance.adk_web_server_client import AdkWebServerClient
from google.adk.cli.adk_web_server import RunAgentRequest


async def multi_turn_with_state_update():
    """Demonstrate mid-session state mutation without running the agent."""
    async with AdkWebServerClient() as client:
        session = await client.create_session(
            app_name="stateful_app",
            user_id="test_user",
            state={"phase": "onboarding"},
        )
        print("Initial phase:", session.state.get("phase"))  # onboarding

        # Mutate state between turns without invoking the agent.
        updated_session = await client.update_session(
            app_name="stateful_app",
            user_id="test_user",
            session_id=session.id,
            state_delta={"phase": "active", "credits": 100},
        )
        print("Updated phase:", updated_session.state.get("phase"))  # active

        request = RunAgentRequest(
            app_name="stateful_app",
            user_id="test_user",
            session_id=session.id,
            new_message="Check my account status",
            streaming=False,
        )
        async for event in client.run_agent(request):
            if event.is_final_response():
                print("Agent response received")


# asyncio.run(multi_turn_with_state_update())
print("Multi-turn state update example defined")
```

---

## 10 · `ToolConnectionAnalyzer` + `ToolConnectionMap` + `StatefulParameter` — LLM-driven tool dependency analysis

**Source:** `google/adk/tools/environment_simulation/tool_connection_analyzer.py` and `tool_connection_map.py`

These three classes form the analysis layer of ADK's `@experimental(FeatureName.ENVIRONMENT_SIMULATION)` environment simulation feature. Rather than requiring developers to manually specify which tools create or consume stateful identifiers (like `ticket_id` or `order_id`), `ToolConnectionAnalyzer` delegates this analysis to an LLM.

**`StatefulParameter(BaseModel)`** — represents a single stateful parameter and its tool associations: `parameter_name: str`, `creating_tools: List[str]`, `consuming_tools: List[str]`.

**`ToolConnectionMap(BaseModel)`** — a Pydantic model with a single field `stateful_parameters: List[StatefulParameter]`. Created via `ToolConnectionMap.model_validate(response_json)` from the LLM's JSON output.

**`ToolConnectionAnalyzer`** — takes `llm_name: str` and `llm_config: GenerateContentConfig` in its constructor. Its `analyze(tools: List[BaseTool]) -> ToolConnectionMap` method builds tool schema JSON, sends it to the LLM with a structured prompt, and parses the JSON response with `re.sub` stripping markdown fences before `json.loads`. The analysis result is then passed to `ToolSpecMockStrategy` to generate realistic mock responses.

### Key behaviours

| Behaviour | Detail |
|---|---|
| LLM-driven analysis | `analyze()` calls `self._llm.generate_content_async(request)` with `response_mime_type="application/json"` — forces JSON-only output. |
| Tool schema extraction | Calls `tool._get_declaration().model_dump(exclude_none=True)` per tool — any tool without a declaration is silently skipped. |
| Markdown fence stripping | `re.sub(r"^```[a-zA-Z]*\n", "", text)` + `re.sub(r"\n```$", "", text)` — handles LLMs that wrap JSON in code blocks. |
| Graceful degradation | On `json.JSONDecodeError`, logs a warning and returns `ToolConnectionMap(stateful_parameters=[])` — analysis is non-fatal. |
| Creating vs consuming | A "creating" tool generates or modifies a resource; a "consuming" tool only reads it. `ToolSpecMockStrategy` uses this to update `state_store` only after creating tool calls. |
| Feature flag guard | `@experimental(FeatureName.ENVIRONMENT_SIMULATION)` — emits `UserWarning` on import. |

### Example 1 — analyzing a customer support toolset

```python
import asyncio
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.environment_simulation.tool_connection_analyzer import ToolConnectionAnalyzer
from google.genai import types as genai_types


def create_ticket(user_id: str, subject: str, priority: str) -> dict:
    """Create a new support ticket for a user."""
    return {"ticket_id": "TKT-001", "status": "open"}


def get_ticket(ticket_id: str) -> dict:
    """Retrieve details for a specific support ticket."""
    return {"ticket_id": ticket_id, "status": "open", "subject": "..."}


def cancel_ticket(ticket_id: str) -> dict:
    """Cancel an existing support ticket."""
    return {"ticket_id": ticket_id, "status": "cancelled"}


def list_user_tickets(user_id: str) -> list:
    """List all tickets for a given user."""
    return []


tools = [
    FunctionTool(func=create_ticket),
    FunctionTool(func=get_ticket),
    FunctionTool(func=cancel_ticket),
    FunctionTool(func=list_user_tickets),
]

# In production: analyzer calls the LLM to detect that ticket_id is a
# stateful parameter created by create_ticket and consumed by get/cancel.
# analyzer = ToolConnectionAnalyzer(
#     llm_name="gemini-2.5-flash",
#     llm_config=genai_types.GenerateContentConfig(temperature=0),
# )
# connection_map = await analyzer.analyze(tools)
# print(connection_map.stateful_parameters[0].parameter_name)  # ticket_id
# print(connection_map.stateful_parameters[0].creating_tools)  # ['create_ticket']
# print(connection_map.stateful_parameters[0].consuming_tools)  # ['get_ticket', 'cancel_ticket', 'list_user_tickets']

print("ToolConnectionAnalyzer example defined — requires Gemini API key to run")
print("Tools:", [t.name for t in tools])
```

### Example 2 — manually building a ToolConnectionMap for testing

```python
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from google.adk.tools.environment_simulation.tool_connection_map import (
    ToolConnectionMap,
    StatefulParameter,
)

# Build a ToolConnectionMap directly for use with ToolSpecMockStrategy
# without calling the LLM (e.g., in unit tests or when the schema is known).
connection_map = ToolConnectionMap(
    stateful_parameters=[
        StatefulParameter(
            parameter_name="order_id",
            creating_tools=["create_order", "clone_order"],
            consuming_tools=["get_order", "cancel_order", "list_orders"],
        ),
        StatefulParameter(
            parameter_name="user_id",
            creating_tools=["create_user"],
            consuming_tools=["get_user", "list_user_orders", "delete_user"],
        ),
    ]
)

print("Stateful parameters:", len(connection_map.stateful_parameters))
for param in connection_map.stateful_parameters:
    print(f"  {param.parameter_name}:")
    print(f"    creating: {param.creating_tools}")
    print(f"    consuming: {param.consuming_tools}")

# Serialise for inspection.
print("\nJSON dump:")
print(connection_map.model_dump_json(indent=2))
```

### Example 3 — integrating ToolConnectionMap with EnvironmentSimulationConfig

```python
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from google.adk.tools.environment_simulation.tool_connection_map import (
    ToolConnectionMap,
    StatefulParameter,
)
from google.adk.tools.environment_simulation.environment_simulation_config import (
    EnvironmentSimulationConfig,
)
from google.genai import types as genai_types

# Pre-build the connection map to skip the LLM analysis phase.
# This is the recommended approach when tool dependencies are well-understood.
connection_map = ToolConnectionMap(
    stateful_parameters=[
        StatefulParameter(
            parameter_name="reservation_id",
            creating_tools=["make_reservation"],
            consuming_tools=["get_reservation", "cancel_reservation"],
        )
    ]
)

# EnvironmentSimulationConfig accepts the connection_map so ToolSpecMockStrategy
# can use it to maintain state across mocked tool calls.
simulation_config = EnvironmentSimulationConfig(
    llm_name="gemini-2.5-flash",
    llm_config=genai_types.GenerateContentConfig(temperature=0.0),
    tool_connection_map=connection_map,
)

print("Simulation config built with pre-analyzed connection map")
print("Creating tools for reservation_id:", connection_map.stateful_parameters[0].creating_tools)
# make_reservation creates the ID; cancel_reservation and get_reservation consume it.
# ToolSpecMockStrategy will update state_store['reservation_id'] after make_reservation.
```
