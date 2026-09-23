"""
Agent: a single LLM-driven worker that talks to a local Ollama server.

Each call returns an AgentResult capturing everything needed later for
failure attribution: prompt, response, latency, retry count, and a
status/error classification.
"""
import asyncio
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import aiohttp

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"


@dataclass
class AgentResult:
    agent_id: str
    role: str
    task_id: str
    round_id: int
    prompt: str
    response: Optional[str]
    status: str  # "success" | "timeout" | "error" | "malformed"
    latency_s: float
    attempt: int
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self):
        return asdict(self)


class Agent:
    def __init__(
        self,
        agent_id: str,
        role: str,
        system_prompt: str,
        model: str = "qwen3:8b",
        timeout: float = 60.0,
        max_retries: int = 1,
        validate_fn=None,
    ):
        self.agent_id = agent_id
        self.role = role
        self.system_prompt = system_prompt
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        # optional callable(response_str) -> bool, to flag "malformed" outputs
        self.validate_fn = validate_fn

    async def run(
        self,
        session: aiohttp.ClientSession,
        task_id: str,
        round_id: int,
        user_prompt: str,
    ) -> AgentResult:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_error = None
        total_attempts = self.max_retries + 1

        for attempt in range(1, total_attempts + 1):
            start = time.time()
            try:
                async with session.post(
                    OLLAMA_CHAT_URL,
                    json={
                        "model": self.model,
                        "messages": messages,
                        "stream": False,
                    },
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as resp:
                    data = await resp.json()
                    latency = time.time() - start
                    content = (data.get("message") or {}).get("content")

                    if not content:
                        last_error = "empty_response"
                        if attempt < total_attempts:
                            continue
                        return AgentResult(
                            self.agent_id, self.role, task_id, round_id,
                            user_prompt, None, "error", latency, attempt,
                            error=last_error,
                        )

                    if self.validate_fn and not self.validate_fn(content):
                        if attempt < total_attempts:
                            last_error = "failed_validation"
                            continue
                        return AgentResult(
                            self.agent_id, self.role, task_id, round_id,
                            user_prompt, content, "malformed", latency, attempt,
                            error="failed_validation",
                        )

                    return AgentResult(
                        self.agent_id, self.role, task_id, round_id,
                        user_prompt, content, "success", latency, attempt,
                    )

            except asyncio.TimeoutError:
                latency = time.time() - start
                last_error = "timeout"
                if attempt >= total_attempts:
                    return AgentResult(
                        self.agent_id, self.role, task_id, round_id,
                        user_prompt, None, "timeout", latency, attempt,
                        error="timeout",
                    )
            except Exception as e:
                latency = time.time() - start
                last_error = str(e)
                if attempt >= total_attempts:
                    return AgentResult(
                        self.agent_id, self.role, task_id, round_id,
                        user_prompt, None, "error", latency, attempt,
                        error=str(e),
                    )

        return AgentResult(
            self.agent_id, self.role, task_id, round_id,
            user_prompt, None, "error", 0.0, total_attempts,
            error=last_error,
        )
