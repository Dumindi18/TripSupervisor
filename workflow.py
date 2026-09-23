"""
WorkflowEngine: executes a DAG of WorkflowNodes.

A node becomes eligible to run once all of its `depends_on` nodes have
completed. All currently-eligible nodes run CONCURRENTLY. Chains of
dependencies give sequential sections; siblings at the same dependency
level give parallel sections -- both from the same scheduler.
"""
import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import aiohttp

from agent import Agent, AgentResult
from blackboard import Blackboard


@dataclass
class WorkflowNode:
    node_id: str
    agent: Agent
    depends_on: List[str] = field(default_factory=list)
    # async or sync callable: (completed_results, blackboard) -> prompt string
    prompt_builder: Callable[[Dict[str, AgentResult], Blackboard], str] = None
    # optional async or sync callable: (this node's AgentResult, blackboard) -> None
    on_success: Optional[Callable[[AgentResult, Blackboard], None]] = None


class WorkflowEngine:
    def __init__(
        self,
        nodes: List[WorkflowNode],
        blackboard: Blackboard,
        max_concurrency: int = 4,
        log_path: str = "logs/workflow.jsonl",
    ):
        self.nodes: Dict[str, WorkflowNode] = {n.node_id: n for n in nodes}
        self.blackboard = blackboard
        self.sem = asyncio.Semaphore(max_concurrency)
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._validate_dag()

    def _validate_dag(self):
        ids = set(self.nodes)
        for n in self.nodes.values():
            for dep in n.depends_on:
                if dep not in ids:
                    raise ValueError(f"Node '{n.node_id}' depends on unknown node '{dep}'")

        visited, visiting = set(), set()

        def visit(nid):
            if nid in visited:
                return
            if nid in visiting:
                raise ValueError(f"Cycle detected in workflow at node '{nid}'")
            visiting.add(nid)
            for dep in self.nodes[nid].depends_on:
                visit(dep)
            visiting.discard(nid)
            visited.add(nid)

        for nid in self.nodes:
            visit(nid)

    def _log(self, result: AgentResult):
        with open(self.log_path, "a") as f:
            f.write(json.dumps(result.to_dict()) + "\n")

    async def _run_node(
        self,
        session: aiohttp.ClientSession,
        task_id: str,
        node: WorkflowNode,
        completed: Dict[str, AgentResult],
    ) -> AgentResult:
        async with self.sem:
            prompt = ""
            if node.prompt_builder:
                maybe = node.prompt_builder(completed, self.blackboard)
                prompt = await maybe if asyncio.iscoroutine(maybe) else maybe

            result = await node.agent.run(session, task_id, round_id=0, user_prompt=prompt)
            self._log(result)

            if result.status == "success" and node.on_success:
                maybe = node.on_success(result, self.blackboard)
                if asyncio.iscoroutine(maybe):
                    await maybe

            return result

    async def run(self, task_id: str) -> Dict[str, AgentResult]:
        completed: Dict[str, AgentResult] = {}
        running: Dict[asyncio.Task, str] = {}
        remaining = set(self.nodes.keys())

        async with aiohttp.ClientSession() as session:
            while remaining or running:
                ready = [
                    nid for nid in remaining
                    if all(dep in completed for dep in self.nodes[nid].depends_on)
                ]
                for nid in ready:
                    remaining.discard(nid)
                    task = asyncio.create_task(
                        self._run_node(session, task_id, self.nodes[nid], completed)
                    )
                    running[task] = nid

                if not running:
                    break

                done, _ = await asyncio.wait(running.keys(), return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    nid = running.pop(task)
                    completed[nid] = task.result()

        return completed
