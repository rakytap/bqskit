"""This module implements the WhileLoopPass class."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.workflow import Workflow
from bqskit.passes.control.predicate import PassPredicate

if TYPE_CHECKING:
    from bqskit.ir.circuit import Circuit
    from bqskit.compiler.passdata import PassData
    from bqskit.compiler.workflow import WorkflowLike

_logger = logging.getLogger(__name__)


class WhileLoopPass(BasePass):
    """
    The WhileLoopPass class.

    This is a control pass that conditionally executes a workflow in a loop.
    """

    def __init__(
        self,
        condition: PassPredicate,
        loop_body: WorkflowLike,
    ) -> None:
        """
        Construct a WhileLoopPass.

        Args:
            condition (PassPredicate): The condition checked.

            loop_body (WorkflowLike): The pass or passes to execute
                while `condition` is true.
        """
        if not isinstance(condition, PassPredicate):
            raise TypeError(f'Expected PassPredicate, got {type(condition)}.')

        self.condition = condition
        self.workflow = Workflow(loop_body)

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        while self.condition(circuit, data):
            _logger.debug('Loop body executing...')
            await self.workflow.run(circuit, data)
