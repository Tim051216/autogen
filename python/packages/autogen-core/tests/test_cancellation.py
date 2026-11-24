import asyncio
import threading
from dataclasses import dataclass

import pytest
from autogen_core import (
    AgentId,
    AgentInstantiationContext,
    CancellationToken,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    message_handler,
)


@dataclass
class MessageType: ...


# Note for future reader:
# To do cancellation, only the token should be interacted with as a user
# If you cancel a future, it may not work as you expect.


class LongRunningAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("A long running agent")
        self.called = False
        self.cancelled = False

    @message_handler
    async def on_new_message(self, message: MessageType, ctx: MessageContext) -> MessageType:
        self.called = True
        sleep = asyncio.ensure_future(asyncio.sleep(100))
        ctx.cancellation_token.link_future(sleep)
        try:
            await sleep
            return MessageType()
        except asyncio.CancelledError:
            self.cancelled = True
            raise


class NestingLongRunningAgent(RoutedAgent):
    def __init__(self, nested_agent: AgentId) -> None:
        super().__init__("A nesting long running agent")
        self.called = False
        self.cancelled = False
        self._nested_agent = nested_agent

    @message_handler
    async def on_new_message(self, message: MessageType, ctx: MessageContext) -> MessageType:
        self.called = True
        response = self.send_message(message, self._nested_agent, cancellation_token=ctx.cancellation_token)
        try:
            val = await response
            assert isinstance(val, MessageType)
            return val
        except asyncio.CancelledError:
            self.cancelled = True
            raise


@pytest.mark.asyncio
async def test_cancellation_with_token() -> None:
    runtime = SingleThreadedAgentRuntime()

    await LongRunningAgent.register(runtime, "long_running", LongRunningAgent)
    agent_id = AgentId("long_running", key="default")
    token = CancellationToken()
    response = asyncio.create_task(runtime.send_message(MessageType(), recipient=agent_id, cancellation_token=token))
    assert not response.done()

    while runtime.unprocessed_messages_count == 0:
        await asyncio.sleep(0.01)

    await runtime._process_next()  # type: ignore

    token.cancel()

    with pytest.raises(asyncio.CancelledError):
        await response

    assert response.done()
    long_running_agent = await runtime.try_get_underlying_agent_instance(agent_id, type=LongRunningAgent)
    assert long_running_agent.called
    assert long_running_agent.cancelled


@pytest.mark.asyncio
async def test_nested_cancellation_only_outer_called() -> None:
    runtime = SingleThreadedAgentRuntime()

    await LongRunningAgent.register(runtime, "long_running", LongRunningAgent)
    await NestingLongRunningAgent.register(
        runtime,
        "nested",
        lambda: NestingLongRunningAgent(AgentId("long_running", key=AgentInstantiationContext.current_agent_id().key)),
    )

    long_running_id = AgentId("long_running", key="default")
    nested_id = AgentId("nested", key="default")
    token = CancellationToken()
    response = asyncio.create_task(runtime.send_message(MessageType(), nested_id, cancellation_token=token))
    assert not response.done()

    while runtime.unprocessed_messages_count == 0:
        await asyncio.sleep(0.01)

    await runtime._process_next()  # type: ignore
    token.cancel()

    with pytest.raises(asyncio.CancelledError):
        await response

    assert response.done()
    nested_agent = await runtime.try_get_underlying_agent_instance(nested_id, type=NestingLongRunningAgent)
    assert nested_agent.called
    assert nested_agent.cancelled
    long_running_agent = await runtime.try_get_underlying_agent_instance(long_running_id, type=LongRunningAgent)
    assert long_running_agent.called is False
    assert long_running_agent.cancelled is False


@pytest.mark.asyncio
async def test_nested_cancellation_inner_called() -> None:
    runtime = SingleThreadedAgentRuntime()

    await LongRunningAgent.register(runtime, "long_running", LongRunningAgent)
    await NestingLongRunningAgent.register(
        runtime,
        "nested",
        lambda: NestingLongRunningAgent(AgentId("long_running", key=AgentInstantiationContext.current_agent_id().key)),
    )

    long_running_id = AgentId("long_running", key="default")
    nested_id = AgentId("nested", key="default")

    token = CancellationToken()
    response = asyncio.create_task(runtime.send_message(MessageType(), nested_id, cancellation_token=token))
    assert not response.done()

    while runtime.unprocessed_messages_count == 0:
        await asyncio.sleep(0.01)

    await runtime._process_next()  # type: ignore
    # allow the inner agent to process
    await runtime._process_next()  # type: ignore
    token.cancel()

    with pytest.raises(asyncio.CancelledError):
        await response

    assert response.done()
    nested_agent = await runtime.try_get_underlying_agent_instance(nested_id, type=NestingLongRunningAgent)
    assert nested_agent.called
    assert nested_agent.cancelled
    long_running_agent = await runtime.try_get_underlying_agent_instance(long_running_id, type=LongRunningAgent)
    assert long_running_agent.called
    assert long_running_agent.cancelled


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_before_link", [False, True])
async def test_link_future_cancellation_is_thread_safe(cancel_before_link: bool) -> None:
    loop = asyncio.get_running_loop()
    was_debug = loop.get_debug()
    loop.set_debug(True)

    try:
        token = CancellationToken()
        if cancel_before_link:
            token.cancel()

        future = loop.create_future()
        token.link_future(future)
        callback_ran = asyncio.Event()
        callback_thread: list[int] = []

        def on_done(_: asyncio.Future[None]) -> None:
            callback_thread.append(threading.get_ident())
            callback_ran.set()

        future.add_done_callback(on_done)

        async def wait_for_future() -> None:
            await future

        waiter = asyncio.create_task(wait_for_future())
        thread_errors: list[BaseException] = []

        if not cancel_before_link:

            def cancel_from_worker() -> None:
                try:
                    token.cancel()
                except BaseException as exc:
                    thread_errors.append(exc)

            worker = threading.Thread(target=cancel_from_worker)
            worker.start()
            await asyncio.to_thread(worker.join)

        with pytest.raises(asyncio.CancelledError):
            await waiter
        await asyncio.wait_for(callback_ran.wait(), timeout=1)

        assert thread_errors == []
        assert callback_thread == [threading.get_ident()]
    finally:
        loop.set_debug(was_debug)


def test_link_future_after_loop_closes_does_not_raise() -> None:
    loop = asyncio.new_event_loop()
    future = loop.create_future()
    future.add_done_callback(lambda _: None)
    token = CancellationToken()
    token.link_future(future)

    loop.close()

    token.cancel()

    assert not future.cancelled()
