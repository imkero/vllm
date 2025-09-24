"""A server wrapper for vLLM's LLMEngine."""
import enum
from concurrent.futures import Future
from typing import Any, Dict, Optional

from vllm import SamplingParams


class VllmReqStatus(enum.Enum):
    """Status of a VllmReq."""
    RUNNING = enum.auto()
    SUCCESS = enum.auto()
    FAILED = enum.auto()


class VllmReq:
    """Represents a request to the VllmServer."""

    def __init__(self, prompt: Dict[str, Any], sampling_params: SamplingParams):
        self.prompt = prompt
        self.sampling_params = sampling_params
        self.status: VllmReqStatus = VllmReqStatus.RUNNING
        self.output: Optional[str] = None
        self.error: Optional[Any] = None
        self.future: Future = Future()
        self.request_id: Optional[str] = None

    def result(self):
        """Waits for the request to complete and returns the result."""
        return self.future.result()

    def __repr__(self) -> str:
        return (f"VllmReq(prompt={self.prompt}, "
                f"status={self.status}, "
                f"output={self.output}, "
                f"error={self.error})")


import queue
import threading
import time
import uuid

from vllm import EngineArgs, LLMEngine, RequestOutput


class VllmServer:
    def __init__(self, **kwargs):
        engine_args = EngineArgs(**kwargs)
        self.engine = LLMEngine.from_engine_args(engine_args)
        self.new_reqs: queue.Queue[VllmReq] = queue.Queue()
        self.abort_reqs: queue.Queue[str] = queue.Queue()
        self.active_reqs: Dict[str, VllmReq] = {}
        self._shutdown_event = threading.Event()
        self._main_loop_thread: Optional[threading.Thread] = None
        self._engine_ready_event = threading.Event()

    def launch(self):
        self._main_loop_thread = threading.Thread(target=self._main_loop)
        self._main_loop_thread.start()
        self._engine_ready_event.wait()

    def _main_loop(self):
        self._engine_ready_event.set()
        while not self._shutdown_event.is_set():
            # Handle new requests
            try:
                while not self.new_reqs.empty():
                    req = self.new_reqs.get_nowait()
                    self.active_reqs[req.request_id] = req
                    self.engine.add_request(req.request_id, **req.prompt,
                                            sampling_params=req.sampling_params)
            except queue.Empty:
                pass

            # Handle abort requests
            try:
                while not self.abort_reqs.empty():
                    req_id = self.abort_reqs.get_nowait()
                    self.engine.abort_request(req_id)
                    # No need to remove from active_reqs, will be handled in step output
            except queue.Empty:
                pass

            # Run engine step
            request_outputs: List[RequestOutput] = self.engine.step()

            # Update request status
            for request_output in request_outputs:
                req_id = request_output.request_id
                if req_id in self.active_reqs:
                    if request_output.finished:
                        req = self.active_reqs.pop(req_id)
                        if request_output.status_code is not None:
                            req.status = VllmReqStatus.FAILED
                            req.error = request_output.status_message
                            req.future.set_exception(
                                RuntimeError(request_output.status_message))
                        else:
                            req.status = VllmReqStatus.SUCCESS
                            req.output = request_output
                            req.future.set_result(request_output)

            if not self.engine.has_unfinished_requests():
                time.sleep(0.01)

    def shutdown(self):
        self._shutdown_event.set()
        if self._main_loop_thread:
            self._main_loop_thread.join()

    def add_request(self, prompt: dict[str, any],
                    sampling_params: SamplingParams) -> VllmReq:
        req = VllmReq(prompt, sampling_params)
        req.request_id = str(uuid.uuid4())
        self.new_reqs.put(req)
        return req
