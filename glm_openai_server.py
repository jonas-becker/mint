#!/usr/bin/env python3
"""Minimal OpenAI-compatible chat endpoint backed by Transformers.

This is intended for models unsupported by vLLM (e.g. GLM-4.7-Flash) while
keeping the same `/v1` API surface MALLM expects.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[Message]
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    stream: bool = False
    stop: list[str] | None = None


class GLMOpenAIServer:
    def __init__(self, model_name: str, trust_remote_code: bool, local_files_only: bool) -> None:
        self.model_name = model_name
        self.trust_remote_code = trust_remote_code
        self.local_files_only = local_files_only
        self._lock = threading.Lock()
        self.streamer_timeout_seconds = float(
            os.environ.get("GLM_SHIM_STREAMER_TIMEOUT_SECONDS", "120")
        )

        tokenizer_kwargs = {
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
        }
        cache_dir = os.environ.get("HF_HUB_CACHE") or os.environ.get("TRANSFORMERS_CACHE")
        if cache_dir:
            tokenizer_kwargs["cache_dir"] = cache_dir
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        except Exception:
            # Some shared caches contain tokenizer.json variants that fail to load
            # in fast mode; slow tokenizer is more tolerant for these checkpoints.
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=False,
                **tokenizer_kwargs,
            )

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
        }
        if cache_dir:
            model_kwargs["cache_dir"] = cache_dir
        if torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.bfloat16
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.model.eval()

    @contextmanager
    def generation_slot(self, timeout_seconds: float | None = None):
        # Serialize generation to protect shared model state.
        acquired = self._lock.acquire(
            timeout=timeout_seconds if timeout_seconds is not None else -1
        )
        if not acquired:
            raise TimeoutError("Timed out waiting for a generation slot.")
        try:
            yield
        finally:
            self._lock.release()

    def _build_prompt(self, messages: list[Message]) -> str:
        raw_messages = [{"role": m.role, "content": m.content} for m in messages]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                raw_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        parts: list[str] = []
        for m in raw_messages:
            parts.append(f"{m['role']}: {m['content']}")
        parts.append("assistant:")
        return "\n".join(parts)

    def _generation_kwargs(self, req: ChatCompletionRequest, streamer: TextIteratorStreamer | None = None) -> dict[str, Any]:
        prompt = self._build_prompt(req.messages)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            # For most HF CausalLM models this is sufficient with device_map="auto".
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        do_sample = req.temperature is not None and req.temperature > 1e-6
        kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": max(1, int(req.max_tokens)),
            "do_sample": do_sample,
            "top_p": float(req.top_p) if req.top_p else 1.0,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            kwargs["temperature"] = float(req.temperature)
        if streamer is not None:
            kwargs["streamer"] = streamer
        return kwargs

    def generate_text(self, req: ChatCompletionRequest) -> str:
        kwargs = self._generation_kwargs(req, streamer=None)
        with torch.inference_mode():
            out = self.model.generate(**kwargs)
        prompt_len = kwargs["input_ids"].shape[-1]
        gen_ids = out[0][prompt_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return text

    def stream_text(self, req: ChatCompletionRequest):
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=self.streamer_timeout_seconds,
        )
        kwargs = self._generation_kwargs(req, streamer=streamer)
        worker_error: list[Exception] = []

        def _run_generate() -> None:
            try:
                with torch.inference_mode():
                    self.model.generate(**kwargs)
            except Exception as exc:
                worker_error.append(exc)
                # Unblock streamer iteration if generation crashes.
                streamer.end()

        worker = threading.Thread(target=_run_generate, daemon=True)
        worker.start()

        full_text = ""
        emitted = 0
        stop_sequences = req.stop or []
        stop_hit = False

        for piece in streamer:
            full_text += piece
            cutoff = len(full_text)
            for marker in stop_sequences:
                idx = full_text.find(marker)
                if idx != -1:
                    cutoff = min(cutoff, idx)
                    stop_hit = True
            if cutoff > emitted:
                yield full_text[emitted:cutoff]
                emitted = cutoff
            if stop_hit:
                break
        if worker_error:
            raise worker_error[0]


def create_app(server: GLMOpenAIServer) -> FastAPI:
    app = FastAPI()
    max_queue_wait_seconds = float(
        os.environ.get("GLM_SHIM_MAX_QUEUE_WAIT_SECONDS", "600")
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": server.model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest):
        requested_model = req.model or server.model_name
        if requested_model != server.model_name:
            raise HTTPException(
                status_code=400,
                detail=f"Loaded model is '{server.model_name}', requested '{requested_model}'.",
            )

        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        if req.stream:
            try:
                slot_context = server.generation_slot(timeout_seconds=max_queue_wait_seconds)
                slot_context.__enter__()
            except TimeoutError as exc:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Model server queue wait exceeded "
                        f"{max_queue_wait_seconds:.1f}s. "
                        "Please retry with lower request concurrency."
                    ),
                ) from exc

            def event_stream():
                first = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": server.model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": ""},
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(first)}\n\n"
                try:
                    for token_text in server.stream_text(req):
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": server.model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": token_text},
                                    "logprobs": None,
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                finally:
                    slot_context.__exit__(None, None, None)

                last = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": server.model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "logprobs": None,
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield f"data: {json.dumps(last)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        try:
            with server.generation_slot(timeout_seconds=max_queue_wait_seconds):
                text = server.generate_text(req)
        except TimeoutError as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Model server queue wait exceeded "
                    f"{max_queue_wait_seconds:.1f}s. "
                    "Please retry with lower request concurrency."
                ),
            ) from exc
        return JSONResponse(
            {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": server.model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "logprobs": None,
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a local OpenAI-compatible endpoint via Transformers.")
    parser.add_argument("--model", required=True, help="Model id or local model path.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    server = GLMOpenAIServer(
        model_name=args.model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    app = create_app(server)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
