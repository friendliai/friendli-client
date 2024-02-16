# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

"""CLI command to sign in Friendli."""

from __future__ import annotations

import threading
import time
import webbrowser
from contextlib import contextmanager
from typing import Iterator, Tuple

import typer
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse

from friendli.client.http.login import LoginClient
from friendli.di.injector import get_injector
from friendli.utils.url import URLProvider

server_app = FastAPI()


@contextmanager
def run_server(port: int) -> Iterator[None]:
    """Run temporary local server to handle SSO redirection."""
    config = uvicorn.Config(
        app=server_app, host="127.0.0.1", port=port, log_level="error"
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run)
    thread.start()
    try:
        yield
    finally:
        server.should_exit = True
        thread.join()


def oauth2_login() -> Tuple[str, str]:
    """Login with SSO."""
    injector = get_injector()
    url_provider = injector.get(URLProvider)
    authorization_url = url_provider.get_suite_uri("/login/cli")

    access_token = None
    refresh_token = None

    @server_app.get("/sso")
    async def callback(request: Request) -> HTMLResponse:
        nonlocal access_token
        nonlocal refresh_token

        access_token = request.query_params.get("access_token")
        refresh_token = request.query_params.get("refresh_token")

        if not access_token:
            raise HTTPException(
                status_code=400, detail="Access token not found in cookies"
            )

        success_page = r"""
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SSO Login Success</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      display: flex;
      justify-content: center;
      align-items: center;
      height: 100vh;
      background-color: #f0f0f0;
    }
    .message-box {
      text-align: center;
      padding: 40px;
      border: 1px solid #d0d0d0;
      background-color: #ffffff;
      box-shadow: 0 0 10px rgba(0,0,0,0.1);
    }
  </style>
</head>
<body>
  <div class="message-box">
    <h1>Authentication was successful</h1>
    <p>You can now close this window and return to CLI.</p>
    <p>Redirecting to <a href="https://docs.periflow.ai/">Friendli Documentation</a> in <span id="countdown">10</span> seconds.</p>
  </div>
  <script>
    var timeLeft = 10;
    var countdownElement = document.getElementById("countdown");

    var timerId = setInterval(function() {
      timeLeft--;
      countdownElement.innerHTML = timeLeft;
      if (timeLeft <= 0) {
        clearInterval(timerId);
        window.location.href = "https://docs.periflow.ai/";
      }
    }, 1000);
  </script>
</body>
</html>
"""
        return HTMLResponse(content=success_page, status_code=200)

    typer.secho(
        f"Opening browser for authentication: {authorization_url}", fg=typer.colors.BLUE
    )

    webbrowser.open(authorization_url)

    with run_server(33333):
        while access_token is None or refresh_token is None:
            time.sleep(1)

    return access_token, refresh_token


def pwd_login(email: str, pwd: str) -> Tuple[str, str]:
    """Login with email and password."""
    client = LoginClient()
    return client.login(email, pwd)
