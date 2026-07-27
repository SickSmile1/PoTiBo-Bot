"""
End-to-end smoke test for DocBot.

A real Matrix user (created by main.py at startup) sends '!ask' queries
to the running docbot and asserts meaningful replies arrive.

Prerequisites
-------------
- Synapse is reachable at MATRIX_HOMESERVER.
- The docbot container is running with a built index.
- TEST_USER_USERNAME and TEST_USER_PASSWORD are set in the environment
  (main.py registers this user via the shared-secret API on every startup).

Run from inside the docbot container:
    pytest /docbot/app/tests/test_smoke.py -v

The tests are skipped automatically when Synapse is not reachable or the
test-user credentials are not configured.
"""

import time

import httpx
import pytest
from nio import (
    AsyncClient,
    LoginResponse,
    RoomCreateResponse,
    RoomInviteResponse,
    RoomMessageText,
    RoomSendResponse,
    SyncResponse,
)

from config import Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _skip_if_not_configured(config: Config) -> None:
    if not config.TEST_USER_USERNAME or not config.TEST_USER_PASSWORD:
        pytest.skip("TEST_USER_USERNAME / TEST_USER_PASSWORD not configured")


async def _synapse_reachable(homeserver: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=5) as h:
            r = await h.get(f"{homeserver}/_matrix/client/versions")
            return r.status_code == 200
    except Exception:
        return False


async def _poll_for_bot_reply(client, room_id, bot_id, since_ts, timeout=30):
    """Poll sync until a reply from the bot arrives, or return None."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        sync_resp = await client.sync(timeout=3_000)
        if not isinstance(sync_resp, SyncResponse):
            continue
        room_sync = sync_resp.rooms.join.get(room_id)
        if not room_sync:
            continue
        for event in room_sync.timeline.events:
            if (
                isinstance(event, RoomMessageText)
                and event.sender == bot_id
                and event.server_timestamp > since_ts
            ):
                return event
    return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def matrix_session():
    """Log in as the pre-registered smoke-test user.

    Skips the whole module if Synapse is unreachable or credentials are missing.
    """
    config = Config()
    _skip_if_not_configured(config)

    if not await _synapse_reachable(config.MATRIX_HOMESERVER):
        pytest.skip(f"Synapse not reachable at {config.MATRIX_HOMESERVER}")

    user_id = f"@{config.TEST_USER_USERNAME}:{config.MATRIX_SERVER_NAME}"
    client = AsyncClient(config.MATRIX_HOMESERVER, user_id)

    resp = await client.login(config.TEST_USER_PASSWORD, device_name="pytest-smoke")
    if not isinstance(resp, LoginResponse):
        await client.close()
        pytest.fail(
            f"Smoke-test user login failed: {resp}\n"
            "The user should have been registered by main.py at startup. "
            "Check TEST_USER_USERNAME / TEST_USER_PASSWORD and "
            "MATRIX_REGISTRATION_SHARED_SECRET."
        )

    await client.sync(timeout=10_000)
    yield client, config
    await client.close()


@pytest.fixture
async def bot_room(matrix_session):
    """Create a fresh room, invite the docbot, wait for it to join, then leave."""
    client, config = matrix_session
    bot_id = f"@{config.MATRIX_BOT_USERNAME}:{config.MATRIX_SERVER_NAME}"

    create_resp = await client.room_create(name="docbot-smoke-test")
    assert isinstance(create_resp, RoomCreateResponse), (
        f"Could not create smoke-test room: {create_resp}"
    )
    room_id = create_resp.room_id

    invite_resp = await client.room_invite(room_id, bot_id)
    assert isinstance(invite_resp, RoomInviteResponse), (
        f"Could not invite bot {bot_id} to room {room_id}: {invite_resp}"
    )

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        await client.sync(timeout=3_000)
        room = client.rooms.get(room_id)
        if room and bot_id in room.users:
            break
    else:
        pytest.fail(
            f"Bot {bot_id} did not join room {room_id} within 30 s. "
            "Is the docbot container running?"
        )

    yield room_id, bot_id

    await client.room_leave(room_id)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("query", "min_length"),
    [
        ("!ask how does user login work", 50),
        ("!ask how to configure the bot", 50),
        ("!ask password reset flow", 50),
    ],
)
async def test_ask_query_receives_relevant_reply(matrix_session, bot_room, query, min_length):
    """Send a documentation query and verify a substantive answer is returned."""
    client, _config = matrix_session
    room_id, bot_id = bot_room

    send_ts = int(time.time() * 1000)

    send_resp = await client.room_send(
        room_id,
        "m.room.message",
        {"msgtype": "m.text", "body": query},
    )
    assert isinstance(send_resp, RoomSendResponse), f"room_send failed: {send_resp}"

    reply = await _poll_for_bot_reply(client, room_id, bot_id, send_ts)
    assert reply is not None, (
        f"Bot did not reply within 30 s for {query!r} (room {room_id}). "
        "Check that the docbot index is built and the bot is online."
    )

    assert "couldn't find" not in reply.body.lower(), (
        f"Bot returned the no-results fallback for {query!r} — "
        f"index may not be built yet:\n{reply.body}"
    )
    assert len(reply.body) > min_length, (
        f"Reply for {query!r} is suspiciously short:\n{reply.body!r}"
    )
