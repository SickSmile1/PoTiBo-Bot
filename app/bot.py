import hashlib
import hmac
import logging
import re
import time

import httpx
from nio import AsyncClient, InviteMemberEvent, RoomMessageText, ReactionEvent
from nio.api import RoomPreset, RoomVisibility
from embedder import Embedder
from searcher import Searcher

logger = logging.getLogger(__name__)


def _compute_mac(shared_secret: str, nonce: str, username: str, password: str) -> str:
    """Compute HMAC-SHA1 for Synapse shared-secret registration."""
    mac = hmac.new(shared_secret.encode(), digestmod=hashlib.sha1)
    mac.update(nonce.encode())
    mac.update(b"\x00")
    mac.update(username.encode())
    mac.update(b"\x00")
    mac.update(password.encode())
    mac.update(b"\x00")
    mac.update(b"notadmin")
    return mac.hexdigest()


async def register_matrix_user(
    homeserver: str, shared_secret: str, username: str, password: str
) -> bool:
    """Register a user on Synapse via the shared-secret API.

    Safe to call when the user already exists — returns True in that case.
    Returns False if the secret is missing or the request fails.
    """
    if not shared_secret:
        return False
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            nonce_resp = await client.get(f"{homeserver}/_synapse/admin/v1/register")
            nonce_resp.raise_for_status()
            nonce = nonce_resp.json().get("nonce")
            if not nonce:
                print(f"No nonce returned when registering @{username}")
                return False

            mac = _compute_mac(shared_secret, nonce, username, password)
            reg_resp = await client.post(
                f"{homeserver}/_synapse/admin/v1/register",
                json={
                    "nonce": nonce,
                    "username": username,
                    "password": password,
                    "admin": False,
                    "mac": mac,
                },
            )
            if reg_resp.status_code in (400, 409):
                body = reg_resp.json()
                if body.get("errcode") in ("M_USER_IN_USE", "M_EXCLUSIVE"):
                    print(f"User @{username} already registered")
                    return True
            reg_resp.raise_for_status()
            print(f"Registered user @{username}")
            return True
    except Exception as e:
        print(f"Failed to register @{username}: {e}")
        return False

_WELCOME_MESSAGE = """\
Welcome! I'm your personal documentation assistant for this project.

Commands:
  !ask <query>               — semantic search across codebase, libraries & reference docs
  !ask --from <tag> <query>  — filter by source type: code | library | reference | web

  Example: !ask --from code How does JWT auth work?

Mark a message with a reaction emoji to forward it to a dedicated room.
Configured reactions: {react_summary}

Returns up to 3 ranked snippets with module:symbol, kind, source path, score, and a 300-char excerpt.

Powered by BGE embeddings via ONNX Runtime for fast local inference, and matrix-nio for Matrix client integration.\
"""


class DocBot:
    def __init__(self, config, embedder: Embedder, searcher: Searcher, chunks: int = 0):
        """Store the config and create a Matrix AsyncClient for the bot user.

        Args:
            config: Application configuration (homeserver, server name, password, etc.).
            embedder: Embedder instance for encoding query text.
            searcher: Searcher instance for cosine-similarity lookup.
            chunks: Number of indexed chunks (for heartbeat reporting).
        """
        self.config = config
        self.embedder = embedder
        self.searcher = searcher
        self.chunks = chunks
        self._room_cache: dict[str, str] = {}  # room_name → room_id
        user_id = f"@{config.MATRIX_BOT_USERNAME}:{config.MATRIX_SERVER_NAME}"
        self.client = AsyncClient(config.MATRIX_HOMESERVER, user_id)

    async def _try_register_user(self):
        """Register the bot Matrix user via the Synapse shared-secret API."""
        if not self.config.MATRIX_REGISTRATION_SHARED_SECRET:
            print("MATRIX_REGISTRATION_SHARED_SECRET not set — skipping user registration")
            return False
        return await register_matrix_user(
            self.config.MATRIX_HOMESERVER,
            self.config.MATRIX_REGISTRATION_SHARED_SECRET,
            self.config.MATRIX_BOT_USERNAME,
            self.config.DOCBOT_PASSWORD,
        )

    async def send_heartbeat(self, status="starting", error=None):
        """Report docbot status to the API backend (fire-and-forget)."""
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                await client.post(
                    "http://api:80/admin/bot/docbot/heartbeat",
                    json={
                        "status": status,
                        "error": error,
                        "chunks": self.chunks,
                    },
                    headers={"X-Heartbeat-Token": self.config.DOCBOT_HEARTBEAT_TOKEN},
                )
        except Exception:
            pass

    async def start(self):
        """Login to Matrix, register event handlers, and listen forever."""
        try:
            await self._try_register_user()
            await self.client.login(self.config.DOCBOT_PASSWORD, device_name="docbot")

            self._started_at_ms = int(time.time() * 1000)
            await self.client.sync(timeout=10000)
            await self._accept_pending_invites()

            self.client.add_event_callback(self._on_invite, InviteMemberEvent)
            self.client.add_event_callback(self._on_message, RoomMessageText)
            self.client.add_event_callback(self._reaction_event_handler, ReactionEvent)
            await self.send_heartbeat("online")

            react_routes = self.config.REACT_ROUTES
            react_summary = ", ".join(f"{e}→{r}" for e, r in react_routes.items()) if react_routes else "none"
            welcome = _WELCOME_MESSAGE.format(react_summary=react_summary)
            self._welcome_message = welcome

            print(f"Bot online as @{self.config.MATRIX_BOT_USERNAME}:{self.config.MATRIX_SERVER_NAME}")
            await self.client.sync_forever(timeout=30000)
        except Exception as e:
            await self.send_heartbeat("error", str(e))
            raise

    async def _accept_pending_invites(self):
        """Join all rooms the bot has already been invited to."""
        for room_id, room in self.client.invited_rooms.items():
            try:
                await self.client.join(room_id)
                await self.client.room_send(
                    room_id,
                    "m.room.message",
                    {"msgtype": "m.text", "body": self._welcome_message},
                )
                print(f"Accepted pending invite and joined {room_id}")
            except Exception as e:
                print(f"Could not join invited room {room_id}: {e}")

    async def _on_invite(self, room, event: InviteMemberEvent):
        """Auto-join any room the bot is invited to, then send the welcome message."""
        if event.membership != "invite":
            return
        if event.state_key != self.client.user_id:
            return
        try:
            await self.client.join(room.room_id)
            await self.client.room_send(
                room.room_id,
                "m.room.message",
                {"msgtype": "m.text", "body": self._welcome_message},
            )
            print(f"Joined and welcomed room {room.room_id}")
        except Exception as e:
            print(f"Failed to join room {room.room_id}: {e}")

    async def _on_message(self, room, event):
        """Handle incoming Matrix messages; respond to !ask commands."""
        if event.sender == self.client.user_id:
            return
        if hasattr(self, "_started_at_ms") and event.server_timestamp < self._started_at_ms:
            return
        body = event.body.strip()
        if body.startswith("!ask "):
            query = body[5:]
            await self._handle_ask(room.room_id, event.event_id, query)

    async def _handle_ask(self, room_id, event_id, raw_query):
        """Parse !ask, embed the query, search the index, and reply in-thread.

        Supports optional ``--from <tag>`` to filter by source_type.
        Appends ``Source: <path>`` to each result for transparency.
        """
        source_tag = None
        m = re.match(r"--from\s+(\S+)\s*(.*)", raw_query)
        if m:
            source_tag = m.group(1)
            raw_query = m.group(2)

        if not raw_query.strip():
            text = "Please provide a query after !ask.  e.g. `!ask how does auth work`"
            await self.client.room_send(room_id, "m.room.message", {
                "msgtype": "m.text", "body": text,
                "m.relates_to": {"rel_type": "m.thread", "event_id": event_id},
            })
            return

        query_emb = self.embedder.embed([raw_query], is_query=True)
        results = self.searcher.search(
            query_emb,
            threshold=self.config.CONFIDENCE_THRESHOLD,
            query_text=raw_query,
        )

        if source_tag:
            results = [r for r in results if r.get("source_type") == source_tag]

        if not results:
            text = "I couldn't find anything relevant in the docs. Try rephrasing."
            if source_tag:
                text = (
                    f"I couldn't find anything relevant in '{source_tag}' sources. "
                    "Try a different tag or omit --from."
                )
        else:
            lines = ["I found these relevant documentation snippets:\n"]
            for r in results:
                doc_text = r.get("text", "").strip()
                source = r.get("source_type", "unknown")
                lines.append(
                    f"**{r['module']}:{r['symbol']}** ({r['kind']}) [{source}]"
                )
                lines.append(f"    Source: `{r['path']}`")
                if doc_text:
                    lines.append(f"    > {doc_text[:300].replace(chr(10), chr(10) + '    > ')}")
                lines.append(f"    _Score: {r['score']}_\n")
            text = "\n".join(lines)

        await self.client.room_send(
            room_id,
            "m.room.message",
            {
                "msgtype": "m.text",
                "body": text,
                "m.relates_to": {"rel_type": "m.thread", "event_id": event_id},
            },
        )

    # ── Reaction → room forwarding ──────────────────────────────────────────

    async def _reaction_event_handler(self, room, event):
        """Handle reaction events for message marking."""
        await self._handle_reaction(room, event)

    async def _handle_reaction(self, room, event):
        """Forward a reacted-to message to the configured target room.

        When a user reacts to a message with a configured emoji (e.g. ⭐),
        the bot fetches the original message and posts it to the target room
        (e.g. "Starred"). The target room is created on first use and the
        reacting user is invited.
        """
        react_routes = self.config.REACT_ROUTES
        if not react_routes:
            return

        if event.sender == self.client.user_id:
            return
        if hasattr(self, "_started_at_ms") and event.server_timestamp < self._started_at_ms:
            return

        content = getattr(event, "content", None)
        if content is None:
            return

        relates_to = content.get("m.relates_to", {})
        if not isinstance(relates_to, dict):
            return

        reaction_key = relates_to.get("key", "")
        target_room_name = react_routes.get(reaction_key)
        if not target_room_name:
            return

        reacted_event_id = relates_to.get("event_id")
        if not reacted_event_id:
            return

        try:
            resp = await self.client.room_get_event(room.room_id, reacted_event_id)
            if not resp or not hasattr(resp, "event"):
                return

            original_event = resp.event
            sender = original_event.sender
            body = getattr(original_event, "body", "")
            if not body:
                return

            forward_text = (
                f"📌 Message forwarded by {event.sender}\n"
                f"Original by {sender} in {room.room_name or room.room_id}:\n\n{body}"
            )

            target_room_id = await self._get_or_create_room(target_room_name)
            if not target_room_id:
                logger.warning("Could not get/create target room '%s'", target_room_name)
                return

            try:
                await self.client.room_invite(target_room_id, event.sender)
            except Exception:
                pass

            await self.client.room_send(
                target_room_id,
                "m.room.message",
                {"msgtype": "m.text", "body": forward_text},
            )
            logger.info(
                "Forwarded reaction %s from %s to room '%s'",
                reaction_key,
                event.sender,
                target_room_name,
            )
        except Exception as e:
            logger.warning("Failed to handle reaction %s: %s", reaction_key, e)

    async def _get_or_create_room(self, room_name: str) -> str | None:
        """Return the room_id for a named room, creating it if it doesn't exist.

        Uses a local cache to avoid repeated lookups. Room aliases are derived
        from the room name (lowercased, spaces → underscores).
        """
        if room_name in self._room_cache:
            return self._room_cache[room_name]

        alias_local = room_name.lower().replace(" ", "_")
        full_alias = f"#{alias_local}:{self.config.MATRIX_SERVER_NAME}"

        try:
            resp = await self.client.room_resolve_alias(full_alias)
            if resp and hasattr(resp, "room_id") and resp.room_id:
                self._room_cache[room_name] = resp.room_id
                return resp.room_id
        except Exception:
            pass

        try:
            resp = await self.client.room_create(
                name=room_name,
                alias=alias_local,
                is_direct=False,
                preset=RoomPreset.trusted_private_chat,
                visibility=RoomVisibility.private,
            )
            if resp and hasattr(resp, "room_id") and resp.room_id:
                room_id = resp.room_id
                self._room_cache[room_name] = room_id
                print(f"Created reaction room '{room_name}' ({room_id})")
                return room_id
        except Exception as e:
            logger.warning("Failed to create room '%s': %s", room_name, e)

        return None
