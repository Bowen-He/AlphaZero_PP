"""Explicit memory for partial-map reactive policies.

Updated by the episode runner after each environment step.
Consulted by ReactiveContext during predicate evaluation.
The BT policy never writes to memory — only the simulation does.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from alphazeropp.instances.doors.dsl.doors_config import DoorsGameConfig


@dataclass
class ReactiveMemory:
    """Agent memory tracking discovered key locations and searched rooms.

    Fields:
        known_key_locs: KeyId → LocId mapping, populated when agent enters
            a room containing a key.
        searched_rooms: Set of RoomIds the agent has physically entered.
    """
    known_key_locs: dict[int, int] = field(default_factory=dict)
    searched_rooms: set[int] = field(default_factory=set)

    @staticmethod
    def empty() -> ReactiveMemory:
        """Create a blank memory with nothing discovered."""
        return ReactiveMemory()

    def discover_keys_in_room(self, room: int, cfg: DoorsGameConfig) -> None:
        """Reveal any keys physically located in *room*.

        Called by the episode runner (never by the BT policy) to simulate
        what a physical agent would observe upon entering a room.
        """
        for k in range(cfg.K):
            loc = cfg.key_loc[k]
            if cfg.loc_room[loc] == room and k not in self.known_key_locs:
                self.known_key_locs[k] = loc

    def mark_searched(self, room: int) -> None:
        """Record that the agent has entered *room*."""
        self.searched_rooms.add(room)

    def loc_of_key(self, k: int) -> int | None:
        """Return the discovered location of key *k*, or None."""
        return self.known_key_locs.get(k)

    def copy(self) -> ReactiveMemory:
        """Return an independent copy."""
        return ReactiveMemory(
            known_key_locs=dict(self.known_key_locs),
            searched_rooms=set(self.searched_rooms),
        )
