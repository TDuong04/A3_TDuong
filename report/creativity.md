# Creativity: arena extensions

Our arena adds two gameplay systems beyond the Part II requirements in A3_Brief.pdf
(pages 6–9 and rubric G on page 17). **Collectible shields** turn destroyed spawners into
temporary protection opportunities: the player must move to a ten-second pickup to gain
one absorbed hit, introducing a timed collection and risk-management decision beyond the
required health and collision systems. **Elite chargers** extend the required enemies that
navigate toward the player with a four-state attack cycle: pursuit, a visible wind-up that
locks direction, a fast non-homing charge, and a recovery window for counterattacking; from
phase 2, their defeat is also required to complete an extended-mode phase. This introduces
anticipation, dodging, and attack timing beyond ordinary pursuit. The pilot-perception
compass and observation overlay support explaining the agent's inputs; muzzle flashes,
hit flicker, explosion particles, and screen shake make events legible. Those presentation
features are supporting evidence, while the shield and elite are the two additional gameplay
systems claimed for creativity. Both systems run in the same environment and unchanged action
spaces for humans and agents, with fixed-step timers and tests showing that rendering preserves
seeded simulation outcomes.

*Integration note for A3-014: this is a section draft, not the completed report. The new mode
uses 36 observations and needs newly trained policies; existing trained results describe the
baseline. No claim of learned mechanics proficiency is made from smoke training.*
