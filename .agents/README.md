# Coding-agent guidance

[AGENTS.md](../AGENTS.md) is the shared instruction source for developing
UserSim. [CLAUDE.md](../CLAUDE.md) points to it for Claude Code.

Keep shared guidance here and development commands in Make. Keep personal
instructions in ignored `AGENTS.local.md`, `CLAUDE.local.md`, or `.agents/local/`;
keep private logs and session artifacts in `.agents/logs/`, `.agents/sessions/`,
or `.agent-work/`. Keep credentials and client settings out of shared files.
Local instructions may need explicit loading by the client.
