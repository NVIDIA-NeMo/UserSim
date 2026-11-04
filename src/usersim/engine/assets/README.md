# Shared asset root

User Sim's built-in probes keep their banks beside their code, in
`usersim/engine/probes/<folder>/assets/<label>/`, so this directory holds no
banks of its own. It is the last root on the asset search path, and naming it
with `--assets-dir` pins the assets that ship with User Sim, which reaches each
built-in probe's own folder.
