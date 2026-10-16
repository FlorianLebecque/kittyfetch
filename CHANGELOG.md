# Changelog

## Unreleased

### Install

kittyfetch can now be installed as a command, so you can just type `kittyfetch`
in any terminal.

- **One-line install:**

  ```bash
  curl -fsSL https://raw.githubusercontent.com/FlorianLebecque/kittyfetch/main/kittyfetch.py | python3 - --install
  ```

- **From a clone:** `python3 kittyfetch.py --install`.
- It installs to `~/.local/bin/kittyfetch` (or `$XDG_BIN_HOME` if set). If that
  folder isn't on your `PATH`, the installer prints the line to add to your
  `~/.bashrc` or `~/.zshrc`.
- `kittyfetch --uninstall` removes it.
- To update, run the install command again: the installed command is a copy,
  so it doesn't follow `git pull` on its own.
