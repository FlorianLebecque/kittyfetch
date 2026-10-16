# Changelog

## v1.0.0 (2026-10-16)

The first release: a cat that lives in your idle terminal, next to a tty-clock
and a fastfetch pane. One Python file, standard library only.

### Install

kittyfetch can be installed as a command, so you can just type `kittyfetch`
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

### The cats

- Seven coats, each with a name: Miso (ginger tabby), Pepper (black), Luna
  (silver tabby), Oreo (tuxedo), Mochi (siamese), Biscuit (tortoiseshell) and
  Yuki (odd-eyed white). One is picked at random each run; `--coat` chooses
  one and `c` cycles through them.
- Moods that drift: energy, boredom and mischief rise and fall and decide what
  the cat does next, with a penalty on repeating recent activities.
- Behaviours that chain and can be interrupted: waking up (yawn, stretch),
  napping (kneads, turns round, curls up, dreams), grooming, loafing,
  slow-blinking at you, wandering and sniffing, bird watching from the window
  ledge, playing with the yarn, laser hunting (stalk, butt wiggle, pounce),
  zoomies, being startled, surveying the room from on top of the fastfetch box,
  and knocking the mug off the ledge while looking you in the eye.

### Movement

- A procedural 3D skeleton seen from the side: a fixed-length 7-vertebra spine
  that bends but never stretches, shoulder blades that follow the front legs,
  digitigrade hind legs, a neck, and a spring-chain tail. Every pose value is
  driven by its own spring, so ears and tail lag the body.
- Paws planted in the world, stepping in diagonal pairs when the body drifts
  away from them; a gait clock tied to distance travelled so they never skate.
- A rotary gallop modelled on Muybridge's photographs, with two flight phases
  per stride and the spine stretching and gathering on those beats.
- Staged jumps: hind paws push off from where they stand, forelegs tuck, then
  reach down and land first.
- Turning by pivoting on the hind legs: the front half walks round in an arc,
  curling into the turn, with the head leading.
- Wall kicks: leap at the screen edge, plant all four paws, push off and twist
  round in the air. Used in zoomies and to climb onto high ledges.

### The world

- The status bar is the floor; the bottom borders of the "outside" pane and of
  the fastfetch box are ledges, joined by a hop planner.
- The sky outside follows your clock (sun, moon, stars, lit windows across the
  street) and the weather changes. Birds fly past and land on the ledge.
- Props: a laser dot (press `l`, or move the mouse), a ball of yarn (`y`) and
  a mug that gets tidied back onto the ledge while the cat sleeps.

### fastfetch and the TUI

- A fastfetch pane with real system info read from `/proc`, `/sys` and
  `/etc/os-release`: OS, host, kernel, uptime, packages (dpkg, pacman, flatpak,
  snap), shell, terminal, CPU, GPU, memory, swap, disk, load, battery and locale,
  plus the cat's mood and the terminal's ANSI colour blocks.
- A tty-clock with the date, and a tmux-style status bar.
- Panes stay behind the cat: their text hides wherever it walks in front.

### Terminals

- Truecolor half-block rendering on the terminal's own background, with a
  plain ASCII mode (`--ascii`, or `a` to toggle).
- Light and dark themes: the background colour is asked from the terminal at
  startup (or set with `--bg`), and the cat's edges and the text colours adapt.
- Adapts to resizing; Ctrl-C or `q` quits and restores the terminal.
- Each frame is diffed and sent as one write; it drops to 10 fps while the cat
  sleeps and typically uses a few percent of one core.

### Development

- `--soak` runs a headless multi-seed soak test with invariant checks and
  frame-time statistics; `--force` runs chosen behaviours first; `--shots`
  dumps the pixel layer as PNGs.
