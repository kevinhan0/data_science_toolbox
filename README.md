# Data Science Toolbox

This repository is dedicated to the Python functions and classes that I've wrote throughout my data science career. I often find myself using them again and again in my projects, which is why I decided to make them into a Python package.

The `default.nix` file replaces the need for a requirements.txt and builds a working Python shell using nix with all the data science packages nicely. I will admit that I'm a `nix` noob and most of what I have here I pieced together by looking at tutorials and other people's code. I will add the reference soon.

---

### Installation

1. Install nix: `curl https://nixos.org/nix/install | sh`.
2. Build the nix-shell `nix-build`.
3. Open nix-shell `nix-shell`.
4. Cheat a with `pip install -e .` (`setup.py` to be added).

---

### Structure

```
- Exploratory Data Analysis (eda)
- Evaluation (evaluation)
    - evaluate.py
```
