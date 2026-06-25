# Tutorial

This repo contains the tutorial unit of the QNumerics Summer School session "Quantum
Optimal Control Theory": a worked example optimizing an entangling gate for two
coupled transmon qubits, for gate concurrence via semi-automatic differentiation.

The notebook is authored as a [jupytext](https://jupytext.readthedocs.io/)-paired
**MyST Markdown** file, `gate_optimization.md`. Edit the `.md`; the executable
`.ipynb` is generated from it.

## Prerequisites

System tools on `PATH`:

- [Julia](https://julialang.org/) installed via [juliaup](https://github.com/JuliaLang/juliaup)
- `jupyter` and `jupytext`
- A shared `IJulia` kernel (see below); `make init` installs it if missing.

### Recommended Jupyter setup

We recommend [`uv`](https://docs.astral.sh/uv/) for managing Python tooling. A
single system-wide JupyterLab environment that provides `jupyter`, `jupytext`,
`nbconvert`, and related utilities can be installed with:

```
uv tool install jupyterlab \
  --with black \
  --with isort \
  --with numpy \
  --with scipy \
  --with sympy \
  --with pandas \
  --with matplotlib \
  --with qutip \
  --with nbconvert \
  --with ipywidgets \
  --with ipyparallel \
  --with jupyter-collaboration \
  --with python-localvenv-kernel \
  --with markdown-kernel \
  --with jupyterlab_execute_time \
  --with jupyterlab_code_formatter \
  --with jupyterlab-spellchecker \
  --with ipdb \
  --with-executables-from ipython \
  --with-executables-from jupytext \
  --with-executables-from nbdime \
  --with-executables-from notebook \
  --with-executables-from nbconvert \
  --with-executables-from jupyter-core
```

The Julia kernel for JupyterLab is registered separately by `IJulia` when you
run `make init` in `tutorial/`.

### IJulia kernel

The project for the notebook is defined in `./Project.toml` and it is assumed that you have an
[`IJulia`](https://ijulia.org/stable/) kernel  that picks up on that environment.

My recommendation is to set up the kernel as follows _in your main environment_ (!!!):

```
julia -e 'using Pkg; Pkg.activate(); Pkg.add("IJulia")'
julia -e 'using IJulia; installkernel("IJulia", "--project=@.", "--threads=auto"; julia=`julia`, displayname="IJulia", specname="ijulia")'
```

Key points:

- The kernel launches the bare `julia` command, so `julia` on `PATH` must be the
  Juliaup launcher; Juliaup then dispatches to whatever Julia version is selected
  (globally or per directory). One kernel works across all installed versions.
- `IJulia` is installed into the default `@v#.#` environment so it is reachable
  regardless of which project is active.
- `--project=@.` activates the project in the notebook's directory (or nearest
  parent with a `Project.toml`); `--threads=auto` uses all CPU threads;
  `displayname`/`specname` drop the version suffix for a single stable kernel.

On macOS this writes `~/Library/Jupyter/kernels/ijulia/kernel.json`. Launch
JupyterLab from the tutorial directory and select "IJulia"; it picks up this
project's `Project.toml` automatically.

You can check installed kernels by running `jupyter kernelspec list`.

### Using a generic Julia kernel

If you instead run the notebook with a generic Julia kernel that does not
activate this directory's `Project.toml`, activate and instantiate the local
project explicitly at the top of the notebook:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Usage

- `make init` — instantiate the Julia project; check for the shared `IJulia`
  kernel and offer to install it if missing.
- `make jupyter-lab` — launch a JupyterLab server. Open `gate_optimization.md`
  (right-click → *Open With → Notebook*) and select the "IJulia" kernel.
- `make ipynb` — generate the unexecuted `gate_optimization.ipynb` from the `.md`.
- `make clean` / `make distclean` — remove generated notebooks / also the Julia
  manifest and init sentinel.

## Layout

- `Project.toml` — the Julia environment for the tutorial.
- `gate_optimization.md` — the notebook source (jupytext / MyST Markdown).
- `jupytext.toml` — pairs the `.md` with a generated `.ipynb`.
- `utils/` — helper Julia files used by the notebook (hints, code display).
- `figures/` — figures used in the notebook.
