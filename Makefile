# Lecture 3 tutorial — Julia project + jupytext-paired notebook.
#
# Expected system tools on PATH (NOT provisioned here; see ../README.md):
#   julia, jupyter, jupytext

.PHONY: help init ipynb jupyter-lab jupyter-notebook clean distclean

.DEFAULT_GOAL := help

NOTEBOOK = gate_optimization

help:   ## Show this help
	@grep -E '^([a-zA-Z_-]+):.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "%-18s %s\n", $$1, $$2}'

.pkg-initialized: Project.toml
	julia --project=. -e 'import Pkg; Pkg.instantiate()'
	@if jupyter kernelspec list 2>/dev/null | grep -q '\bijulia\b'; then \
	  echo "Found shared 'IJulia' kernel."; \
	else \
	  echo "No shared 'IJulia' kernel found (see README.md)."; \
	  read -p "Install it now? [y/N] " ans; \
	  case "$$ans" in \
	    [yY]*) \
	      julia -e 'using Pkg; Pkg.activate(); Pkg.add("IJulia")'; \
	      julia -e 'using IJulia; installkernel("IJulia", "--project=@.", "--threads=auto"; julia=`julia`, displayname="IJulia", specname="ijulia")';; \
	    *) echo "Skipped. Install it manually per README.md before launching JupyterLab.";; \
	  esac; \
	fi
	@touch $@

init: .pkg-initialized  ## Instantiate the project and ensure the shared IJulia kernel exists

ipynb: $(NOTEBOOK).ipynb  ## Generate the unexecuted notebook from the .md source

$(NOTEBOOK).ipynb: $(NOTEBOOK).md
	jupytext --to notebook $(NOTEBOOK).md

jupyter-lab: .pkg-initialized  ## Run a JupyterLab server (select the "IJulia" kernel)
	jupyter lab --no-browser

jupyter-notebook: .pkg-initialized  ## Run a classic Jupyter notebook server
	jupyter notebook --no-browser

clean:  ## Remove generated notebooks and checkpoints
	rm -rf .ipynb_checkpoints .virtual_documents

distclean: clean  ## Also remove the Julia manifest and init sentinel
	rm -f .pkg-initialized
	rm -f $(NOTEBOOK).ipynb
	rm -f Manifest.toml
