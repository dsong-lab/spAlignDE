SPHINXOPTS    ?=
SPHINXBUILD   ?= env -u PYTHONPATH PYTHONNOUSERSITE=1 sphinx-build
PYTHON        ?= python3
DOCSDIR       := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
REPOROOT      := $(abspath $(DOCSDIR)/..)
SOURCEDIR     = $(DOCSDIR)/source
BUILDDIR      = $(DOCSDIR)/build

.PHONY: html clean

html:
	$(PYTHON) "$(REPOROOT)/tools/stage_documentation.py"
	$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

clean:
	rm -rf "$(BUILDDIR)"
