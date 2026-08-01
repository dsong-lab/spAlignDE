SPHINXOPTS    ?=
SPHINXBUILD   ?= env -u PYTHONPATH PYTHONNOUSERSITE=1 sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

.PHONY: html clean

html:
	$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

clean:
	rm -rf "$(BUILDDIR)"
