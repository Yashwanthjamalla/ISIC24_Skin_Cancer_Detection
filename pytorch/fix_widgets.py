
# fix_widgets.py — remove stale ipywidgets metadata so GitHub can render the notebook
import nbformat, sys

nb_path = sys.argv[1] if len(sys.argv) > 1 else "pytorch/Image+MetaCombined.ipynb"

# load without changing version
nb = nbformat.read(nb_path, as_version=nbformat.NO_CONVERT)

# GitHub expects metadata.widgets.state; if missing, drop the whole widgets block
widgets = nb.metadata.get('widgets')
if isinstance(widgets, dict) and 'state' not in widgets:
    nb.metadata.pop('widgets', None)

# strip widget-view outputs that depend on front-end state
for cell in nb.cells:
    if cell.get('outputs'):
        cell.outputs = [
            o for o in cell.outputs
            if not ('data' in o and isinstance(o['data'], dict)
                    and 'application/vnd.jupyter.widget-view+json' in o['data'])
        ]

nbformat.write(nb, nb_path)
print(f"[ok] sanitized: {nb_path}")
