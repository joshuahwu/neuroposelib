"""
Generate the code reference pages and navigation.

The script is run by mkdocs at build time and performs the following tasks:

- Generates a "reference/" folder

This allows the code reference to be built using literate navigation.

For more details, refer to the [mkdocstrings recipes](https://mkdocstrings.github.io/recipes/#generate-pages-on-the-fly).
"""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

for path in sorted(Path("src").rglob("*.py")):
    module_path = path.relative_to("src").with_suffix("")
    doc_path = path.relative_to("src").with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)
    if parts[-1] == "__init__":
        # continue
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"::: {ident}")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open("reference/neuroposelib.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
