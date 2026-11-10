"""Executes the generated Colab notebook cells with google.colab simulated."""

import base64
import io
import re
import shutil
import sys
import types
import zipfile
from pathlib import Path

import pytest
from fakes import FakeClient, proposal

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import build_notebook  # noqa: E402


def cell_sources(nb):
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_notebook_embeds_current_package():
    nb = build_notebook.build()
    setup = cell_sources(nb)[0]
    payload = re.search(r'PACKAGE_ZIP = "([^"]+)"', setup).group(1)
    names = zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))).namelist()
    expected = {f"cutpreacher/{p.name}" for p in (ROOT / "cutpreacher").glob("*.py")}
    assert set(names) == expected
    committed = (ROOT / "notebook" / "CutPreacher.ipynb").read_text(encoding="utf-8")
    assert payload in committed, "notebook/CutPreacher.ipynb está desatualizado: rode scripts/build_notebook.py"


@pytest.fixture
def fake_colab(tmp_path, monkeypatch):
    downloads = []
    drive_root = tmp_path / "content"

    def mount(path):
        (drive_root / "drive" / "MyDrive").mkdir(parents=True, exist_ok=True)

    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    colab.drive = types.SimpleNamespace(mount=mount)
    colab.userdata = types.SimpleNamespace(get=lambda key: "sk-test")
    colab.files = types.SimpleNamespace(download=downloads.append)
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    saved_path, saved_modules = list(sys.path), {k: v for k, v in sys.modules.items() if k.startswith("cutpreacher")}
    yield drive_root, downloads
    sys.path[:] = saved_path
    for k in [k for k in sys.modules if k.startswith("cutpreacher")]:
        del sys.modules[k]
    sys.modules.update(saved_modules)


def test_notebook_runs_end_to_end(fake_colab, sermon_video, monkeypatch):
    drive_root, downloads = fake_colab
    setup, run_cell, download_cell = (s.replace("/content", str(drive_root)) for s in cell_sources(build_notebook.build()))
    ns: dict = {}
    exec(setup, ns)

    base = drive_root / "drive" / "MyDrive" / "CutPreacher"
    assert (base / "entrada").is_dir() and (base / "saida").is_dir()
    import os
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-test"

    import cutpreacher.pipeline as pipeline
    from test_integration import fake_transcript

    assert pipeline.__file__.startswith(str(drive_root)), "notebook must use the embedded package"
    duration = pipeline.probe(sermon_video).duration
    real_select = pipeline.select_clips
    fake = FakeClient([[proposal(5, 40, "Um"), proposal(45, 90, "Dois"), proposal(100, 140, "Três")]])
    monkeypatch.setattr(pipeline, "transcribe", lambda *a, **k: fake_transcript(duration))
    monkeypatch.setattr(pipeline, "select_clips", lambda t, s, client=None, log=print: real_select(t, s, client=fake, log=log))

    shutil.copy(sermon_video, base / "entrada" / "culto.mp4")
    run_cell = run_cell.replace('arquivo_no_drive = ""', 'arquivo_no_drive = "culto.mp4"')
    exec(run_cell, ns)

    result = ns["resultado"]
    assert len(result.files) == 3 and all(f.is_file() for f in result.files)
    assert result.output_dir.parent == base / "saida"
    assert fake.calls[0]["model"] == "claude-opus-5-5"

    exec(download_cell, ns)
    assert downloads == [str(f) for f in result.files]


def test_notebook_requires_input(fake_colab):
    drive_root, _ = fake_colab
    setup, run_cell, _ = (s.replace("/content", str(drive_root)) for s in cell_sources(build_notebook.build()))
    ns: dict = {}
    exec(setup, ns)
    with pytest.raises(ValueError, match="Preencha"):
        exec(run_cell, ns)
