import gc
import warnings

from remedi.data_handling.dataset.smiles_storage import SmilesStorage


def _make_store(tmp_path, lines=("CCO", "c1ccccc1", "CC(=O)O")):
    text_path = tmp_path / "smiles.txt"
    index_path = tmp_path / "smiles_idx"
    SmilesStorage.create_files(text_path, index_path)
    store = SmilesStorage(text_path=text_path, index_path=index_path)
    store.append_lines(lines)
    return store


def test_roundtrip(tmp_path):
    store = _make_store(tmp_path)
    assert store.to_list() == ["CCO", "c1ccccc1", "CC(=O)O"]
    store.close()


def test_context_manager_closes(tmp_path):
    text_path = tmp_path / "smiles.txt"
    index_path = tmp_path / "smiles_idx"
    SmilesStorage.create_files(text_path, index_path)
    with SmilesStorage(text_path=text_path, index_path=index_path) as store:
        store.append_lines(["CCO"])
        text_file = store._text_file
    assert text_file.closed


def test_del_releases_file_handle(tmp_path):
    store = _make_store(tmp_path)
    text_file = store._text_file
    assert not text_file.closed
    del store
    gc.collect()
    assert text_file.closed


def test_no_resource_warning_when_not_closed(tmp_path):
    # Simulate a crash that drops the store without calling close().
    with warnings.catch_warnings():
        warnings.simplefilter("error", ResourceWarning)
        store = _make_store(tmp_path)
        del store
        gc.collect()
