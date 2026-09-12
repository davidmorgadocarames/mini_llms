"""Batch-hash registry: Cracked-D writes it; the Etapa 2 models verify against it
and stop on any mismatch (guards the freeze rule)."""

import pytest

from compare_lab.train.pretrain import HashRegistry


def test_writer_appends_and_reader_verifies(tmp_path):
    path = tmp_path / "batch_hashes.txt"
    writer = HashRegistry(path, write=True)
    writer.check_or_append(0, "aaa")
    writer.check_or_append(1, "bbb")
    assert path.read_text().split() == ["aaa", "bbb"]

    reader = HashRegistry(path, write=False)
    reader.check_or_append(0, "aaa")  # matches -> ok
    reader.check_or_append(1, "bbb")


def test_reader_raises_on_mismatch(tmp_path):
    """A genuine config drift: same-shaped hash, different value."""
    path = tmp_path / "batch_hashes.txt"
    HashRegistry(path, write=True).check_or_append(0, "a" * 40)
    reader = HashRegistry(path, write=False)
    with pytest.raises(RuntimeError, match="hash mismatch"):
        reader.check_or_append(0, "b" * 40)


def test_truncated_registry_entry_is_reported_as_corruption(tmp_path):
    """A wrong-LENGTH entry means the file was torn mid-append (a run died), not
    that the config drifted -- saying "config drifted" there would send you
    hunting for a change that never happened."""
    path = tmp_path / "batch_hashes.txt"
    path.write_text("abc123\n")  # truncated line
    reader = HashRegistry(path, write=False)
    with pytest.raises(RuntimeError, match="corrupt"):
        reader.check_or_append(0, "a" * 40)


def test_mismatch_message_shows_full_hashes(tmp_path):
    """Truncating to 12 chars could print two identical-looking hashes that
    differ further along -- unreadable at hour five of a run."""
    path = tmp_path / "batch_hashes.txt"
    a = "f" * 30 + "1111111111"
    b = "f" * 30 + "2222222222"
    HashRegistry(path, write=True).check_or_append(0, a)
    reader = HashRegistry(path, write=False)
    with pytest.raises(RuntimeError) as exc:
        reader.check_or_append(0, b)
    assert a in str(exc.value) and b in str(exc.value)


def test_writer_self_verifies_on_resume(tmp_path):
    """After a hard kill the registry can be AHEAD of the checkpoint, so a
    resumed writer revisits batches it already recorded: those must verify, not
    duplicate, and not raise a bogus mismatch."""
    path = tmp_path / "batch_hashes.txt"
    a, b = "a" * 40, "b" * 40
    w = HashRegistry(path, write=True)
    w.check_or_append(0, a)
    w.check_or_append(1, b)

    w2 = HashRegistry(path, write=True)      # resumed run, restarts at batch 0
    w2.check_or_append(0, a)
    w2.check_or_append(1, b)
    w2.check_or_append(2, "c" * 40)          # then extends
    assert path.read_text().split() == [a, b, "c" * 40]

    with pytest.raises(RuntimeError, match="hash mismatch"):
        w2.check_or_append(0, "z" * 40)


def test_missing_registry_for_reader_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        HashRegistry(tmp_path / "nope.txt", write=False)
