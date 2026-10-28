from evals.integrity import PROTECTED, compute_checksum, expected_checksum, main, write_expected


def make_tree(root, body=b"x\n"):
    for rel in PROTECTED:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)


def test_checksum_changes_when_a_protected_file_changes(tmp_path):
    make_tree(tmp_path)
    before = compute_checksum(tmp_path)
    (tmp_path / PROTECTED[0]).write_bytes(b"y\n")
    assert compute_checksum(tmp_path) != before


def test_line_endings_do_not_change_the_checksum(tmp_path):
    lf, crlf = tmp_path / "lf", tmp_path / "crlf"
    make_tree(lf, b"a\nb\n")
    make_tree(crlf, b"a\r\nb\r\n")
    assert compute_checksum(lf) == compute_checksum(crlf)


def test_write_and_read_expected(tmp_path):
    make_tree(tmp_path)
    assert expected_checksum(tmp_path) is None
    write_expected(tmp_path)
    assert expected_checksum(tmp_path) == compute_checksum(tmp_path)


def test_repo_checksum_matches_committed_file():
    from evals.integrity import REPO_ROOT

    assert expected_checksum(REPO_ROOT) == compute_checksum(REPO_ROOT)


def test_cli_check_exits_zero_on_the_repo():
    assert main([]) == 0
