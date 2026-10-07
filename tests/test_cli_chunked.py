import pytest

from faker.cli import Command


class TestCLIChunkedOutput:
    """Tests for the command-line chunked repeat pathway."""

    def test_chunked_output_is_streamed_to_stdout(self, capsys):
        Command(
            [
                "faker",
                "-l",
                "en_US",
                "--seed",
                "1",
                "--chunk-size",
                "5",
                "-r",
                "12",
                "fixed_width",
            ],
        ).execute()

        captured = capsys.readouterr()
        non_empty_lines = [line for line in captured.out.splitlines() if line.strip()]
        assert len(non_empty_lines) == 12  # 5 + 5 + 2 rows delivered in three chunks

    def test_chunked_output_is_streamed_and_flushed_to_file(self, tmp_path):
        output_path = tmp_path / "chunked.txt"
        Command(
            [
                "faker",
                "-l",
                "en_US",
                "--seed",
                "1",
                "--chunk-size",
                "5",
                "-r",
                "12",
                "-o",
                str(output_path),
                "fixed_width",
            ],
        ).execute()

        lines = [line for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) == 12

    def test_capacity_rejection_removes_empty_output_file(self, tmp_path, capsys):
        output_path = tmp_path / "never.csv"
        with pytest.raises(SystemExit) as excinfo:
            Command(
                [
                    "faker",
                    "-l",
                    "en_US",
                    "--chunk-size",
                    "5",
                    "-r",
                    "100",
                    "--max-chunks",
                    "3",
                    "-o",
                    str(output_path),
                    "csv",
                ],
            ).execute()

        assert excinfo.value.code == 1
        assert not output_path.exists()  # rejected run leaves no empty artifact behind
        assert "max_chunks" in capsys.readouterr().err

    def test_chunked_output_requires_structured_fake(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            Command(["faker", "--chunk-size", "5", "-r", "10", "name"]).execute()
        assert excinfo.value.code == 2
        assert "chunked output only supports" in capsys.readouterr().err

    def test_legacy_repeat_loop_unchanged_without_chunk_size(self, capsys):
        Command(["faker", "-l", "en_US", "--seed", "1", "-r", "2", "name"]).execute()
        lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
        assert len(lines) == 2
