"""Tests for fetch_experimental_data tool."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from tools.fetch_experimental_data import (
    compute_sha256,
    main,
    verify_checksums,
)


class TestFetchExperimental:
    """Tests for experimental data fetch and verify functionality."""

    def test_verify_only_passes_with_correct_checksums(self) -> None:
        """Verify that --verify-only passes when checksums match."""
        # This test verifies the actual committed checksums match
        result = verify_checksums(
            Path(__file__).parent.parent.parent / "experimental",
            Path(__file__).parent.parent.parent / "experimental" / "CHECKSUMS.sha256",
        )
        assert result is True, "Committed experimental files should match checksums"

    def test_verify_only_fails_with_tampered_file(self, tmp_path: Path) -> None:
        """Verify that --verify-only fails when file is tampered."""
        # Create temp data file
        data_file = tmp_path / "test.dat"
        data_file.write_text("test data")

        # Create checksums file with wrong hash
        checksums_file = tmp_path / "CHECKSUMS.sha256"
        checksums_file.write_text("0000000000000000  test.dat\n")

        result = verify_checksums(tmp_path, checksums_file)
        assert result is False, "Tampered files should fail verification"

    def test_manifest_parses_four_sources(self) -> None:
        """Verify that FETCH_MANIFEST.toml has 4 sources."""
        manifest_path = (
            Path(__file__).parent.parent.parent
            / "experimental"
            / "FETCH_MANIFEST.toml"
        )

        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore

        with open(manifest_path, "rb") as f:
            manifest = tomllib.load(f)

        sources = manifest.get("source", [])
        assert len(sources) == 4, f"Expected 4 sources, got {len(sources)}"

    def test_sha256_committed_checksums_match_files(self) -> None:
        """Verify that CHECKSUMS.sha256 hashes match actual .dat files."""
        exp_dir = Path(__file__).parent.parent.parent / "experimental"

        for dat_file in sorted(exp_dir.glob("*.dat")):
            actual_hash = compute_sha256(dat_file)
            filename = dat_file.name

            # Check hash matches in CHECKSUMS.sha256
            with open(exp_dir / "CHECKSUMS.sha256") as f:
                for line in f:
                    if filename in line:
                        expected_hash = line.split()[0]
                        assert actual_hash == expected_hash, (
                            f"{filename}: hash mismatch\n"
                            f"  Expected: {expected_hash}\n"
                            f"  Got:      {actual_hash}"
                        )
                        break

    @patch("tools.fetch_experimental_data.urllib.request.urlopen")
    def test_retry_on_503(self, mock_urlopen: object) -> None:
        """Verify that fetch retries on HTTP 503 and succeeds on 3rd attempt."""
        from unittest.mock import MagicMock
        from urllib.error import HTTPError

        # Mock urlopen to fail twice with 503, then succeed
        mock_response = MagicMock()
        mock_response.__enter__.return_value.read.return_value = b'{"values": []}'

        side_effects = [
            HTTPError("url", 503, "Service Unavailable", {}, None),
            HTTPError("url", 503, "Service Unavailable", {}, None),
            mock_response,
        ]

        mock_urlopen.side_effect = side_effects  # type: ignore

        from tools.fetch_experimental_data import fetch_hepdata_table

        # Use deterministic RNG for testing (minimal jitter)
        rng = __import__("random").Random(0)
        result = fetch_hepdata_table("test_id", "table", rng=rng)

        assert result == [], "Should return empty list after retry success"
        assert mock_urlopen.call_count == 3, "Should have retried 3 times"  # type: ignore

    @patch("tools.fetch_experimental_data.urllib.request.urlopen")
    def test_no_retry_on_404(self, mock_urlopen: object) -> None:
        """Verify that fetch does NOT retry on HTTP 404."""
        from urllib.error import HTTPError

        mock_urlopen.side_effect = HTTPError("url", 404, "Not Found", {}, None)  # type: ignore

        from tools.fetch_experimental_data import fetch_hepdata_table

        with pytest.raises(HTTPError):
            fetch_hepdata_table("test_id", "table")

        assert mock_urlopen.call_count == 1, "Should NOT retry on 404"  # type: ignore

    def test_main_verify_only_mode(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify that main() --verify-only mode works."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--verify-only"])
        assert exc_info.value.code == 0, "Should exit with code 0 on success"
        captured = capsys.readouterr()
        assert "Verifying" in captured.out
        assert "OK:" in captured.out
