"""Focused tests for resumable archive downloads without cloud access."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from flow360_utils import flow360_archive as archive


class InterruptedBody:
    def __init__(self, data, fail_after=None):
        self.stream = io.BytesIO(data)
        self.fail_after = fail_after
        self.closed = False

    def read(self, size):
        if self.fail_after is not None and self.stream.tell() >= self.fail_after:
            raise TimeoutError("simulated stalled stream")
        if self.fail_after is not None:
            size = min(size, self.fail_after - self.stream.tell())
        return self.stream.read(size)

    def close(self):
        self.closed = True


class FakeS3Client:
    def __init__(self, data, etag='"first"'):
        self.data = data
        self.etag = etag
        self.requests = []
        self.fail_after = None
        self.short_response = False
        self.bad_range = False

    def head_object(self, **kwargs):
        return {"ContentLength": len(self.data), "ETag": self.etag}

    def get_object(self, **kwargs):
        self.requests.append(kwargs)
        start = int(kwargs["Range"].split("=")[1].split("-")[0])
        expected = self.data[start:]
        data = expected[:-1] if self.short_response else expected
        return {
            "ContentRange": (
                "bytes 0-0/1" if self.bad_range else f"bytes {start}-{len(self.data) - 1}/{len(self.data)}"
            ),
            "ContentLength": len(expected),
            "ETag": self.etag,
            "Body": InterruptedBody(data, self.fail_after),
        }


class ResumableDownloadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.local_file = Path(self.directory.name) / "result.csv"
        self.progress_file = Path(self.directory.name) / "progress.json"
        self.client = FakeS3Client(b"abcdefghij")
        token = SimpleNamespace(get_bucket=lambda: "bucket", get_s3_key=lambda: "key")
        transfer = SimpleNamespace(_get_s3_sts_token=lambda *args: token)
        self.case = SimpleNamespace(id="case-1", s3_transfer_method=transfer)
        client_patch = patch.object(archive, "_bounded_s3_client", return_value=self.client)
        client_patch.start()
        self.addCleanup(client_patch.stop)

    def download(self):
        archive.download_resumable_file(
            self.case, "results/result.csv", self.local_file, self.progress_file
        )

    def test_interrupted_stream_resumes_from_written_prefix(self):
        self.client.fail_after = 4
        with self.assertRaises(TimeoutError):
            self.download()
        partial, metadata = archive._download_paths(self.local_file)
        self.assertEqual(partial.read_bytes(), b"abcd")
        self.assertTrue(metadata.exists())
        self.assertFalse(self.local_file.exists())

        self.client.fail_after = None
        self.download()
        self.assertEqual(self.client.requests[1]["Range"], "bytes=4-9")
        self.assertEqual(self.local_file.read_bytes(), b"abcdefghij")
        self.assertFalse(partial.exists())
        self.assertFalse(metadata.exists())
        self.assertEqual(json.loads(self.progress_file.read_text())["downloaded"], 10)

    def test_changed_etag_discards_partial_file(self):
        self.client.fail_after = 4
        with self.assertRaises(TimeoutError):
            self.download()
        self.client.etag = '"second"'
        self.client.data = b"0123456789"
        self.client.fail_after = None
        self.download()
        self.assertEqual(self.client.requests[1]["Range"], "bytes=0-9")
        self.assertEqual(self.local_file.read_bytes(), b"0123456789")

    def test_changed_size_discards_partial_file(self):
        self.client.fail_after = 4
        with self.assertRaises(TimeoutError):
            self.download()
        self.client.data = b"longer object"
        self.client.fail_after = None
        self.download()
        self.assertEqual(self.client.requests[1]["Range"], "bytes=0-12")
        self.assertEqual(self.local_file.read_bytes(), b"longer object")

    def test_short_response_keeps_partial_and_never_publishes_final_file(self):
        self.client.short_response = True
        with self.assertRaisesRegex(IOError, "Short S3 response"):
            self.download()
        self.assertFalse(self.local_file.exists())
        self.assertEqual(archive._download_paths(self.local_file)[0].read_bytes(), b"abcdefghi")

    def test_wrong_range_is_rejected_before_writing(self):
        self.client.bad_range = True
        with self.assertRaisesRegex(IOError, "Unexpected S3 range"):
            self.download()
        self.assertFalse(self.local_file.exists())
        self.assertFalse(archive._download_paths(self.local_file)[0].exists())

    def test_empty_object_completes(self):
        self.client.data = b""
        self.download()
        self.assertEqual(self.local_file.read_bytes(), b"")
        self.assertEqual(self.client.requests, [])


class CaseIdMarkerTests(unittest.TestCase):
    def test_case_id_file_is_written_for_download_and_restart(self):
        case_id = "case-123"
        case = SimpleNamespace(
            name="step0",
            get_download_file_list=lambda: [{"fileName": "simulation.json"}],
        )
        project = SimpleNamespace(get_case_ids=lambda: [case_id])
        fl = SimpleNamespace(
            Project=SimpleNamespace(from_cloud=lambda _: project),
            Case=SimpleNamespace(from_cloud=lambda _: case),
        )
        project_ref = archive.ProjectRef("project-1", "Demo", (), "Demo", "project-1", "project")

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)

            def download(**kwargs):
                kwargs["local_file"].write_text("result", encoding="utf-8")
                return str(kwargs["local_file"]), 0

            with patch.object(archive, "download_file_with_watchdog", side_effect=download):
                archive.download_project_cases(fl, project_ref, target, False, False, 0, 1, 1)

            marker = target / "Demo" / "step0" / f"{case_id}.txt"
            self.assertEqual(marker.read_text(encoding="utf-8"), f"{case_id}\n")

            with patch.object(archive, "download_file_with_watchdog") as downloader:
                archive.download_project_cases(fl, project_ref, target, False, True, 0, 1, 1)
                downloader.assert_not_called()

            self.assertEqual(marker.read_text(encoding="utf-8"), f"{case_id}\n")


if __name__ == "__main__":
    unittest.main()
