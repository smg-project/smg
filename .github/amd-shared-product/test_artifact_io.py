"""Offline behavioral controls for bounded API GET retries; no artifacts imported."""

import io
import os
import unittest
import urllib.error
import urllib.parse
from unittest import mock

import artifact_io


class ApiGetTests(unittest.TestCase):
    def setUp(self):
        self.token = "unit-test-api-token"
        self.environment = mock.patch.dict(os.environ, {"GITHUB_TOKEN": self.token})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.opener = mock.Mock()
        self.builder = mock.patch.object(
            artifact_io.urllib.request, "build_opener", return_value=self.opener
        )
        self.build = self.builder.start()
        self.addCleanup(self.builder.stop)
        self.blobs = mock.patch.object(
            artifact_io.urllib.request,
            "urlopen",
            side_effect=AssertionError("unexpected signed-URL request"),
        )
        self.blob_open = self.blobs.start()
        self.addCleanup(self.blobs.stop)

    def http_error(self, code, headers=None):
        body = io.BytesIO(b"original error body")
        error = urllib.error.HTTPError(
            artifact_io.API + "actions/runs/1", code, "test error", headers or {}, body
        )
        self.addCleanup(error.close)
        return error, body

    def assert_get_opens(self, count):
        self.assertEqual(self.opener.open.call_count, count)
        for args, kwargs in self.opener.open.call_args_list:
            request = args[0]
            self.assertEqual(request.get_method(), "GET")
            self.assertEqual(urllib.parse.urlsplit(request.full_url).hostname, "api.github.com")
            self.assertEqual(request.get_header("Authorization"), "Bearer " + self.token)
            self.assertEqual(kwargs, {"timeout": 30})
        self.assertTrue(
            any(isinstance(handler, artifact_io.NoRedirect) for handler in self.build.call_args.args)
        )

    def test_success_opens_once_and_returns_original_response(self):
        response = io.BytesIO(b"normal JSON response")
        self.opener.open.return_value = response
        with artifact_io.api_get("actions/runs/1") as actual:
            self.assertIs(actual, response)
            self.assertEqual(actual.read(), b"normal JSON response")
        self.assert_get_opens(1)

    def test_transient_http_then_success_closes_discarded_response(self):
        for code in (502, 503, 504):
            with self.subTest(code=code):
                self.opener.open.reset_mock()
                error, body = self.http_error(code)
                self.opener.open.side_effect = [error, io.BytesIO(b"ok")]
                with artifact_io.api_get("actions/runs/1") as response:
                    self.assertEqual(response.read(), b"ok")
                self.assertTrue(body.closed)
                self.assert_get_opens(2)

    def test_real_timeouts_then_success(self):
        for error in (TimeoutError("open timed out"), urllib.error.URLError(TimeoutError("open timed out"))):
            with self.subTest(error=type(error).__name__):
                self.opener.open.reset_mock()
                self.opener.open.side_effect = [error, io.BytesIO(b"ok")]
                with artifact_io.api_get("actions/runs/1") as response:
                    self.assertEqual(response.read(), b"ok")
                self.assert_get_opens(2)

    def test_exhausted_http_preserves_final_exception_and_body(self):
        first, first_body = self.http_error(502)
        final, final_body = self.http_error(504)
        self.opener.open.side_effect = [first, final, io.BytesIO(b"must not be reached")]
        with self.assertRaises(urllib.error.HTTPError) as caught:
            artifact_io.api_get("actions/runs/1")
        self.assertIs(caught.exception, final)
        self.assertTrue(first_body.closed)
        self.assertFalse(final_body.closed)
        self.assertEqual(caught.exception.read(), b"original error body")
        self.assert_get_opens(2)

    def test_exhausted_timeouts_preserve_final_exception(self):
        for wrapped in (False, True):
            with self.subTest(wrapped=wrapped):
                self.opener.open.reset_mock()
                first, final = TimeoutError("first"), TimeoutError("final")
                if wrapped:
                    first, final = urllib.error.URLError(first), urllib.error.URLError(final)
                self.opener.open.side_effect = [first, final, io.BytesIO(b"must not be reached")]
                with self.assertRaises(type(final)) as caught:
                    artifact_io.api_get("actions/runs/1")
                self.assertIs(caught.exception, final)
                self.assert_get_opens(2)

    def test_other_http_errors_are_immediate_and_unchanged(self):
        for code in (401, 403, 302, 404, 429, 500):
            with self.subTest(code=code):
                self.opener.open.reset_mock()
                error, body = self.http_error(code)
                self.opener.open.side_effect = [error, io.BytesIO(b"must not be reached")]
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    artifact_io.api_get("actions/runs/1")
                self.assertIs(caught.exception, error)
                self.assertFalse(body.closed)
                self.assert_get_opens(1)

    def test_non_timeout_url_errors_are_immediate(self):
        for reason in (OSError("connection reset"), "timed out"):
            with self.subTest(reason=reason):
                self.opener.open.reset_mock()
                error = urllib.error.URLError(reason)
                self.opener.open.side_effect = [error, io.BytesIO(b"must not be reached")]
                with self.assertRaises(urllib.error.URLError) as caught:
                    artifact_io.api_get("actions/runs/1")
                self.assertIs(caught.exception, error)
                self.assert_get_opens(1)

    def test_response_body_timeout_is_not_retried(self):
        class BodyTimeout(io.BytesIO):
            def read(self, *args):
                raise TimeoutError("body read timed out")

        self.opener.open.return_value = BodyTimeout()
        with self.assertRaises(TimeoutError):
            with artifact_io.api_get("actions/runs/1") as response:
                response.read()
        self.assert_get_opens(1)

    def test_archive_302_uses_plain_signed_url_without_api_token(self):
        signed_url = "https://blob.example.invalid/archive.zip?signature=test-only"
        redirect, body = self.http_error(302, {"Location": signed_url})
        self.opener.open.side_effect = redirect
        self.blob_open.side_effect = None
        self.blob_open.return_value = io.BytesIO(b"zip bytes")
        self.assertEqual(artifact_io.archive_bytes({"id": 7, "bytes": 9}), b"zip bytes")
        self.assert_get_opens(1)
        self.assertFalse(body.closed)
        self.blob_open.assert_called_once_with(signed_url, timeout=60)
        self.assertNotIn(self.token, self.blob_open.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
