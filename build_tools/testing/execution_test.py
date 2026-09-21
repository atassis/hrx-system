# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import execution


class ExecutionUnitTest(unittest.TestCase):
    def test_substitution_preserves_unknown_braces(self):
        with tempfile.TemporaryDirectory() as directory:
            tool_path = Path(directory) / "tool"
            substituter = execution._Substituter(
                {"tmp": directory},
                {"fixture": execution.ToolCommand(executable=str(tool_path))},
            )
            self.assertEqual(
                substituter.substitute("{tmp}/file {json: true} {tool:fixture}"),
                f"{directory}/file {{json: true}} {tool_path}",
            )

    def test_substitution_rejects_multi_argument_tool_command(self):
        substituter = execution._Substituter(
            {},
            {
                "fixture": execution.ToolCommand(
                    executable=sys.executable,
                    arguments=("/path/to/fixture.py",),
                )
            },
        )
        with self.assertRaisesRegex(execution.SchemaError, "multi-argument command"):
            substituter.substitute("{tool:fixture}")

    def test_capture_stores_stripped_stdout(self):
        substituter = execution._Substituter({}, {})
        substituter.capture("greeting", b"  hello world  \n")
        self.assertEqual(substituter.substitute("say {greeting}"), "say hello world")

    def test_runner_launches_tool_command_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "cases": [
                            {
                                "name": "command prefix",
                                "run": {
                                    "tool": "fixture",
                                    "args": ["manifest argument"],
                                },
                                "stdout": {"contains": ["manifest argument"]},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            runner = execution.ExecutionRunner(
                tools={
                    "fixture": execution.ToolCommand(
                        executable=sys.executable,
                        arguments=(
                            "-c",
                            "import sys; print(sys.argv[1])",
                        ),
                    )
                }
            )

            self.assertEqual(
                runner.run_manifest(manifest_path), execution.RunSummary(case_count=1)
            )

    def test_capture_value_is_available_to_later_step_substitution(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "cases": [
                            {
                                "name": "capture then substitute",
                                "steps": [
                                    {
                                        "name": "produce",
                                        "run": {
                                            "tool": "fixture",
                                            "args": ["captured value"],
                                        },
                                        "capture": "greeting",
                                    },
                                    {
                                        "name": "consume",
                                        "run": {
                                            "tool": "fixture",
                                            "args": ["{greeting} suffix"],
                                        },
                                        "stdout": {
                                            "contains": ["captured value suffix"]
                                        },
                                    },
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            runner = execution.ExecutionRunner(
                tools={
                    "fixture": execution.ToolCommand(
                        executable=sys.executable,
                        arguments=("-c", "import sys; print(sys.argv[1])"),
                    )
                }
            )

            self.assertEqual(
                runner.run_manifest(manifest_path), execution.RunSummary(case_count=1)
            )

    def test_capture_referenced_before_definition_is_unresolved(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "cases": [
                            {
                                "name": "capture ordering",
                                "steps": [
                                    {
                                        "name": "before",
                                        "run": {
                                            "tool": "fixture",
                                            "args": ["{greeting}"],
                                        },
                                        "stdout": {"contains": ["{greeting}"]},
                                    },
                                    {
                                        "name": "after",
                                        "run": {"tool": "fixture", "args": ["value"]},
                                        "capture": "greeting",
                                    },
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            runner = execution.ExecutionRunner(
                tools={
                    "fixture": execution.ToolCommand(
                        executable=sys.executable,
                        arguments=("-c", "import sys; print(sys.argv[1])"),
                    )
                }
            )

            self.assertEqual(
                runner.run_manifest(manifest_path), execution.RunSummary(case_count=1)
            )

    def test_capture_does_not_leak_across_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "cases": [
                            {
                                "name": "captures greeting",
                                "steps": [
                                    {
                                        "name": "capture",
                                        "run": {"tool": "fixture", "args": ["hello"]},
                                        "capture": "greeting",
                                    }
                                ],
                            },
                            {
                                "name": "does not see prior capture",
                                "run": {"tool": "fixture", "args": ["{greeting}"]},
                                "stdout": {"contains": ["{greeting}"]},
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            runner = execution.ExecutionRunner(
                tools={
                    "fixture": execution.ToolCommand(
                        executable=sys.executable,
                        arguments=("-c", "import sys; print(sys.argv[1])"),
                    )
                }
            )

            self.assertEqual(
                runner.run_manifest(manifest_path), execution.RunSummary(case_count=2)
            )

    def test_parse_tool_bindings_appends_fixed_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "fixture"
            executable.touch()

            tools = execution.parse_tool_bindings(
                [f"fixture={executable}"],
                ["fixture=first", "fixture=second=value"],
            )

            self.assertEqual(
                tools["fixture"],
                execution.ToolCommand(
                    executable=str(executable),
                    arguments=("first", "second=value"),
                ),
            )

    def test_write_text_preserves_utf8_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.txt"
            text = "héllo\nworld\n"
            runner = execution.ExecutionRunner(tools={})

            runner._run_write_step(
                "fixture case", "write fixture", {"path": str(path), "text": text}
            )

            self.assertEqual(path.read_bytes(), text.encode("utf-8"))

    def test_contains_ordering(self):
        runner = execution.ExecutionRunner(tools={})
        runner._check_contains(
            "case", "step", "stdout", "alpha\nbeta\ngamma\n", ["alpha", "gamma"]
        )
        with self.assertRaises(execution.CaseFailure):
            runner._check_contains(
                "case", "step", "stdout", "alpha\nbeta\ngamma\n", ["gamma", "alpha"]
            )

    def test_non_empty_stream(self):
        runner = execution.ExecutionRunner(tools={})
        runner._check_stream(
            "case",
            "step",
            ["fixture"],
            "stdout",
            b"generated output\n",
            {"non_empty": True},
        )
        with self.assertRaisesRegex(execution.CaseFailure, "expected non-empty stdout"):
            runner._check_stream(
                "case",
                "step",
                ["fixture"],
                "stdout",
                b"",
                {"non_empty": True},
            )

    def test_stream_rejects_conflicting_empty_expectations(self):
        runner = execution.ExecutionRunner(tools={})
        with self.assertRaisesRegex(
            execution.SchemaError, "'empty' and 'non_empty' are mutually exclusive"
        ):
            runner._check_stream(
                "case",
                "step",
                ["fixture"],
                "stdout",
                b"",
                {"empty": True, "non_empty": True},
            )

    def test_file_equality_compares_exact_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            actual_path = Path(directory) / "actual.bin"
            expected_path = Path(directory) / "expected.bin"
            actual_path.write_bytes(b"\x00fixture\xff")
            expected_path.write_bytes(b"\x00fixture\xff")
            runner = execution.ExecutionRunner(tools={})

            runner._check_files(
                "case",
                "step",
                [{"path": str(actual_path), "equals": str(expected_path)}],
            )
            expected_path.write_bytes(b"\x00fixture\xfe")
            with self.assertRaisesRegex(execution.CaseFailure, "differs from"):
                runner._check_files(
                    "case",
                    "step",
                    [{"path": str(actual_path), "equals": str(expected_path)}],
                )

    def test_file_content_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text('{"count":5,"unknown":false}\n', encoding="utf-8")
            runner = execution.ExecutionRunner(tools={})
            for expectation in (
                {"contains": ['"count":5', {"regex": '"unknown":false'}]},
                {"not_contains": ['"count":0', {"regex": '"unknown":true'}]},
                {"contains": {"unordered": ['"unknown":false', '"count":5']}},
                {
                    "normalize": [
                        {"kind": "regex", "pattern": "5", "replacement": "N"}
                    ],
                    "contains": ['"count":N'],
                },
            ):
                with self.subTest(expectation=expectation):
                    runner._check_files(
                        "case", "step", [{"path": str(path), **expectation}]
                    )
            for expectation in (
                {"contains": ['"count":0']},
                {"contains": [{"regex": '"unknown":true'}]},
                {"not_contains": ['"count":5']},
                {"not_contains": [{"regex": '"unknown":false'}]},
                {"contains": ['"unknown":false', '"count":5']},
                {"empty": True},
            ):
                with self.subTest(expectation=expectation):
                    with self.assertRaises(execution.CaseFailure):
                        runner._check_files(
                            "case", "step", [{"path": str(path), **expectation}]
                        )

    def test_file_checks_reject_unknown_expectations(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text("{}", encoding="utf-8")
            runner = execution.ExecutionRunner(tools={})
            with self.assertRaisesRegex(execution.SchemaError, "unknown.*contain"):
                runner._check_files(
                    "case", "step", [{"path": str(path), "contain": ["count"]}]
                )

    def test_file_existence_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.bin"
            runner = execution.ExecutionRunner(tools={})
            runner._check_files("case", "step", [{"path": str(path), "exists": False}])
            with self.assertRaisesRegex(execution.CaseFailure, "to exist"):
                runner._check_files("case", "step", [{"path": str(path)}])
            path.write_bytes(b"\x00\xff")
            runner._check_files(
                "case", "step", [{"path": str(path), "non_empty": True}]
            )
            with self.assertRaisesRegex(execution.CaseFailure, "not to exist"):
                runner._check_files(
                    "case", "step", [{"path": str(path), "exists": False}]
                )
            with self.assertRaisesRegex(execution.SchemaError, "content.*absent"):
                runner._check_files(
                    "case",
                    "step",
                    [{"path": str(path), "exists": False, "contains": ["count"]}],
                )

    def test_sanitizer_env_resolves_suppressions_runfile(self):
        with tempfile.TemporaryDirectory() as directory:
            runfiles_dir = Path(directory)
            workspace_name = "test_workspace"
            suppression_path = (
                runfiles_dir
                / workspace_name
                / "build_tools"
                / "sanitizer"
                / "lsan_suppressions_vulkan.txt"
            )
            suppression_path.parent.mkdir(parents=True)
            suppression_path.write_text("# test\n", encoding="utf-8")

            env = {
                "LSAN_OPTIONS": (
                    "verbosity=1" + os.pathsep + "suppressions=build_tools/sanitizer/"
                    "lsan_suppressions_vulkan.txt"
                ),
            }
            with mock.patch.dict(
                os.environ,
                {
                    "RUNFILES_DIR": str(runfiles_dir),
                    "TEST_WORKSPACE": workspace_name,
                },
            ):
                execution._resolve_sanitizer_env(env)

            self.assertEqual(
                env["LSAN_OPTIONS"],
                "verbosity=1" + os.pathsep + f"suppressions={suppression_path}",
            )


if __name__ == "__main__":
    unittest.main()
