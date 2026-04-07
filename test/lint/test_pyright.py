# Copyright 2026 Damien SIX (damien@robotsix.net)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pyright type-checking test for colcon/ament test integration."""

import json
import os
import shutil
import subprocess

import pytest


@pytest.mark.pyright
@pytest.mark.linter
def test_pyright():
    """Run pyright type checker and fail if any type errors are found."""
    pyright_bin = shutil.which('pyright')
    if pyright_bin is None:
        # Try npx as fallback (pyright installed via npm)
        npx_bin = shutil.which('npx')
        if npx_bin is None:
            pytest.skip('pyright is not installed (neither pyright nor npx found)')
        cmd = [npx_bin, 'pyright', '--outputjson']
    else:
        cmd = [pyright_bin, '--outputjson']

    # Run from the package root so pyrightconfig.json is picked up
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    result = subprocess.run(
        cmd,
        cwd=package_root,
        capture_output=True,
        text=True,
    )

    # pyright exits 0 on success, 1 on errors/warnings, 2+ on fatal errors
    # --outputjson writes JSON to stdout
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError:
        pytest.fail(
            'pyright did not produce valid JSON output.\n'
            f'stdout: {result.stdout[:500]}\n'
            f'stderr: {result.stderr[:500]}'
        )

    summary = report.get('summary', {})
    error_count = summary.get('errorCount', 0)

    if error_count == 0:
        return

    # Format a readable error list for the assertion message
    lines = [f'Found {error_count} pyright type error(s):']
    for diag in report.get('generalDiagnostics', []):
        if diag.get('severity') != 'error':
            continue
        file_path = diag.get('file', '<unknown>')
        # Make path relative to package root for readability
        try:
            file_path = os.path.relpath(file_path, package_root)
        except ValueError:
            pass
        rang = diag.get('range', {})
        start = rang.get('start', {})
        line = start.get('line', 0) + 1  # pyright uses 0-based lines
        col = start.get('character', 0) + 1
        message = diag.get('message', '')
        lines.append(f'  {file_path}:{line}:{col}: {message}')

    assert error_count == 0, '\n'.join(lines)
