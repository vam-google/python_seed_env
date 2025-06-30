import unittest
from unittest.mock import patch, MagicMock, call
import os
import sys
import tempfile
import shutil

import build_maxtext_lock_cli

# Configuration Constants
GITHUB_ORG = "AI-Hypercomputer"
GITHUB_REPO = "maxtext"
# Name of the requirements file in the GitHub
REQUIREMENTS_FILE_NAME = "requirements.txt"
# Supported python versions of the CLI tool
SUPPORTED_PYTHON_VERSIONS = {"3.11", "3.12"}
# Latest JAX version
LATEST_JAX_VERSION = "jax-v0.6.2"
# Patch file path for JAX on Python 3.10 version
JAX_PATCH_FILE = "jax_requirements_lock_3_10.patch"
# Name of the GPU constrains file
CONSTRAINTS_GPU_ONLY = "constraints_gpu_only.txt"
# Name of the TPU constrains file
CONSTRAINTS_TPU_ONLY = "constraints_tpu_only.txt"

class TestLockFileGenerator(unittest.TestCase):

    def setUp(self):
        """
        Set up for each test: mock sys.argv and create a temporary directory to
        avoid polluting the actual file system.
        """
        self.original_argv = sys.argv
        self.temp_dir = tempfile.mkdtemp()
        os.chdir(self.temp_dir)

    def tearDown(self):
        sys.argv = self.original_argv
        os.chdir(self.original_argv[0].rsplit(os.sep, 1)[0] if os.sep in self.original_argv[0] else '.')
        shutil.rmtree(self.temp_dir)

    @patch('build_maxtext_lock_cli.utils.is_valid_commit')
    @patch('build_maxtext_lock_cli.utils.download_remote_file')
    @patch('build_maxtext_lock_cli.utils.fix_maxtext_requirements')
    @patch('build_maxtext_lock_cli.prepare_jax_seed.run')
    @patch('build_maxtext_lock_cli.utils.generate_pyproject_toml')
    @patch('build_maxtext_lock_cli.utils.build_seed_env')
    @patch('build_maxtext_lock_cli.shutil.move')
    @patch('build_maxtext_lock_cli.os.makedirs')
    @patch('build_maxtext_lock_cli.os.remove')
    @patch('build_maxtext_lock_cli.os.path.exists', return_value=True)
    @patch('builtins.print')
    def test_main_missing_required_arguments_fails(self, mock_print, mock_exists, mock_remove, mock_makedirs,
                                     mock_move, mock_build_seed_env, mock_generate_pyproject_toml,
                                     mock_prepare_jax_seed_run, mock_fix_maxtext_requirements,
                                     mock_download_remote_file, mock_is_valid_commit):
        """
        Test that main function exits with error when required arguments are missing.
        """
        # Set the command line
        sys.argv = ['build_maxtext_lock_cli.py']

        with self.assertRaises(SystemExit) as cm:
            build_maxtext_lock_cli.main()
        self.assertEqual(cm.exception.code, 2)

        mock_download_remote_file.assert_not_called()
        mock_fix_maxtext_requirements.assert_not_called()
        mock_is_valid_commit.assert_not_called()
        mock_prepare_jax_seed_run.assert_not_called()
        mock_generate_pyproject_toml.assert_not_called()
        mock_build_seed_env.assert_not_called()
        mock_move.assert_not_called()
        mock_makedirs.assert_not_called()
        mock_remove.assert_not_called()


    @patch('build_maxtext_lock_cli.utils.is_valid_commit')
    @patch('build_maxtext_lock_cli.utils.download_remote_file')
    @patch('build_maxtext_lock_cli.utils.fix_maxtext_requirements')
    @patch('build_maxtext_lock_cli.prepare_jax_seed.run')
    @patch('build_maxtext_lock_cli.utils.generate_pyproject_toml')
    @patch('build_maxtext_lock_cli.utils.build_seed_env')
    @patch('build_maxtext_lock_cli.shutil.move')
    @patch('build_maxtext_lock_cli.os.makedirs')
    @patch('build_maxtext_lock_cli.os.remove')
    @patch('build_maxtext_lock_cli.os.path.exists', return_value=True)
    @patch('builtins.print')
    def test_main_custom_arguments(self, mock_print, mock_exists, mock_remove, mock_makedirs,
                                    mock_move, mock_build_seed_env, mock_generate_pyproject_toml,
                                    mock_prepare_jax_seed_run, mock_fix_maxtext_requirements,
                                    mock_download_remote_file, mock_is_valid_commit):
        """Test main function with custom maxtext commit, jax commit, and python versions."""
        # Set the command line
        custom_maxtext_commit = "abcdef123456"
        custom_jax_commit = "abcdef123456"
        custom_python_versions = ["3.11", "3.12"]
        sys.argv = [
            'build_maxtext_lock_cli.py',
            '--maxtext-github-commit', custom_maxtext_commit,
            '--jax-github-commit-or-version', custom_jax_commit,
            '--python-versions', '3.11', '3.12'
        ]

        mock_is_valid_commit.return_value = True

        with patch('build_maxtext_lock_cli.apply_patch') as mock_apply_patch:
            result = build_maxtext_lock_cli.main()

            self.assertEqual(result, 0)

            expected_remote_url = f"https://raw.githubusercontent.com/{GITHUB_ORG}/{GITHUB_REPO}/{custom_maxtext_commit}/{REQUIREMENTS_FILE_NAME}"
            mock_download_remote_file.assert_called_once_with(expected_remote_url)
            mock_fix_maxtext_requirements.assert_called_once_with(REQUIREMENTS_FILE_NAME)

            # Check prepare_jax_seed.run calls
            mock_prepare_jax_seed_run.assert_has_calls([
                call(custom_jax_commit, "3.11"),
                call(custom_jax_commit, "3.12"),
            ], any_order=True)

            # Check build_seed_env calls for all combinations
            expected_build_seed_env_calls = []
            for pv in custom_python_versions:
                py_version_sanitized = pv.replace('.', '_')
                jax_temp_lock_file = f"requirements_lock_{py_version_sanitized}.txt"
                for mt in ['tpu', 'gpu']:
                    output_maxtext_requirement_lock_file = f"maxtext_requirements_lock_{mt}_{py_version_sanitized}.txt"
                    constraints_file = CONSTRAINTS_TPU_ONLY if mt == 'tpu' else CONSTRAINTS_GPU_ONLY
                    expected_build_seed_env_calls.append(
                        call(
                            seed_file=jax_temp_lock_file,
                            project_requirements_file=REQUIREMENTS_FILE_NAME,
                            output_file=output_maxtext_requirement_lock_file,
                            constraints_file=constraints_file,
                        )
                    )
            mock_build_seed_env.assert_has_calls(expected_build_seed_env_calls, any_order=True)

    @patch('build_maxtext_lock_cli.utils.is_valid_commit')
    @patch('build_maxtext_lock_cli.utils.download_remote_file')
    @patch('build_maxtext_lock_cli.sys.stderr', new_callable=MagicMock)
    @patch('builtins.print')
    def test_main_invalid_maxtext_commit(self, mock_print, mock_stderr, mock_download_remote_file, mock_is_valid_commit):
        """Test main function with an invalid MaxText commit."""
        # Set the command line
        invalid_commit = "cdefgh123456"
        sys.argv = [
            'build_maxtext_lock_cli.py',
            '--maxtext-github-commit', invalid_commit,
            '--jax-github-commit-or-version', invalid_commit,
            '--python-versions', '3.11', '3.12']

        # Simulate invalid commit
        mock_is_valid_commit.return_value = False

        result = build_maxtext_lock_cli.main()

        # Assert to for an error result
        self.assertEqual(result, 1)
        mock_is_valid_commit.assert_called_once_with(invalid_commit, f"{GITHUB_ORG}/{GITHUB_REPO}")
        mock_print.assert_any_call(f"Error: Provided commit/branch '{invalid_commit}' is not valid. Exiting.", file=mock_stderr)
        mock_download_remote_file.assert_not_called()

    @patch('build_maxtext_lock_cli.os.path.exists')
    @patch('build_maxtext_lock_cli.os.remove')
    @patch('builtins.print')
    def test_cleanup_files(self, mock_print, mock_remove, mock_exists):
        """Test the _cleanup_files utility function."""
        mock_exists.side_effect = [True, False, True]
        files = ["file1.txt", "file2.tmp", "file3.lock"]

        build_maxtext_lock_cli._cleanup_files(files)

        # file1.txt and file3.lock should have been removed
        mock_remove.assert_has_calls([call("file1.txt"), call("file3.lock")])
        self.assertEqual(mock_remove.call_count, 2)
        mock_print.assert_has_calls([
            call("Cleaned up: 'file1.txt'"),
            call("Cleaned up: 'file3.lock'"),
        ], any_order=True)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
