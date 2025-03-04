import pytest
from unittest.mock import patch, Mock
from promptflow_tool_semantic_kernel.tools.tracing_disabler import TracingDisabler

import promptflow.tracing


class TestTracingDisabler:

    def test_tracing_is_disabled_within_context(self):
        # Mock the original is_tracing_disabled function
        original_func = Mock(return_value=False)

        with patch('promptflow.tracing._utils.is_tracing_disabled',
                   original_func):
            # Verify initial state
            assert not promptflow.tracing._utils.is_tracing_disabled()

            # Enter the context
            with TracingDisabler():
                # Verify tracing is disabled inside context
                assert promptflow.tracing._utils.is_tracing_disabled()

            # Verify original function is restored
            assert not promptflow.tracing._utils.is_tracing_disabled()

    def test_original_function_restored_after_exception(self):
        # Mock the original is_tracing_disabled function
        original_func = Mock(return_value=False)

        with patch('promptflow.tracing._utils.is_tracing_disabled',
                   original_func):
            # Verify initial state
            assert not promptflow.tracing._utils.is_tracing_disabled()

            # Enter context and raise exception
            try:
                with TracingDisabler():
                    assert promptflow.tracing._utils.is_tracing_disabled()
                    raise ValueError("Test exception")
            except ValueError:
                pass

            # Verify original function is restored even after exception
            assert not promptflow.tracing._utils.is_tracing_disabled()

    def test_context_manager_returns_itself(self):
        with patch('promptflow.tracing._utils.is_tracing_disabled',
                   Mock(return_value=False)):
            disabler = TracingDisabler()
            with disabler as result:
                assert result is disabler
