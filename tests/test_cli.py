"""Unit tests for the asynchronous Typer CLI that manages API keys.

Each test launches the CLI with Typer's :class:`CliRunner` and injects
an async replacement for get_service so no real database is used.
The mocks return plain, lightweight objects (types.SimpleNamespace)
so the CLI prints predictable values instead of AsyncMock IDs.

All docstrings follow Google format; comments are concise and in English.
"""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from template.cli.commands.api_keys import cli_keys

# Shared runner used in every test.
runner = CliRunner()


@pytest.fixture()
def service_mock():
    """Return a mock that emulates APIKeyService."""
    svc = AsyncMock()

    # No return expected for delete.
    svc.delete.return_value = None

    # Object returned by service.create()
    svc.create.return_value = SimpleNamespace(id=1, name="TestKey", plain_key="raw-key")

    # Objects for activate/deactivate (id + is_active flag)
    svc.activate.side_effect = lambda key_id, active: SimpleNamespace(id=key_id, is_active=active)

    # Objects for list_all()
    svc.list_all.return_value = [
        SimpleNamespace(id=1, name="KeyA", is_active=True),
        SimpleNamespace(id=2, name="KeyB", is_active=False),
    ]
    return svc


def patch_get_service(target_module: str, service: AsyncMock):
    """Return a context manager that replaces get_service with a mock."""

    @asynccontextmanager
    async def _fake_get_service():
        yield service  # behaves exactly like the real one

    return patch(f"{target_module}.get_service", new=_fake_get_service)


def test_create_key(service_mock):
    """CLI create should print the new key's ID and plain key."""
    with patch_get_service("template.controller.commands.api_keys", service_mock):
        result = runner.invoke(
            cli_keys,
            [
                "create",
                "--name",
                "TestKey",
                "--description",
                "Some description",
                "--is-active",
            ],
        )

    assert result.exit_code == 0, result.output
    assert "✅ Key created: ID=1" in result.output
    assert "🔑 Plain key (save it now): raw-key" in result.output
    service_mock.create.assert_awaited_once()


def test_delete_key(service_mock):
    """CLI delete should call service.delete exactly once."""
    with patch_get_service("template.controller.commands.api_keys", service_mock):
        result = runner.invoke(cli_keys, ["delete", "1"])

    assert result.exit_code == 0, result.output
    assert "🗑️ Key with ID=1 deleted successfully." in result.output
    service_mock.delete.assert_awaited_once_with(1)


def test_activate_key(service_mock):
    """CLI activate should set the key to active and echo confirmation."""
    with patch_get_service("template.controller.commands.api_keys", service_mock):
        result = runner.invoke(cli_keys, ["activate", "1"])

    assert result.exit_code == 0, result.output
    assert "✅ Key ID=1 activated." in result.output
    service_mock.activate.assert_awaited_once_with(1, active=True)


def test_deactivate_key(service_mock):
    """CLI deactivate should set the key to inactive and echo confirmation."""
    with patch_get_service("template.controller.commands.api_keys", service_mock):
        result = runner.invoke(cli_keys, ["deactivate", "1"])

    assert result.exit_code == 0, result.output
    assert "🚫 Key ID=1 deactivated." in result.output
    service_mock.activate.assert_awaited_once_with(1, active=False)


def test_list_keys(service_mock):
    """CLI list must print every key returned by the service."""
    with patch_get_service("template.controller.commands.api_keys", service_mock):
        result = runner.invoke(cli_keys, ["list"])

    assert result.exit_code == 0, result.output
    assert "ID=1 | Name=KeyA | Active=True" in result.output
    assert "ID=2 | Name=KeyB | Active=False" in result.output
    service_mock.list_all.assert_awaited_once()
