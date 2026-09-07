"""Trusted, data-only completion checks configured by the embedding application."""

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class VerificationPolicy(BaseModel):
    """All commands must pass in one attempt; retries rerun the complete suite.

    Commands are fixed application configuration, not model output or repository
    discovery. Supplying this policy authorizes these exact verification commands.
    Shell/workspace restrictions and explicit action-policy denials still apply.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    test_command: str | None = Field(default="pytest", max_length=4096, strict=True)
    lint_command: str | None = Field(default=None, max_length=4096, strict=True)
    required_commands: tuple[str, ...] = Field(default=(), max_length=32)
    max_verification_attempts: int = Field(default=3, gt=0, strict=True)

    @field_validator("test_command", "lint_command")
    @classmethod
    def valid_command(cls, value: str | None) -> str | None:
        if value is not None and (not value.strip() or "\x00" in value):
            raise ValueError("Verification commands must be nonblank and contain no NUL")
        return value

    @field_validator("required_commands")
    @classmethod
    def valid_required_commands(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            cls.valid_command(value)
            if len(value) > 4096:
                raise ValueError("Verification command exceeds 4096 characters")
        return values

    @model_validator(mode="after")
    def nonempty_suite(self):
        if not self.commands:
            raise ValueError("Verification requires at least one command")
        return self

    @property
    def commands(self) -> tuple[str, ...]:
        return tuple(command for command in (self.test_command, self.lint_command) if command is not None) + (
            self.required_commands
        )