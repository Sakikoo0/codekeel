import pytest
from pydantic import ValidationError

from codekeel.models import ToolCall
from codekeel.runtime.policy import ActionPolicy, CommandRule, Decision, Risk


def call(command):
    return ToolCall(id="call", name="shell", arguments={"command": command})


@pytest.mark.parametrize("name,arguments,risk,decision", [
    ("read_file", {"path": "README.md"}, Risk.LOW, Decision.ALLOW),
    ("write_file", {"path": "x", "content": "x"}, Risk.MEDIUM, Decision.ALLOW),
    ("shell", {"command": "pytest -q"}, Risk.LOW, Decision.ALLOW),
    ("shell", {"command": "pip install thing"}, Risk.HIGH, Decision.REQUIRE_APPROVAL),
    ("shell", {"command": "git commit -m 'fix thing'"}, Risk.HIGH, Decision.REQUIRE_APPROVAL),
    ("shell", {"command": "git push origin main"}, Risk.HIGH, Decision.DENY),
    ("shell", {"command": "rm -rf directory"}, Risk.HIGH, Decision.DENY),
    ("custom", {}, Risk.UNKNOWN, Decision.REQUIRE_APPROVAL),
])
def test_default_rules(name, arguments, risk, decision):
    result = ActionPolicy().assess(ToolCall(id="call", name=name, arguments=arguments))
    assert result.risk is risk and result.decision is decision


@pytest.mark.parametrize("mode,risk,expected", [
    ("always", Risk.LOW, Decision.REQUIRE_APPROVAL),
    ("never", Risk.UNKNOWN, Decision.ALLOW),
    ("risky", Risk.HIGH, Decision.REQUIRE_APPROVAL),
    ("risky", Risk.UNKNOWN, Decision.REQUIRE_APPROVAL),
    ("risky", Risk.MEDIUM, Decision.ALLOW),
])
def test_modes_roundtrip(mode, risk, expected):
    policy = ActionPolicy(mode=mode, tool_risks={"custom": risk})
    restored = ActionPolicy.model_validate_json(policy.model_dump_json())
    assert restored == policy
    assert restored.assess(ToolCall(id="c", name="custom")).decision is expected


def test_approval_handling_roundtrip():
    policy = ActionPolicy(approval_handling="unavailable")
    assert ActionPolicy.model_validate_json(policy.model_dump_json()) == policy


def test_configured_threshold_unknown_and_command_rules():
    assert ActionPolicy(threshold=Risk.MEDIUM).assess(ToolCall(id="c", name="write_file")).decision is (
        Decision.REQUIRE_APPROVAL
    )
    assert ActionPolicy(confirm_unknown=False).assess(call("custom")).decision is Decision.ALLOW
    policy = ActionPolicy(command_rules=(CommandRule(prefix=("custom", "check"), risk=Risk.LOW),))
    assert policy.assess(call("custom check -q")).decision is Decision.ALLOW
    assert policy.assess(call("custom checking")).decision is Decision.REQUIRE_APPROVAL


@pytest.mark.parametrize("mode", ["always", "never", "risky"])
def test_deny_wins_over_modes_and_overlapping_allow_rules(mode):
    policy = ActionPolicy(mode=mode, denied_tools=("read_file",), command_rules=(
        CommandRule(prefix=("git", "push", "origin"), risk=Risk.LOW),
        CommandRule(prefix=("git", "push"), risk=Risk.HIGH, deny=True),
    ))
    assert policy.assess(call("git push origin")).decision is Decision.DENY
    assert policy.assess(ToolCall(id="c", name="read_file")).decision is Decision.DENY


@pytest.mark.parametrize("command", [
    "pytest; git push", "pytest && rm -rf x", "pytest | sh", "pytest\ngit push",
    "pytest $(git push)", "echo `git push`", "pytest > file", "pytest < file",
    "pytest & git push", "pytest # comment", "pytest\\\ngit push", "echo ${COMMAND}",
    "echo *", "pytest\x00", "pytest 'unterminated", "/bin/rm -rf x", "./rm -rf x",
    "git -C repo push", "git -c alias.foo=push push", "/usr/bin/git push",
])
def test_shell_bypass_attempts_are_denied(command):
    assert ActionPolicy().assess(call(command)).decision is Decision.DENY


@pytest.mark.parametrize("command", ["env pytest", "sh -c 'git push'", "/bin/pytest", "pytest-malicious", "eval push"])
def test_wrappers_paths_and_prefix_lookalikes_never_inherit_low_risk(command):
    assert ActionPolicy().assess(call(command)).decision is Decision.REQUIRE_APPROVAL


def test_model_cannot_supply_its_own_risk_or_approval():
    action = call("pip install malware")
    action.arguments.update(risk="low", approved=True)
    assert ActionPolicy().assess(action).decision is Decision.REQUIRE_APPROVAL


@pytest.mark.parametrize("kwargs", [
    {"threshold": "unknown"}, {"mode": "typo"}, {"approval_handling": "typo"}, {"unknown_option": True},
])
def test_bad_policy_configuration_fails(kwargs):
    with pytest.raises(ValidationError):
        ActionPolicy(**kwargs)


@pytest.mark.parametrize("prefix", [(), ("",), ("git ",)])
def test_bad_command_rule_fails(prefix):
    with pytest.raises(ValidationError):
        CommandRule(prefix=prefix, risk=Risk.LOW)


@pytest.mark.parametrize("command", [
    "git log --oneline -5 -- astropy/wcs/wcsapi/fitswcs.py",
    "git log --oneline -5 -- 'path with spaces/file.py'",
    "git log --oneline -5 --",
])
def test_bounded_git_log_is_low_risk(command):
    result = ActionPolicy().assess(call(command))
    assert result.decision is Decision.ALLOW and result.risk is Risk.LOW
    assert ActionPolicy(mode="always").assess(call(command)).decision is Decision.REQUIRE_APPROVAL


@pytest.mark.parametrize("command", [
    "git log", "git log --oneline -5", "git log --oneline -500 -- file.py",
    "git log --oneline -5 --output=out -- file.py",
    "git -c core.pager=evil log --oneline -5 -- file.py",
    "/usr/bin/git log --oneline -5 -- file.py", "env git log --oneline -5 -- file.py",
])
def test_other_git_log_forms_do_not_inherit_permission(command):
    assert ActionPolicy().assess(call(command)).decision is not Decision.ALLOW


@pytest.mark.parametrize("suffix", ["; rm file", " && git push", " > out", "\nrm file", " $(evil)"])
def test_bounded_git_log_cannot_bypass_syntax_denial(suffix):
    assert ActionPolicy(mode="never").assess(call("git log --oneline -5 -- file" + suffix)).decision is Decision.DENY


def test_explicit_git_log_denial_still_wins():
    defaults = ActionPolicy().command_rules
    policy = ActionPolicy(command_rules=(*defaults, CommandRule(prefix=("git", "log"), risk=Risk.HIGH, deny=True)))
    assert policy.assess(call("git log --oneline -5 -- file.py")).decision is Decision.DENY


@pytest.mark.parametrize("command", [
    "rmdir directory", "mkfs device", "dd if=input of=output", "shutdown now", "git clean -fdx",
    "git reset --hard", "/usr/bin/rm file", "git -C repo reset --hard",
])
@pytest.mark.parametrize("mode", ["always", "never", "risky"])
def test_destructive_commands_are_denied_in_every_mode(command, mode):
    assert ActionPolicy(mode=mode).assess(call(command)).decision is Decision.DENY


@pytest.mark.parametrize("command,reason", [
    ("python - <<'PY'\nprint('PRIVATE_TOKEN')\nPY", "Heredocs and multiline"),
    ("echo PRIVATE_TOKEN\npwd", "Heredocs and multiline"),
    ("echo PRIVATE_TOKEN > out", "redirections"),
    ("echo 'PRIVATE_TOKEN", "quoting is invalid"),
    ("rm PRIVATE_TOKEN", "explicitly denied"),
])
def test_shell_denial_explains_reason_without_echoing_command(command, reason):
    result = ActionPolicy().assess(call(command))
    assert result.decision is Decision.DENY
    assert reason in result.reason
    assert "PRIVATE_TOKEN" not in result.reason