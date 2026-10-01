# Review and implementation companion

Prepared September 30, 2026, for the research agent working on ContourMark.

1. [CGF research review](CGF_RESEARCH_REVIEW.md): evidence assessment, independent discussion of both architectures, priorities and experiment plan.
2. [Code companion](CODE_COMPANION.md): implementation demonstrations and unresolved research details for the difficult parts of that review.
3. [Reproduced findings](findings.json) and [reproduction script](reproduce_findings.py): observed counterexamples against commit `da841d1`.
4. [Executable examples](companion_examples.py) and [check results](companion-checks.json): 12 CPU checks passed during this session.
5. [Draft integration patch](proposed-fixes.patch): proposed changes only, **not applied or integration-tested**. `git apply --check` passed against the reviewed snapshot. Review carefully before applying to another agent's newer work.

The existing 90-test suite passed during the review. The companion is additive documentation/demo code; production implementation and experiments remain unchanged. No model was started and no training or downloads were performed in the timed follow-up. The request changed from implementation to a code companion during that follow-up; the early edits were archived in the patch and removed from production files.

The time-limited session stops after committing and pushing this package. Remaining tasks are research/implementation recommendations, not completed fixes. In particular, the visibility and distribution counterexamples still apply to the reviewed implementation.
