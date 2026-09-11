# Protocol and interaction

[Documentation index](../README.md)

The evaluation protocol is independent of the agent implementation.
It presents M training tasks followed by N test tasks, allowing `step` and `reset` during training and only `step` during testing.
Agents choose how to use their experience and the allowed calls; the protocol does not require an LLM, a conversation, or a simulator.

| Document | Use |
|---|---|
| [Overview](overview.md) | Intuitive introduction, recorded metrics, and launch/viewer examples; embedded first results are historical. |
| [Interaction driver](interaction-driver.md) | Shared execution of requests from the implemented model-based and model-free agents. |
| [Full design](design.md) | Detailed protocol decisions, implementation history, and operational behavior. |
| [Prompt review](../prompt-review/README.md) | Dated review of the coding agents' prompts. |

For current result selections, use the [results index](../uncertainty-results/INDEX.md).
For observations and uncertainty, see the [uncertainty guides](../uncertainty/README.md).
