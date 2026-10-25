---
title: "Microsoft Agent Framework (Rust) — Evaluation"
description: "Evaluating agent-framework-rs agents and workflows: what of Python's evaluation module is ported (nothing, apart from the harness Judge), then rule-based checks, LLM-as-judge with Judge, rubric and pairwise scoring with ResponseFormat, golden datasets, regression suites on a scripted client, tool-use and multi-turn evaluation, workflow evaluation, cost and latency, and an evaluation gate for CI."
framework: "microsoft-agent-framework"
language: "rust"
sidebar:
  label: "Evaluation"
  order: 17
---

Python's Agent Framework has an evaluation module: `evaluate_agent` and `evaluate_workflow` run a set of queries, turn each result into an `EvalItem`, and apply checks such as `keyword_check` and `tool_call_args_match` through a `LocalEvaluator`, with Foundry's hosted evaluators as an option. **The Rust port has none of it.** Upstream marks evaluation experimental, and the port's `docs/feature-stages.md` lists it (`EVALS`) among the experimental features not ported yet, to go behind an `experimental-*` feature when it is. What the crate does have is everything an evaluation is built from: responses you can inspect and serialise, typed structured output, summed token usage, spans, the scripted-client pattern from the [testing page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_testing/#a-scripted-chat-client), and one evaluation type, the experimental harness's `Judge`.

This page shows how to evaluate with those pieces. Every program below was compiled against `agent-framework` %%version=agent-framework-rs%%, and run, either on its own test double or against a local stand-in for the OpenAI API (whose canned answers explain the outputs shown). The [recipes page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_recipes/#a-regression-suite-with-a-judge-model) has a complete evaluation suite in one program; this page takes each part of that further.

## What is and is not ported

| Python (`agent_framework`) | Rust %%version=agent-framework-rs%% |
| --- | --- |
| `evaluate_agent`, `evaluate_workflow` | Not ported. Loop over your cases calling `run_once` or `Workflow::run`. |
| `EvalItem`, `EvalCheck`, `CheckResult`, `Evaluator`, `LocalEvaluator` | Not ported. An `AgentResponse` already holds the messages, tool calls, results, usage and finish reason; a check is a closure over it. |
| `@evaluator` | No equivalent needed: checks are ordinary functions or closures. |
| `keyword_check`, `tool_called_check`, `tool_calls_present`, `tool_call_args_match`, `ExpectedToolCall` | Not ported. Each is a few lines over `Message::function_calls` and `FunctionCallContent::parse_arguments` ([below](#rule-based-checks)). |
| `ConversationSplit`, `ConversationSplitter`, `EvalItem.per_turn_items` | Not ported. Drive the conversation one turn at a time and check each reply ([multi-turn](#multi-turn-conversations)). |
| `responses=` (pre-recorded responses) | `AgentResponse` implements `Serialize` and `Deserialize`, so save responses and score them later ([golden datasets](#golden-datasets-and-recorded-runs)). |
| `num_repetitions` | A loop ([CI gate](#an-evaluation-gate-for-ci)). |
| `FoundryEvals`, `AgentEvalConverter` | Not ported; the Foundry crate has no evaluation client. |
| `EvalNotPassedError`, `raise_for_status` | Exit with a non-zero code, or `assert!` in a test. |
| LLM judge | `Judge` and `JudgeVerdict` in `agent_framework::harness`, behind `experimental-harness` ([below](#llm-as-judge-with-judge)); or any agent with a `ResponseFormat` ([rubric scoring](#rubric-and-pairwise-scoring)). |

.NET's `Microsoft.Extensions.AI.Evaluation` libraries have no Rust counterpart either.

## Rule-based checks

Rules are cheap, exact and repeatable, so they come first. The program below is the shape of Python's `LocalEvaluator` in a page of code: a case lists the query and the tool calls expected; running it produces an item holding the response and the time taken; each check maps an item to a pass or fail with a reason. `keyword_check`, `tool_called_check` and `tool_call_args_match` behave like their Python namesakes: keywords are matched case-insensitively, the argument match is a subset match (extra arguments pass) and is order-insensitive, and an expected call with no arguments checks only the name.

```rust
//! A small evaluation harness: cases, an item per run, and checks.
use std::time::{Duration, Instant};

use agent_framework::prelude::*;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};

/// One row of the test set.
struct EvalCase {
    name: &'static str,
    query: &'static str,
    /// Tool calls the agent should make: a name and, optionally, the
    /// arguments that must be present (extra arguments are allowed).
    expected_tool_calls: Vec<(&'static str, Option<Value>)>,
}

/// What one run produced, normalised for the checks.
struct EvalItem {
    response: AgentResponse,
    elapsed: Duration,
}

impl EvalItem {
    fn text(&self) -> String {
        self.response.text()
    }

    /// Every tool call the run made, in order, with parsed arguments.
    fn tool_calls(&self) -> Vec<(String, Value)> {
        self.response
            .messages
            .iter()
            .flat_map(|m| m.function_calls())
            .map(|c| {
                let args = c.parse_arguments().map(|m| json!(m)).unwrap_or(Value::Null);
                (c.name.clone(), args)
            })
            .collect()
    }
}

struct CheckResult {
    check: String,
    passed: bool,
    reason: String,
}

type Check = Box<dyn Fn(&EvalCase, &EvalItem) -> CheckResult + Send + Sync>;

fn result(check: &str, passed: bool, reason: impl Into<String>) -> CheckResult {
    CheckResult { check: check.into(), passed, reason: reason.into() }
}

/// Every keyword appears in the answer (case-insensitive).
fn keyword_check(keywords: &'static [&'static str]) -> Check {
    Box::new(move |_, item| {
        let text = item.text().to_lowercase();
        let missing: Vec<_> = keywords.iter().filter(|k| !text.contains(&k.to_lowercase())).collect();
        result("keyword", missing.is_empty(), format!("missing {missing:?}"))
    })
}

/// The named tools were called at least once each.
fn tool_called_check(names: &'static [&'static str]) -> Check {
    Box::new(move |_, item| {
        let called: Vec<String> = item.tool_calls().into_iter().map(|(n, _)| n).collect();
        let missing: Vec<_> = names.iter().filter(|n| !called.iter().any(|c| c == *n)).collect();
        result("tool_called", missing.is_empty(), format!("not called {missing:?}; called {called:?}"))
    })
}

/// Each expected call was made with (at least) the expected arguments.
/// Order-insensitive; extra calls and extra arguments are allowed.
fn tool_call_args_match() -> Check {
    Box::new(|case, item| {
        let calls = item.tool_calls();
        for (name, expected) in &case.expected_tool_calls {
            let matched = calls.iter().any(|(n, args)| {
                n == name
                    && match expected {
                        None => true,
                        Some(Value::Object(want)) => want.iter().all(|(k, v)| args.get(k) == Some(v)),
                        Some(other) => args == other,
                    }
            });
            if !matched {
                let want = expected.as_ref().map(Value::to_string).unwrap_or_default();
                return result("tool_args", false, format!("no call {name} matching {want}; got {calls:?}"));
            }
        }
        result("tool_args", true, "")
    })
}

/// The run finished within a time limit.
fn latency_check(limit: Duration) -> Check {
    Box::new(move |_, item| result("latency", item.elapsed <= limit, format!("{:?}", item.elapsed)))
}

#[derive(Deserialize, JsonSchema)]
struct WeatherArgs {
    /// The city to look up.
    city: String,
}

fn agent_under_test(client: impl ChatClient + 'static) -> Agent {
    let weather = FunctionTool::typed("get_weather", "Current weather for a city.", |args: WeatherArgs| async move {
        Ok(json!({ "city": args.city, "temperature_c": 18, "conditions": "cloudy" }))
    });
    Agent::builder(client)
        .name("weather")
        .instructions("Answer weather questions with the get_weather tool. Name the city in the answer.")
        .tool(weather.into_definition())
        .temperature(0.0)
        .build()
}

async fn evaluate_agent(agent: &Agent, cases: &[EvalCase], checks: &[Check]) -> Vec<(String, Vec<CheckResult>)> {
    let mut report = Vec::new();
    for case in cases {
        let started = Instant::now();
        let results = match agent.run_once(case.query).await {
            Ok(response) => {
                let item = EvalItem { response, elapsed: started.elapsed() };
                checks.iter().map(|check| check(case, &item)).collect()
            }
            Err(e) => vec![result("run", false, e.to_string())],
        };
        report.push((case.name.to_string(), results));
    }
    report
}

#[tokio::main]
async fn main() -> Result<()> {
    let agent = agent_under_test(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?);
    let cases = vec![
        EvalCase {
            name: "weather in Paris",
            query: "What's the weather in Paris?",
            expected_tool_calls: vec![("get_weather", Some(json!({ "city": "Paris" })))],
        },
        EvalCase { name: "any weather call", query: "Is it cold in Paris today?", expected_tool_calls: vec![("get_weather", None)] },
    ];
    let checks = vec![
        keyword_check(&["Paris"]),
        tool_called_check(&["get_weather"]),
        tool_call_args_match(),
        latency_check(Duration::from_secs(30)),
    ];

    let report = evaluate_agent(&agent, &cases, &checks).await;
    let mut failed = 0;
    for (name, results) in &report {
        let ok = results.iter().all(|r| r.passed);
        failed += usize::from(!ok);
        println!("{} {name}", if ok { "PASS" } else { "FAIL" });
        for r in results.iter().filter(|r| !r.passed) {
            println!("    {}: {}", r.check, r.reason);
        }
    }
    println!("{}/{} cases passed", report.len() - failed, report.len());
    if failed > 0 {
        std::process::exit(1);
    }
    Ok(())
}
```

```text
PASS weather in Paris
PASS any weather call
2/2 cases passed
```

The tool calls come from the response, not from a log: the tool loop returns every message of the run, so `response.messages` holds each `FunctionCall` the model made and each `FunctionResult` it got back, in order, followed by the answer. `parse_arguments` turns the call's arguments into a map whether the provider sent a JSON string or an object, and fails on malformed JSON, which a check should count as a failure rather than an empty map. Two limits of reading calls from the response: a call the loop refused (an unknown tool, unparseable arguments) is still there, with an error result, and a call answered from a cache by middleware looks like any other. To know what actually ran, record it in function middleware, as in [tool use](#evaluating-tool-use).

Other rules worth having, each a line or two over `AgentResponse`: `finish_reason` is `stop` rather than `length` or `content_filter` (a cut-off answer can read as complete); `has_refusal()` is false; `parse_json::<T>()` succeeds for structured output; the answer stays under a length; `usage_details` stays under a token budget.

## LLM-as-judge with `Judge`

Rules cannot judge tone, refusals or whether an answer is complete; a second model can. The crate's one evaluation type is `Judge`, from the experimental agent harness. It exists to drive `LoopAgent::with_judge`, which re-runs an agent until a judge says the request is answered (see [a judge](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_context_and_memory/#a-judge)), but its `verdict` method can be called on its own. It takes a `LoopContext`, whose fields are all public, and reads only two of them: `original_messages` (the request) and `last_result` (the agent's response).

```toml
[dependencies]
agent-framework = { version = "%%version=agent-framework-rs%%", features = ["experimental-harness"] }
```

```rust
//! LLM-as-judge with the harness's Judge, used outside a loop.
//! Needs the `experimental-harness` feature.
use std::sync::Arc;

use agent_framework::harness::{Judge, JudgeVerdict, LoopContext};
use agent_framework::prelude::*;

/// Ask `judge` whether `response` fully answers `input`.
async fn judge_once(judge: &Judge, input: Vec<Message>, response: AgentResponse) -> Result<JudgeVerdict> {
    // Judge::verdict reads only `original_messages` and `last_result`;
    // the other fields exist for LoopAgent's callbacks.
    let ctx = LoopContext {
        iteration: 1,
        last_result: response,
        messages: input.clone(),
        original_messages: input,
        session: AgentSession::new(),
        progress: Vec::new(),
        feedback: None,
    };
    judge.verdict(&ctx).await
}

#[tokio::main]
async fn main() -> Result<()> {
    let agent = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?)
        .instructions("You answer questions about the company's leave policy.")
        .build();

    // A different, stronger model judges. Any ChatClient works.
    let judge_client: Arc<dyn ChatClient> = Arc::new(OpenAIChatCompletionClient::from_env("%%model=openai-flagship%%")?);
    let judge = Judge::new(judge_client).criteria([
        "States how many days of annual leave a full-time employee gets.",
        "Does not invent policy details that were not asked for.",
    ]);

    let questions = ["How much annual leave do I get?", "Can I carry leave over to next year?"];
    let mut answered = 0;
    for question in questions {
        let input = vec![Message::user(question)];
        let response = agent.run_once(input.clone()).await?;
        let verdict = judge_once(&judge, input, response).await?;
        answered += usize::from(verdict.answered);
        println!("{question}\n  answered: {}  reasoning: {}", verdict.answered, verdict.reasoning);
    }
    println!("{answered}/{} answered", questions.len());

    // What the judge is told, criteria included.
    println!("--- judge instructions ---\n{}", judge.rendered_instructions());
    Ok(())
}
```

```text
How much annual leave do I get?
  answered: true  reasoning: example reasoning
Can I carry leave over to next year?
  answered: true  reasoning: example reasoning
2/2 answered
--- judge instructions ---
You are an evaluator. You are given a user's original request and an agent's latest response. Decide whether the agent has fully addressed the original request. Set 'answered' to true if the request has been fully addressed, or false if more work is still required, and use 'reasoning' to briefly justify your decision. If you cannot return structured output, end your reply with a line reading exactly 'VERDICT: DONE' when the request has been fully addressed or 'VERDICT: MORE' when more work is still required.

The response must satisfy all of the following criteria:
- States how many days of annual leave a full-time employee gets.
- Does not invent policy details that were not asked for.
```

`Judge::new(client)` calls the client directly, with no tools, session or middleware: a system message with the judge instructions, then the original request, the agent's latest response and the question "Has the original request been fully addressed?". It asks for a `JudgeVerdict { answered: bool, reasoning: String }` through a strict JSON schema (`JudgeVerdict::response_format()`). The reply is read in this order:

1. a `verdict_parser(f)` you supplied, whose errors are returned as they are;
2. the response's parsed `value`;
3. the reply text parsed as a `JudgeVerdict`;
4. the markers `VERDICT: MORE` and `VERDICT: DONE` (`JUDGE_VERDICT_MORE`, `JUDGE_VERDICT_DONE`): `answered` is true only if `DONE` appears and `MORE` does not, and the whole text becomes the reasoning.

So a judge whose reply cannot be read counts as "not answered", which is the safe direction for an evaluation. `criteria([...])` adds a bullet list to the instructions at the `{{criteria}}` placeholder (`CRITERIA_PLACEHOLDER`); `instructions(text)` replaces `DEFAULT_JUDGE_INSTRUCTIONS`, and without the placeholder the criteria are dropped; `rendered_instructions()` shows the result. `response_format(None)` sends no format, for a provider that rejects strict schemas, and then the markers do the work. The verdict is binary. For scores, use your own schema, as in the next section.

Judges have known biases. Use a different, preferably stronger, model from the one under test (a model tends to favour answers in its own style); keep the temperature at zero where the provider allows it; give the judge the reference material it should check against; and spot-check its verdicts by hand before trusting a pass rate. The judge sees the whole request and response, so it is sent whatever they contain.

## Rubric and pairwise scoring

For more than pass or fail, give a judge agent a `ResponseFormat` with one field per criterion and read the reply with `parse_json`. A strict schema (`strict: Some(true)` on `ResponseFormat::JsonSchema`, with `additionalProperties: false` and every property required) makes OpenAI-family models keep to it; other providers treat it as guidance, and Anthropic folds it into the prompt, so a parse failure must count as a failed evaluation. The program below scores two variants of an agent on four criteria, combines them with weights, and compares the two directly.

```rust
//! Structured scoring with ResponseFormat: a multi-criterion rubric, and a
//! pairwise comparison of two agents run in both orders.
use agent_framework::prelude::*;
use serde::Deserialize;
use serde_json::json;

/// One score per criterion, 1 (poor) to 5 (excellent).
#[derive(Deserialize)]
struct RubricScore {
    relevance: u8,
    groundedness: u8,
    completeness: u8,
    concision: u8,
    rationale: String,
}

impl RubricScore {
    /// Weighted mean on a 0-1 scale. Groundedness counts double.
    fn weighted(&self) -> f64 {
        let parts = [(self.relevance, 1.0), (self.groundedness, 2.0), (self.completeness, 1.0), (self.concision, 0.5)];
        let total: f64 = parts.iter().map(|(_, w)| w).sum();
        parts.iter().map(|(s, w)| (f64::from(*s).clamp(1.0, 5.0) - 1.0) / 4.0 * w).sum::<f64>() / total
    }
}

#[derive(Deserialize)]
struct Preference {
    winner: String,
    reason: String,
}

fn score_1_to_5(description: &str) -> serde_json::Value {
    json!({ "type": "integer", "minimum": 1, "maximum": 5, "description": description })
}

fn rubric_judge(client: impl ChatClient + 'static) -> Agent {
    let schema = json!({
        "type": "object",
        "properties": {
            "relevance": score_1_to_5("Does the answer address the question asked?"),
            "groundedness": score_1_to_5("Is every claim supported by the reference material?"),
            "completeness": score_1_to_5("Does it cover everything the question needs?"),
            "concision": score_1_to_5("Is it free of padding?"),
            "rationale": { "type": "string" }
        },
        "required": ["relevance", "groundedness", "completeness", "concision", "rationale"],
        "additionalProperties": false
    });
    Agent::builder(client)
        .instructions("You grade answers against reference material. Score each criterion from 1 to 5. Be strict: 5 means nothing to improve.")
        .response_format(ResponseFormat::JsonSchema { name: "RubricScore".into(), description: None, schema, strict: Some(true) })
        .temperature(0.0)
        .build()
}

fn pairwise_judge(client: impl ChatClient + 'static) -> Agent {
    let schema = json!({
        "type": "object",
        "properties": { "winner": { "type": "string", "enum": ["A", "B", "tie"] }, "reason": { "type": "string" } },
        "required": ["winner", "reason"],
        "additionalProperties": false
    });
    Agent::builder(client)
        .instructions("Compare two answers to the same question. Pick the better one, or tie. Ignore length and order.")
        .response_format(ResponseFormat::json_schema("Preference", schema))
        .temperature(0.0)
        .build()
}

/// Ask the judge twice with the answers swapped; count a win only when both
/// orders agree, which cancels the judge's preference for a position.
async fn compare(judge: &Agent, question: &str, a: &str, b: &str) -> Result<&'static str> {
    let ask = |first: &str, second: &str| format!("Question: {question}\n\nAnswer A:\n{first}\n\nAnswer B:\n{second}");
    let forward: Preference = judge.run_once(ask(a, b)).await?.parse_json()?;
    let backward: Preference = judge.run_once(ask(b, a)).await?.parse_json()?;
    println!("    forward: {} ({}); swapped: {} ({})", forward.winner, forward.reason, backward.winner, backward.reason);
    Ok(match (forward.winner.as_str(), backward.winner.as_str()) {
        ("A", "B") => "baseline",
        ("B", "A") => "candidate",
        _ => "tie",
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    let model = || OpenAIChatCompletionClient::from_env("%%model=openai-small%%");
    let baseline = Agent::builder(model()?).instructions("Answer in one sentence.").build();
    let candidate = Agent::builder(model()?).instructions("Answer briefly, quoting the reference.").build();
    let judge_client = OpenAIChatCompletionClient::from_env("%%model=openai-flagship%%")?;
    let rubric = rubric_judge(judge_client.clone());
    let pairwise = pairwise_judge(judge_client);

    let reference = "Employees get 25 days of annual leave. Up to 5 unused days carry over to the next year.";
    let question = "Can I carry over unused leave?";
    let prompt = format!("Reference: {reference}\n\nQuestion: {question}");

    let answer_a = baseline.run_once(prompt.as_str()).await?.text();
    let answer_b = candidate.run_once(prompt.as_str()).await?.text();

    for (name, answer) in [("baseline", &answer_a), ("candidate", &answer_b)] {
        let grading = format!("Reference: {reference}\n\nQuestion: {question}\n\nAnswer: {answer}");
        let score: RubricScore = rubric.run_once(grading).await?.parse_json()?;
        println!(
            "{name}: relevance {} groundedness {} completeness {} concision {} -> {:.2}\n    {}",
            score.relevance, score.groundedness, score.completeness, score.concision, score.weighted(), score.rationale
        );
    }
    println!("pairwise: {}", compare(&pairwise, question, &answer_a, &answer_b).await?);
    Ok(())
}
```

```text
baseline: relevance 3 groundedness 3 completeness 3 concision 3 -> 0.50
    example rationale
candidate: relevance 3 groundedness 3 completeness 3 concision 3 -> 0.50
    example rationale
    forward: A (example reason); swapped: A (example reason)
pairwise: tie
```

The stand-in fills every integer field with 3 and every enum with its first value, which shows two things a real judge does too. Absolute scores bunch in the middle of the scale, so prefer comparisons between variants to thresholds on a single score. And a judge prefers a position: asked twice with the answers swapped, the stand-in chose "A" both times, which `compare` reports as a tie rather than a win. Count a win only when both orders agree. Clamp scores before using them (`minimum` and `maximum` in a schema are not enforced by every provider), and keep the rationale: it is what you read when a score looks wrong.

## Golden datasets and recorded runs

A golden dataset is a file of cases with what a good answer must contain or do, kept under version control beside the prompts. JSON Lines suits it: one case per line, easy to add to and to diff. The program below loads one, runs the agent once over it, saves exactly what came back, and scores the saved run. Scoring a saved run is Rust's counterpart of Python's `responses=`: it makes no model calls, so you can change the checks and score again, or score transcripts captured from production.

```rust
//! A golden dataset in JSON Lines: run the agent once, save the responses,
//! and score the saved run offline.
use std::path::Path;

use agent_framework::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// One line of `golden.jsonl`.
#[derive(Debug, Deserialize)]
struct GoldenCase {
    id: String,
    input: String,
    /// Substrings the answer must contain (case-insensitive).
    #[serde(default)]
    must_contain: Vec<String>,
    /// Substrings the answer must not contain.
    #[serde(default)]
    must_not_contain: Vec<String>,
    /// Tool calls, in order, that the agent must make (a subsequence:
    /// other calls may come between them). Arguments are a subset match.
    #[serde(default)]
    tool_calls: Vec<ExpectedCall>,
}

#[derive(Debug, Deserialize)]
struct ExpectedCall {
    name: String,
    #[serde(default)]
    arguments: Option<serde_json::Map<String, Value>>,
}

/// One saved run: the case id and exactly what the agent returned.
#[derive(Serialize, Deserialize)]
struct Recorded {
    id: String,
    response: AgentResponse,
}

const GOLDEN: &str = r#"
{"id": "inv-1", "input": "What is the status of invoice INV-7?", "must_contain": ["INV-7"], "tool_calls": [{"name": "lookup_invoice", "arguments": {"invoice_id": "INV-7"}}]}
{"id": "inv-2", "input": "Summarise invoice INV-9 for me.", "must_contain": ["INV-9"], "must_not_contain": ["I cannot"], "tool_calls": [{"name": "lookup_invoice"}]}
"#;

fn load_golden(path: &Path) -> Result<Vec<GoldenCase>> {
    let text = std::fs::read_to_string(path).map_err(Error::other)?;
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).map_err(Error::from))
        .collect()
}

fn agent_under_test(client: impl ChatClient + 'static) -> Agent {
    let lookup = FunctionTool::new(
        "lookup_invoice",
        "Look up an invoice by id.",
        json!({ "type": "object", "properties": { "invoice_id": { "type": "string" } }, "required": ["invoice_id"] }),
        |args| async move { Ok(json!({ "invoice_id": args["invoice_id"], "status": "paid", "amount": 120.0 })) },
    );
    Agent::builder(client)
        .instructions("You answer questions about invoices. Always look the invoice up first and quote its id.")
        .tool(lookup.into_definition())
        .temperature(0.0)
        .build()
}

/// Score one recorded response against its golden case.
fn score(case: &GoldenCase, response: &AgentResponse) -> Vec<String> {
    let mut problems = Vec::new();
    let answer = response.text().to_lowercase();
    for s in &case.must_contain {
        if !answer.contains(&s.to_lowercase()) {
            problems.push(format!("missing {s:?}"));
        }
    }
    for s in &case.must_not_contain {
        if answer.contains(&s.to_lowercase()) {
            problems.push(format!("contains {s:?}"));
        }
    }
    // Ordered subsequence match over the calls actually made.
    let calls: Vec<_> = response.messages.iter().flat_map(|m| m.function_calls()).collect();
    let mut next = calls.iter();
    for want in &case.tool_calls {
        let found = next.any(|call| {
            call.name == want.name
                && want.arguments.as_ref().is_none_or(|expected| {
                    let actual = call.parse_arguments().unwrap_or_default();
                    expected.iter().all(|(k, v)| actual.get(k) == Some(v))
                })
        });
        if !found {
            let args = want.arguments.clone().map(|a| Value::Object(a).to_string()).unwrap_or_default();
            problems.push(format!("expected call {} {args} not found in order", want.name));
        }
    }
    problems
}

#[tokio::main]
async fn main() -> Result<()> {
    let dir = std::env::temp_dir().join("eval-golden");
    std::fs::create_dir_all(&dir).map_err(Error::other)?;
    let golden_path = dir.join("golden.jsonl");
    std::fs::write(&golden_path, GOLDEN.trim_start()).map_err(Error::other)?;
    let cases = load_golden(&golden_path)?;

    // 1. Run once against the model and save exactly what came back.
    let agent = agent_under_test(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?);
    let mut recorded = Vec::new();
    for case in &cases {
        let response = agent.run_once(case.input.as_str()).await?;
        recorded.push(Recorded { id: case.id.clone(), response });
    }
    let run_path = dir.join("run.json");
    std::fs::write(&run_path, serde_json::to_string_pretty(&recorded)?).map_err(Error::other)?;

    // 2. Score the saved run. This step makes no model calls, so it can be
    //    rerun as the checks change, or on a transcript from production.
    let saved: Vec<Recorded> = serde_json::from_str(&std::fs::read_to_string(&run_path).map_err(Error::other)?)?;
    let mut failures = 0;
    for case in &cases {
        let Some(run) = saved.iter().find(|r| r.id == case.id) else {
            println!("FAIL {}: no recorded response", case.id);
            failures += 1;
            continue;
        };
        let problems = score(case, &run.response);
        if problems.is_empty() {
            println!("PASS {}", case.id);
        } else {
            failures += 1;
            println!("FAIL {}: {}", case.id, problems.join("; "));
        }
    }
    println!("saved run: {}", run_path.display());
    std::process::exit(i32::from(failures > 0));
}
```

```text
FAIL inv-1: missing "INV-7"; expected call lookup_invoice {"invoice_id":"INV-7"} not found in order
FAIL inv-2: missing "INV-9"
saved run: /tmp/eval-golden/run.json
```

Both cases fail against the stand-in, which is the point: it answers with whatever arguments it makes up (here `"invoice_id": "example invoice_id"`), so the expected call and the quoted id are missing. Against a real model the same file passes or fails on the model's behaviour. Expected tool calls here are an ordered subsequence, a stricter match than Python's order-insensitive `tool_call_args_match`: use it when the order matters (look up before answering) and the set match from the [first program](#rule-based-checks) when it does not.

Keep each saved run with the model, the prompt version and the date, and the golden file becomes a history: you can rescore last month's run with this month's checks to see whether a change in the score came from the model or from the checks.

## Regression suites on a scripted client

A real model is the only way to evaluate the model's choices, but most regressions are in your own code: a tool that changed its output, an instruction edited by accident, middleware that drops a result. Those are cheaper to catch with recorded model replies played back through a scripted client, so the suite needs no key, costs nothing and gives the same answer every time. Record a reply from a real run once (the `messages` of a saved `ChatResponse`), script it, and assert on what your code did with it.

```rust
//! A regression suite on recorded model replies: replay them through a
//! scripted client, then check the agent's own behaviour and the requests
//! it sent, including a snapshot of its instructions and tool schemas.
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use agent_framework::prelude::*;
use agent_framework::types::FunctionArguments;
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};

/// Replays replies in order and records each request.
#[derive(Clone, Default)]
struct ScriptedClient {
    replies: Arc<Mutex<VecDeque<ChatResponse>>>,
    requests: Arc<Mutex<Vec<(Vec<Message>, ChatOptions)>>>,
}

impl ScriptedClient {
    fn text(self, text: &str) -> Self {
        self.replies.lock().unwrap().push_back(ChatResponse::from_text(text));
        self
    }

    fn tool_call(self, call_id: &str, name: &str, args: Value) -> Self {
        let call = FunctionCallContent::new(call_id, name, Some(FunctionArguments::Raw(args.to_string())));
        self.replies.lock().unwrap().push_back(ChatResponse {
            messages: vec![Message::with_contents(Role::assistant(), vec![Content::FunctionCall(call)])],
            finish_reason: Some(FinishReason::tool_calls()),
            ..Default::default()
        });
        self
    }

    fn requests(&self) -> Vec<(Vec<Message>, ChatOptions)> {
        self.requests.lock().unwrap().clone()
    }

    fn remaining(&self) -> usize {
        self.replies.lock().unwrap().len()
    }
}

#[async_trait]
impl ChatClient for ScriptedClient {
    async fn get_response(&self, messages: Vec<Message>, options: ChatOptions) -> Result<ChatResponse> {
        self.requests.lock().unwrap().push((messages, options));
        let next = self.replies.lock().unwrap().pop_front();
        next.ok_or_else(|| Error::other("the script has no more replies"))
    }

    async fn get_streaming_response(&self, messages: Vec<Message>, options: ChatOptions) -> Result<ChatStream> {
        let response = self.get_response(messages, options).await?;
        let updates: Vec<Result<ChatResponseUpdate>> = response
            .messages
            .into_iter()
            .map(|m| Ok(ChatResponseUpdate { contents: m.contents, role: Some(m.role), ..Default::default() }))
            .collect();
        Ok(futures::stream::iter(updates).boxed())
    }
}

/// The production constructor, shared with the application.
fn build_agent(client: impl ChatClient + 'static) -> Agent {
    let refund = FunctionTool::new(
        "issue_refund",
        "Refund an order. Only for orders under 100 GBP.",
        json!({ "type": "object", "properties": { "order": { "type": "string" }, "amount": { "type": "number" } }, "required": ["order", "amount"] }),
        |args| async move {
            let amount = args["amount"].as_f64().unwrap_or(0.0);
            if amount >= 100.0 {
                return Err(Error::tool("refunds of 100 GBP or more need a manager"));
            }
            Ok(json!({ "refunded": amount, "order": args["order"] }))
        },
    );
    Agent::builder(client)
        .instructions("You handle refunds. Refund orders under 100 GBP; refer anything larger to a manager.")
        .tool(refund.into_definition())
        .build()
}

/// What the agent sends that a prompt or tool edit would change.
fn request_snapshot(messages: &[Message], options: &ChatOptions) -> Value {
    json!({
        "system": messages.iter().filter(|m| m.role == Role::system()).map(Message::text).collect::<Vec<_>>(),
        "tools": options.tools.iter().map(|t| json!({ "name": t.name, "description": t.description, "parameters": t.parameters })).collect::<Vec<_>>(),
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut failures = Vec::new();

    // Case 1: a small refund goes through.
    let client = ScriptedClient::default()
        .tool_call("c1", "issue_refund", json!({ "order": "A1", "amount": 40.0 }))
        .text("I've refunded 40 GBP for order A1.");
    let response = build_agent(client.clone()).run_once("Refund 40 GBP on order A1").await?;
    let results: Vec<_> = response.messages.iter().flat_map(|m| m.function_results()).collect();
    if results.len() != 1 || results[0].result.as_ref().map(|r| r["refunded"] == json!(40.0)) != Some(true) {
        failures.push(format!("case 1: unexpected tool results {results:?}"));
    }

    // Case 2: the tool refuses a large refund; the error reaches the model
    // as a tool result, and the model's second request carries it.
    let client2 = ScriptedClient::default()
        .tool_call("c1", "issue_refund", json!({ "order": "B2", "amount": 250.0 }))
        .text("That refund needs a manager; I've passed it on.");
    let response2 = build_agent(client2.clone()).run_once("Refund 250 GBP on order B2").await?;
    let errored = response2.messages.iter().flat_map(|m| m.function_results()).any(|r| r.exception.is_some());
    if !errored {
        failures.push("case 2: expected the tool to fail".into());
    }
    if !response2.text().contains("manager") {
        failures.push(format!("case 2: answer was {:?}", response2.text()));
    }
    let second_request = &client2.requests()[1].0;
    if !second_request.iter().any(|m| !m.function_results().is_empty()) {
        failures.push("case 2: the tool result was not sent back to the model".into());
    }

    // Every reply in each script was used, and no more.
    for (name, c) in [("case 1", &client), ("case 2", &client2)] {
        if c.remaining() != 0 {
            failures.push(format!("{name}: {} scripted replies unused", c.remaining()));
        }
    }

    // Snapshot: compare the first request with the stored snapshot. Set
    // UPDATE_SNAPSHOTS=1 to accept a deliberate change.
    let (messages, options) = &client.requests()[0];
    let snapshot = serde_json::to_string_pretty(&request_snapshot(messages, options))?;
    let path = std::env::temp_dir().join("refund-agent.snapshot.json");
    match std::fs::read_to_string(&path) {
        Ok(stored) if std::env::var("UPDATE_SNAPSHOTS").is_err() => {
            if stored != snapshot {
                failures.push(format!("prompt snapshot changed; diff {} against the new request", path.display()));
            }
        }
        _ => std::fs::write(&path, &snapshot).map_err(Error::other)?,
    }

    if failures.is_empty() {
        println!("all regression checks passed");
    } else {
        for f in &failures {
            println!("FAIL {f}");
        }
        std::process::exit(1);
    }
    Ok(())
}
```

```text
all regression checks passed
```

Three kinds of assertion appear. Behavioural: given these model replies, the tool ran and returned this, and a refused refund reached the model as an error result (`exception` is set on the `FunctionResultContent`; the text the model sees is "tool execution failed" unless `include_detailed_errors` is on, or the tool returns `Error::tool_rejected`, whose message is passed on as it is). Protocol: the second request carried the tool result back, and the script was used exactly (`remaining() == 0`), so the agent made neither more nor fewer model calls than recorded. And a snapshot of what the agent sends that a prompt edit would change, the system message and the tool names, descriptions and schemas, compared with a stored copy: a changed snapshot fails the run until someone accepts it with `UPDATE_SNAPSHOTS=1`, which turns a silent prompt change into a reviewed one. Keep the snapshot file in the repository rather than in a temporary directory, as the program does only so that it can run anywhere. The [testing page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_testing/#a-scripted-chat-client) has a fuller `ScriptedClient` with usage, delays, streaming and request inspection.

A scripted suite cannot tell you that the model will still call the tool after a prompt change; pair it with a small live evaluation for that.

## Evaluating tool use

For an agent whose job is to act, the answer matters less than the trajectory: which tools ran, in what order, with what arguments, and whether anything that needed approval waited for it. Function middleware sees every call that actually runs, with its arguments as the tool received them and its result, so it makes a reliable recorder. The program below books a flight across an approval round trip and checks six things about how it got there.

```rust
//! Evaluating tool use: the trajectory of calls, their arguments, what ran,
//! and the approval round trip.
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use agent_framework::middleware::FnMiddleware;
use agent_framework::prelude::*;
use agent_framework::types::FunctionArguments;
use async_trait::async_trait;
use serde_json::{json, Value};

/// Replays scripted replies in order (see the testing page for a fuller one).
#[derive(Clone, Default)]
struct ScriptedClient(Arc<Mutex<VecDeque<ChatResponse>>>);

impl ScriptedClient {
    fn tool_call(self, name: &str, args: Value) -> Self {
        let id = format!("call_{}", self.0.lock().unwrap().len());
        let call = FunctionCallContent::new(id, name, Some(FunctionArguments::Raw(args.to_string())));
        self.0.lock().unwrap().push_back(ChatResponse {
            messages: vec![Message::with_contents(Role::assistant(), vec![Content::FunctionCall(call)])],
            finish_reason: Some(FinishReason::tool_calls()),
            ..Default::default()
        });
        self
    }
    fn text(self, text: &str) -> Self {
        self.0.lock().unwrap().push_back(ChatResponse::from_text(text));
        self
    }
}

#[async_trait]
impl ChatClient for ScriptedClient {
    async fn get_response(&self, _m: Vec<Message>, _o: ChatOptions) -> Result<ChatResponse> {
        let next = self.0.lock().unwrap().pop_front();
        next.ok_or_else(|| Error::other("the script has no more replies"))
    }
    async fn get_streaming_response(&self, _m: Vec<Message>, _o: ChatOptions) -> Result<ChatStream> {
        Err(Error::other("not used"))
    }
}

/// A tool call that actually ran, as function middleware saw it.
#[derive(Debug, Clone)]
struct Executed {
    name: String,
    arguments: Value,
    result: std::result::Result<Value, String>,
}

fn build_agent(client: impl ChatClient + 'static, log: Arc<Mutex<Vec<Executed>>>) -> Agent {
    let search = FunctionTool::new(
        "search_flights",
        "Search flights between two airports on a date.",
        json!({ "type": "object", "properties": { "from": { "type": "string" }, "to": { "type": "string" }, "date": { "type": "string" } }, "required": ["from", "to", "date"] }),
        |_| async move { Ok(json!([{ "flight": "BA117", "price": 412 }, { "flight": "VS3", "price": 398 }])) },
    );
    let book = FunctionTool::new(
        "book_flight",
        "Book a flight by number.",
        json!({ "type": "object", "properties": { "flight": { "type": "string" } }, "required": ["flight"] }),
        |args| async move { Ok(json!({ "booked": args["flight"], "reference": "XK42QP" })) },
    )
    .with_approval_mode(ApprovalMode::AlwaysRequire);

    // Records every call that ran, with its outcome.
    let recorder = FnMiddleware::new(move |ctx: FunctionInvocationContext, next: Next<FunctionInvocationContext>| {
        let log = log.clone();
        async move {
            let (name, arguments) = (ctx.function_name.clone(), ctx.arguments.clone());
            let outcome = next.run(ctx).await;
            let result = match &outcome {
                Ok(ctx) => Ok(ctx.result.clone().unwrap_or(Value::Null)),
                Err(e) => Err(e.to_string()),
            };
            log.lock().unwrap().push(Executed { name, arguments, result });
            outcome
        }
    });

    Agent::builder(client)
        .instructions("You book flights. Search first, then book the cheapest direct flight.")
        .tool(search.into_definition())
        .tool(book.into_definition())
        .function_middleware(Arc::new(recorder))
        .build()
}

#[tokio::main]
async fn main() -> Result<()> {
    let client = ScriptedClient::default()
        .tool_call("search_flights", json!({ "from": "LHR", "to": "JFK", "date": "2026-11-02" }))
        .tool_call("book_flight", json!({ "flight": "VS3" }))
        .text("Booked VS3, reference XK42QP.");
    let log = Arc::new(Mutex::new(Vec::new()));
    let agent = build_agent(client, log.clone());

    let mut session = agent.create_session();
    let mut transcript: Vec<Message> = Vec::new();
    let mut approvals_asked = Vec::new();
    let mut input = vec![Message::user("Book me the cheapest flight from LHR to JFK on 2 November.")];
    let answer = loop {
        let response = agent.run(input, Some(&mut session)).await?;
        transcript.extend(response.messages.clone());
        let requests = response.user_input_requests();
        if requests.is_empty() {
            break response.text();
        }
        // The evaluation approves everything, but records what was asked
        // and checks that nothing ran before the answer.
        let ran_before: Vec<String> = log.lock().unwrap().iter().map(|e| e.name.clone()).collect();
        let mut answers = Vec::new();
        for request in requests {
            approvals_asked.push((request.function_call.name.clone(), ran_before.clone()));
            answers.push(Content::FunctionApprovalResponse(request.create_response(true)));
        }
        input = vec![Message::with_contents(Role::user(), answers)];
    };

    let executed = log.lock().unwrap().clone();
    let mut failures = Vec::new();

    // 1. Trajectory: exactly these tools ran, in this order.
    let order: Vec<&str> = executed.iter().map(|e| e.name.as_str()).collect();
    if order != ["search_flights", "book_flight"] {
        failures.push(format!("trajectory {order:?}"));
    }
    // 2. Arguments of the first call.
    if executed.first().map(|e| &e.arguments["to"]) != Some(&json!("JFK")) {
        failures.push("search_flights was not called with to=JFK".into());
    }
    // 3. Grounded arguments: the flight booked came from a search result,
    //    and it was the cheapest one.
    let results: Vec<Value> = executed.iter().filter(|e| e.name == "search_flights").filter_map(|e| e.result.clone().ok()).collect();
    let cheapest = results.iter().flat_map(|r| r.as_array().cloned().unwrap_or_default()).min_by_key(|f| f["price"].as_i64());
    let booked = executed.iter().find(|e| e.name == "book_flight").map(|e| e.arguments["flight"].clone());
    if booked != cheapest.map(|f| f["flight"].clone()) {
        failures.push(format!("booked {booked:?}, cheapest was another flight"));
    }
    // 4. Approval: asked for book_flight, and it had not run yet.
    match approvals_asked.as_slice() {
        [(name, ran)] if name == "book_flight" && !ran.contains(name) => {}
        other => failures.push(format!("approvals {other:?}")),
    }
    // 5. No tool failed, and the model was not told about failures.
    if executed.iter().any(|e| e.result.is_err())
        || transcript.iter().flat_map(|m| m.function_results()).any(|r| r.exception.is_some())
    {
        failures.push("a tool call failed".into());
    }
    // 6. The answer quotes the booking reference from the tool result.
    if !answer.contains("XK42QP") {
        failures.push(format!("answer {answer:?} lacks the reference"));
    }

    println!("executed: {order:?}\nanswer: {answer}");
    if failures.is_empty() {
        println!("PASS");
        Ok(())
    } else {
        failures.iter().for_each(|f| println!("FAIL {f}"));
        std::process::exit(1);
    }
}
```

```text
executed: ["search_flights", "book_flight"]
answer: Booked VS3, reference XK42QP.
PASS
```

The checks:

1. **Trajectory.** The exact list of tools that ran, in order. Use an exact match for a fixed procedure and a subsequence or set match where the model may reasonably add steps.
2. **Arguments.** Values the user gave must arrive in the call.
3. **Grounded arguments.** The value booked must come from an earlier tool result, not from the model's imagination, and must be the right one of them (the cheapest). This catches a model that books a flight it never searched for.
4. **Approval.** The approval request named `book_flight`, and the recorder had not seen `book_flight` when it was raised. The tool loop guarantees this for `ApprovalMode::AlwaysRequire`; the check guards against someone changing the tool's approval mode.
5. **No failures.** No call errored, as seen both by the recorder and in the error results the model received.
6. **Use of results.** The answer quotes the booking reference the tool returned.

The middleware records calls that ran; `response.messages` records calls the model asked for. Comparing the two finds calls that were requested but never ran: an unknown tool name, unparseable arguments, a rejected approval, or a tool call the budget stopped (see [the three limits](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_tools/#the-three-limits)). For an agent that may run calls in parallel, the recorder's order is completion order; set `allow_concurrent_invocation: false` in `FunctionInvocationConfig` if the evaluation needs a fixed order.

## Multi-turn conversations

Python splits a recorded conversation into per-turn items. In Rust, drive the conversation one turn at a time on a session and check every reply, which also tests that history reaches the model. The stand-in below remembers a name only if earlier turns are in the request, so running the same script without a session shows what a memory regression looks like.

```rust
//! Multi-turn evaluation: drive a scripted conversation through one session
//! and check every turn, not only the last.
use agent_framework::prelude::*;
use async_trait::async_trait;
use futures::StreamExt;

/// A stand-in that remembers a name if one was given earlier in the
/// messages it receives. Replace with a real client for a real evaluation.
#[derive(Clone)]
struct StandIn;

#[async_trait]
impl ChatClient for StandIn {
    async fn get_response(&self, messages: Vec<Message>, _o: ChatOptions) -> Result<ChatResponse> {
        let users: Vec<String> = messages.iter().filter(|m| m.role == Role::user()).map(Message::text).collect();
        let last = users.last().cloned().unwrap_or_default();
        let name = users.iter().find_map(|t| t.strip_prefix("My name is ").map(|n| n.trim_end_matches('.').to_string()));
        let reply = match (last.contains("my name"), name) {
            (true, Some(name)) => format!("Your name is {name}."),
            (true, None) => "I don't know your name.".to_string(),
            _ => "Noted.".to_string(),
        };
        Ok(ChatResponse::from_text(reply))
    }
    async fn get_streaming_response(&self, m: Vec<Message>, o: ChatOptions) -> Result<ChatStream> {
        let r = self.get_response(m, o).await?;
        Ok(futures::stream::iter(vec![Ok(ChatResponseUpdate::text(r.text()))]).boxed())
    }
}

/// A user turn and what the reply to it must (or must not) contain.
struct Turn {
    user: &'static str,
    expect: Option<&'static str>,
    forbid: Option<&'static str>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let agent = Agent::builder(StandIn).instructions("You are a helpful assistant.").build();
    let script = [
        Turn { user: "My name is Ada.", expect: None, forbid: None },
        Turn { user: "I'm planning a trip to Lisbon.", expect: None, forbid: None },
        Turn { user: "What is my name?", expect: Some("Ada"), forbid: Some("don't know") },
    ];

    // The same script with and without a session: the second shows what a
    // memory regression looks like.
    for with_session in [true, false] {
        let mut session = agent.create_session();
        let mut failures = 0;
        for (i, turn) in script.iter().enumerate() {
            let response = if with_session {
                agent.run(vec![Message::user(turn.user)], Some(&mut session)).await?
            } else {
                agent.run_once(turn.user).await?
            };
            let text = response.text();
            let ok = turn.expect.is_none_or(|e| text.contains(e)) && turn.forbid.is_none_or(|f| !text.contains(f));
            failures += usize::from(!ok);
            println!("  turn {}: {} {:?} -> {:?}", i + 1, if ok { "ok  " } else { "FAIL" }, turn.user, text);
        }
        println!("{}: {failures} failing turn(s)", if with_session { "with session" } else { "without session" });
    }
    Ok(())
}
```

```text
  turn 1: ok   "My name is Ada." -> "Noted."
  turn 2: ok   "I'm planning a trip to Lisbon." -> "Noted."
  turn 3: ok   "What is my name?" -> "Your name is Ada."
with session: 0 failing turn(s)
  turn 1: ok   "My name is Ada." -> "Noted."
  turn 2: ok   "I'm planning a trip to Lisbon." -> "Noted."
  turn 3: FAIL "What is my name?" -> "I don't know your name."
without session: 1 failing turn(s)
```

With a real model, run each script several times: later turns depend on earlier replies, so variation compounds. For a long conversation, check the turns that matter (a fact recalled, a constraint kept) rather than every turn, and remember that compaction, if configured, may drop what a late turn needs, which is worth its own case.

## Evaluating workflows

A workflow run records every event, so a workflow can be evaluated on more than its final output: which executors ran (and so which branch a router took), what each agent said, how many tokens each used, and whether anything failed. The program below evaluates a routing workflow, with a classifier agent, a function node that reads its label and conditional edges to two team agents, against cases with the expected team.

```rust
//! Evaluating a workflow: routing accuracy, the path through the graph,
//! each agent's output and the final output, from the run's events.
use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::workflow::{AgentExecutor, FunctionExecutor};
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};

/// A stand-in model: the classifier labels by keyword, the teams echo.
/// Swap in a real client to evaluate the real prompts.
#[derive(Clone)]
struct StandIn;

#[async_trait]
impl ChatClient for StandIn {
    async fn get_response(&self, messages: Vec<Message>, _o: ChatOptions) -> Result<ChatResponse> {
        let system = messages.first().map(Message::text).unwrap_or_default();
        let user = messages.iter().rev().find(|m| m.role == Role::user()).map(Message::text).unwrap_or_default();
        let reply = if system.starts_with("Classify") {
            let team = if user.contains("crash") || user.contains("error") { "tech" } else { "billing" };
            json!({ "team": team }).to_string()
        } else {
            format!("{} team: we are looking into \"{user}\"", system.split_whitespace().next().unwrap_or("?"))
        };
        let mut response = ChatResponse::from_text(reply);
        response.usage_details = Some(UsageDetails { input_token_count: Some(30), output_token_count: Some(10), ..Default::default() });
        Ok(response)
    }

    async fn get_streaming_response(&self, messages: Vec<Message>, options: ChatOptions) -> Result<ChatStream> {
        let response = self.get_response(messages, options).await?;
        let mut update = ChatResponseUpdate::text(response.text());
        if let Some(usage) = response.usage_details {
            update.contents.push(Content::Usage(agent_framework::types::UsageContent { details: usage }));
        }
        Ok(futures::stream::iter(vec![Ok(update)]).boxed())
    }
}

fn build_workflow(client: impl ChatClient + Clone + 'static) -> Result<Workflow> {
    let agent = |instructions: &str| -> Arc<dyn SupportsAgentRun> {
        Arc::new(Agent::builder(client.clone()).instructions(instructions).build())
    };
    let classify = agent("Classify the ticket. Reply with JSON {\"team\": \"billing\" | \"tech\"}.");
    // Reads the classifier's label and forwards the ticket with it.
    let route = FunctionExecutor::new("route", |message, ctx| async move {
        let conversation: Vec<Message> = serde_json::from_value(message)?;
        let ticket = conversation.first().map(Message::text).unwrap_or_default();
        let label: Value = serde_json::from_str(&conversation.last().map(Message::text).unwrap_or_default())?;
        ctx.send_message(json!({ "team": label["team"], "ticket": ticket })).await?;
        Ok(())
    });
    let to_text = |prefix: &'static str| {
        FunctionExecutor::new(prefix, |message, ctx| async move {
            ctx.send_message(message["ticket"].clone()).await?;
            Ok(())
        })
    };
    WorkflowBuilder::new()
        .add_executor(Arc::new(AgentExecutor::new("classifier", classify)))
        .add_executor(Arc::new(route))
        .add_executor(Arc::new(to_text("to_billing")))
        .add_executor(Arc::new(to_text("to_tech")))
        .add_executor(Arc::new(AgentExecutor::new("billing", agent("Billing: answer billing tickets.")).with_output(true)))
        .add_executor(Arc::new(AgentExecutor::new("tech", agent("Tech: answer technical tickets.")).with_output(true)))
        .set_start("classifier")
        .add_edge("classifier", "route")
        .add_conditional_edge("route", "to_billing", |m: &Value| m["team"] == "billing")
        .add_conditional_edge("route", "to_tech", |m: &Value| m["team"] == "tech")
        .add_edge("to_billing", "billing")
        .add_edge("to_tech", "tech")
        .build()
}

#[tokio::main]
async fn main() -> Result<()> {
    let workflow = build_workflow(StandIn)?;
    let cases = [
        ("I was charged twice this month", "billing"),
        ("The app shows error 500 on login", "tech"),
        ("My card was charged after the app crashed", "billing"),
    ];

    let mut correct = 0;
    for (ticket, expected) in cases {
        let run = workflow.run(ticket).await?;
        let mut path = Vec::new();
        let mut tokens = 0;
        let mut failed = None;
        for event in run.events() {
            match event {
                WorkflowEvent::ExecutorInvoked { executor_id } => path.push(executor_id.clone()),
                WorkflowEvent::AgentRun { response, .. } => {
                    let response: AgentResponse = serde_json::from_value(response.clone())?;
                    tokens += response.usage_details.and_then(|u| u.input_token_count.zip(u.output_token_count)).map_or(0, |(i, o)| i + o);
                }
                WorkflowEvent::ExecutorFailed { executor_id, error } => failed = Some(format!("{executor_id}: {error}")),
                _ => {}
            }
        }
        let routed_to = ["billing", "tech"].into_iter().find(|team| path.iter().any(|p| p == team));
        let output: Vec<Message> = match run.last_output() {
            Some(v) => serde_json::from_value(v)?,
            None => Vec::new(),
        };
        let answer = output.last().map(Message::text).unwrap_or_default();

        let ok = failed.is_none() && routed_to == Some(expected) && answer.contains(ticket);
        correct += usize::from(routed_to == Some(expected));
        println!(
            "{} {ticket:?}\n    routed to {routed_to:?} (expected {expected}), path {path:?}, {tokens} tokens{}",
            if ok { "PASS" } else { "FAIL" },
            failed.map(|f| format!(", failed at {f}")).unwrap_or_default()
        );
    }
    println!("routing accuracy: {correct}/{}", cases.len());
    Ok(())
}
```

```text
PASS "I was charged twice this month"
    routed to Some("billing") (expected billing), path ["classifier", "route", "to_billing", "billing"], 80 tokens
PASS "The app shows error 500 on login"
    routed to Some("tech") (expected tech), path ["classifier", "route", "to_tech", "tech"], 80 tokens
FAIL "My card was charged after the app crashed"
    routed to Some("tech") (expected billing), path ["classifier", "route", "to_tech", "tech"], 80 tokens
routing accuracy: 2/3
```

The useful events are `ExecutorInvoked` (the path, in superstep order and, within a superstep, in executor-id order), `AgentRun` (each agent executor's finished `AgentResponse` as JSON, including its `usage_details`), `ExecutorFailed` and `Failed`, and `Output` for what the workflow yielded. Routing accuracy is the headline number for a router; a confusion table (expected against actual team) is the next step once there are more than two branches. The third case shows why a keyword classifier needs a golden set: "charged after the app crashed" is a billing complaint that mentions a crash.

Orchestrations evaluate the same way, with one difference: group chat, handoff and Magentic run their whole loop inside one executor, so their `AgentRun` and `AgentRunUpdate` events all carry the orchestrator's id (`group_chat_orchestrator`, `handoff_coordinator`, `magentic_orchestrator`), and the participant is the `author_name` on the response's messages. Check the final conversation for who spoke and in what order, and for handoff whether the right specialist answered. A workflow that pauses for input ends with `IdleWithPendingRequests`; an evaluation drives it with scripted answers through `send_response`, as the [workflows page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_workflows/#human-in-the-loop) shows.

## Cost and latency

An evaluation should report what a change costs as well as whether it is better. Each `AgentResponse` carries `usage_details`, summed over every model request in the run, including those that carried tool results; a provider that reports no usage leaves the fields `None` rather than zero. Wall-clock time per run is an `Instant`. For where the time goes, the spans the crate already emits (`invoke_agent` per run, `chat` per model request with `ObservableChatClient`, `execute_tool` per tool call) can be timed by a small `tracing_subscriber` layer, with no OpenTelemetry setup.

```rust
//! Cost and latency: token usage from each response, wall-clock time per
//! run, and per-span timings collected by a small tracing layer.
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use agent_framework::observability::ObservableChatClient;
use agent_framework::prelude::*;
use serde_json::json;
use tracing::span::{Attributes, Id};
use tracing::Subscriber;
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
use tracing_subscriber::registry::LookupSpan;

/// Records how long every span named in `names` was open.
#[derive(Clone, Default)]
struct SpanTimings(Arc<Mutex<BTreeMap<&'static str, Vec<Duration>>>>);

struct Opened(Instant);

impl<S: Subscriber + for<'a> LookupSpan<'a>> Layer<S> for SpanTimings {
    fn on_new_span(&self, _attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            span.extensions_mut().insert(Opened(Instant::now()));
        }
    }

    fn on_close(&self, id: Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(&id) else { return };
        let name = span.metadata().name();
        if !["invoke_agent", "chat", "execute_tool"].contains(&name) {
            return;
        }
        let elapsed = span.extensions().get::<Opened>().map(|Opened(started)| started.elapsed());
        if let Some(elapsed) = elapsed {
            self.0.lock().unwrap().entry(name).or_default().push(elapsed);
        }
    }
}

/// The value below which `p` of the samples fall (nearest rank).
fn percentile(samples: &mut [Duration], p: f64) -> Duration {
    samples.sort();
    let rank = ((p * samples.len() as f64).ceil() as usize).clamp(1, samples.len());
    samples[rank - 1]
}

/// Prices per million tokens, from your provider's price list.
struct Price {
    input_per_million: f64,
    output_per_million: f64,
}

fn cost(usage: &UsageDetails, price: &Price) -> f64 {
    let input = usage.input_token_count.unwrap_or(0) as f64;
    let output = usage.output_token_count.unwrap_or(0) as f64;
    (input * price.input_per_million + output * price.output_per_million) / 1_000_000.0
}

#[tokio::main]
async fn main() -> Result<()> {
    let timings = SpanTimings::default();
    tracing::subscriber::set_global_default(tracing_subscriber::registry().with(timings.clone()))
        .expect("no other subscriber is installed");

    // ObservableChatClient adds a `chat` span per model call; the agent adds
    // `invoke_agent` and `execute_tool` spans itself.
    let client = ObservableChatClient::new(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?, "openai");
    let stock = FunctionTool::new(
        "stock_level",
        "Units in stock for a product code.",
        json!({ "type": "object", "properties": { "sku": { "type": "string" } }, "required": ["sku"] }),
        |args| async move { Ok(json!({ "sku": args["sku"], "units": 14 })) },
    );
    let agent = Agent::builder(client).instructions("Answer stock questions with the tool.").tool(stock.into_definition()).build();
    let price = Price { input_per_million: 0.40, output_per_million: 1.60 };

    let queries = ["How many A-100 are left?", "Is B-7 in stock?", "Stock for C-3?", "Do we have D-9?"];
    let mut run_times = Vec::new();
    let mut total = UsageDetails::default();
    for query in queries {
        let started = Instant::now();
        let response = agent.run_once(query).await?;
        run_times.push(started.elapsed());
        // usage_details is summed over every model call in the run,
        // including the ones that carried tool results.
        let usage = response.usage_details.clone().unwrap_or_default();
        println!(
            "{query:<28} {:>4} in {:>4} out  ${:.6}",
            usage.input_token_count.unwrap_or(0),
            usage.output_token_count.unwrap_or(0),
            cost(&usage, &price)
        );
        total.add_assign(&usage);
    }

    println!(
        "total: {} tokens, ${:.6}; run p50 {:?}, p95 {:?}",
        total.total_token_count.unwrap_or(0),
        cost(&total, &price),
        percentile(&mut run_times.clone(), 0.5),
        percentile(&mut run_times, 0.95)
    );
    for (name, samples) in timings.0.lock().unwrap().iter_mut() {
        println!("{name:<13} n={} p50 {:?} p95 {:?}", samples.len(), percentile(samples, 0.5), percentile(samples, 0.95));
    }
    Ok(())
}
```

```text
How many A-100 are left?       24 in   16 out  $0.000035
Is B-7 in stock?               24 in   16 out  $0.000035
Stock for C-3?                 24 in   16 out  $0.000035
Do we have D-9?                24 in   16 out  $0.000035
total: 160 tokens, $0.000141; run p50 3.377049ms, p95 4.772611ms
chat          n=8 p50 1.807461ms p95 2.252337ms
execute_tool  n=4 p50 14.155µs p95 20.994µs
invoke_agent  n=4 p50 3.276177ms p95 4.584827ms
```

Each run made two model requests, one that asked for the tool and one that answered from its result, and the usage shows both (the stand-in reports 12 input and 8 output tokens per request, and answers locally, hence the tiny times). Prices change and differ by provider and region, so keep them in configuration rather than code, and count cached input tokens (`cache_read_input_token_count`) and reasoning tokens (`reasoning_output_token_count`) separately where your provider bills them differently. Report percentiles rather than means: a few slow runs, usually retries or long tool calls, are what users notice. In production the same numbers come from the `otel-metrics` histograms (`gen_ai.client.token.usage`, `gen_ai.client.operation.duration`, `agent_framework.function.invocation.duration`); see [metrics](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_observability/#metrics).

## An evaluation gate for CI

The last piece turns an evaluation into a check a pull request must pass. A model's answers vary, so a single run is not a measurement: run each case several times and compare pass rates, against an absolute floor and against a stored baseline with some tolerance. The program below does both, writes a JSON report that CI can keep as an artefact, and exits with code 1 on a failure.

```rust
//! An evaluation gate for CI: repeat each case, write a JSON report, and
//! fail on a drop against the stored baseline or below a floor.
use std::collections::BTreeMap;
use std::path::PathBuf;

use agent_framework::prelude::*;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::json;

struct Case {
    id: &'static str,
    input: &'static str,
    must_call: Option<&'static str>,
    must_contain: &'static [&'static str],
}

#[derive(Serialize, Deserialize, Default)]
struct Report {
    model: String,
    /// Case id -> fraction of repeats that passed.
    pass_rates: BTreeMap<String, f64>,
    overall: f64,
    total_tokens: u64,
}

fn agent(client: impl ChatClient + 'static) -> Agent {
    let lookup = FunctionTool::new(
        "lookup_order",
        "Find an order by number.",
        json!({ "type": "object", "properties": { "order": { "type": "string" } }, "required": ["order"] }),
        |args| async move { Ok(json!({ "order": args["order"], "status": "dispatched" })) },
    );
    Agent::builder(client).instructions("Use lookup_order for order questions.").tool(lookup.into_definition()).build()
}

/// One attempt at one case: pass or fail, and the tokens it used.
async fn attempt(agent: &Agent, case: &Case) -> (bool, u64) {
    let Ok(response) = agent.run_once(case.input).await else { return (false, 0) };
    let tokens = response.usage_details.as_ref().and_then(|u| u.total_token_count).unwrap_or(0);
    let text = response.text().to_lowercase();
    let called = |name: &str| response.messages.iter().flat_map(|m| m.function_calls()).any(|c| c.name == name);
    let passed = case.must_call.is_none_or(called) && case.must_contain.iter().all(|w| text.contains(&w.to_lowercase()));
    (passed, tokens)
}

#[tokio::main]
async fn main() -> Result<()> {
    let model = std::env::var("EVAL_MODEL").unwrap_or_else(|_| "%%model=openai-small%%".to_string());
    let repeats: usize = std::env::var("EVAL_REPEATS").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    let floor: f64 = std::env::var("EVAL_FLOOR").ok().and_then(|v| v.parse().ok()).unwrap_or(0.8);
    let tolerance = 0.05; // allowed drop against the baseline
    let baseline_path = PathBuf::from(std::env::var("EVAL_BASELINE").unwrap_or_else(|_| "eval-baseline.json".into()));
    let report_path = PathBuf::from(std::env::var("EVAL_REPORT").unwrap_or_else(|_| "eval-report.json".into()));

    let agent = agent(RetryingChatClient::new(OpenAIChatCompletionClient::from_env(model.as_str())?));
    let cases = [
        Case { id: "order-status", input: "Where is order 1001?", must_call: Some("lookup_order"), must_contain: &["dispatched"] },
        Case { id: "greeting", input: "Hello!", must_call: None, must_contain: &[] },
    ];

    // Every (case, repeat) pair, four at a time.
    let attempts: Vec<(&str, (bool, u64))> = futures::stream::iter(cases.iter().flat_map(|c| std::iter::repeat_n(c, repeats)))
        .map(|case| {
            let agent = &agent;
            async move { (case.id, attempt(agent, case).await) }
        })
        .buffer_unordered(4)
        .collect()
        .await;

    let mut report = Report { model, ..Default::default() };
    for case in &cases {
        let runs: Vec<_> = attempts.iter().filter(|(id, _)| *id == case.id).collect();
        let passed = runs.iter().filter(|(_, (ok, _))| *ok).count();
        report.pass_rates.insert(case.id.to_string(), passed as f64 / runs.len() as f64);
    }
    report.total_tokens = attempts.iter().map(|(_, (_, t))| t).sum();
    report.overall = report.pass_rates.values().sum::<f64>() / report.pass_rates.len() as f64;
    std::fs::write(&report_path, serde_json::to_string_pretty(&report)?).map_err(Error::other)?;

    let mut problems = Vec::new();
    if report.overall < floor {
        problems.push(format!("overall pass rate {:.2} is below the floor {floor:.2}", report.overall));
    }
    match std::fs::read_to_string(&baseline_path) {
        Ok(text) => {
            let baseline: Report = serde_json::from_str(&text)?;
            for (id, rate) in &report.pass_rates {
                if let Some(before) = baseline.pass_rates.get(id) {
                    if rate + tolerance < *before {
                        problems.push(format!("{id}: pass rate fell from {before:.2} to {rate:.2}"));
                    }
                }
            }
        }
        Err(_) => println!("no baseline at {}; this run's report can become one", baseline_path.display()),
    }

    for (id, rate) in &report.pass_rates {
        println!("{id:<14} {:.0}% of {repeats}", rate * 100.0);
    }
    println!("overall {:.0}%, {} tokens, report {}", report.overall * 100.0, report.total_tokens, report_path.display());
    if !problems.is_empty() {
        problems.iter().for_each(|p| eprintln!("FAIL {p}"));
        std::process::exit(1);
    }
    Ok(())
}
```

```text
no baseline at eval-baseline.json; this run's report can become one
greeting       100% of 3
order-status   100% of 3
overall 100%, 240 tokens, report eval-report.json
```

With `EVAL_FLOOR=1.1` (an impossible floor), or with a baseline that claims a higher pass rate, the same program prints the reason on standard error and exits with 1:

```text
FAIL overall pass rate 1.00 is below the floor 1.10
```

Settings come from the environment so that one binary serves several jobs: `EVAL_MODEL` to compare models, `EVAL_REPEATS` to trade cost for confidence, `EVAL_BASELINE` and `EVAL_REPORT` for the files. Commit the baseline, and update it deliberately (copy a good report over it) in the same pull request as the change that improved it. `RetryingChatClient` keeps a rate-limited request from counting as a failed case, and `buffer_unordered(4)` bounds concurrency so a suite does not trip the provider's limits.

In CI, keep the scripted regression suite in `cargo test` on every push, and run the live evaluation where a key is available and spending is acceptable: on a schedule, on demand, or on pull requests that touch prompts and tools.

```yaml
# .github/workflows/evals.yml
name: evals
on:
  schedule:
    - cron: "41 4 * * *"
  workflow_dispatch:
  pull_request:
    paths: ["src/agents/**", "prompts/**", "evals/**"]

jobs:
  evals:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - run: cargo run --locked --release --bin evals
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          EVAL_REPEATS: "3"
          EVAL_BASELINE: evals/baseline.json
          EVAL_REPORT: eval-report.json
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: eval-report
          path: eval-report.json
```

This workflow was not run as part of checking this page. The [testing page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_testing/#running-the-suite-in-ci) covers the scripted and live test jobs, caching and `--locked`.

## Practical notes

- **Build the agent under test with the production constructor.** Pass the client in, so the evaluation runs the real instructions, tools and middleware; a copy drifts.
- **Rules before judges.** Run the deterministic checks first and call the judge only for what they cannot decide; a judge call costs as much as the run it grades.
- **Small and curated beats large and noisy.** Twenty cases that each catch a real failure are worth more than a thousand generated ones. Add a case for every bug found in production.
- **Version everything that changes a score:** the golden file, the judge's instructions and model, the agent's prompts and model, and the crate version.
- **Watch for drift in the provider.** A model alias can change behind the same name; pin a dated model version in evaluation runs where the provider offers one.
- **Evaluation data is data.** Cases, saved runs and judge prompts can hold customer text; keep them under the same controls as the logs they came from.
