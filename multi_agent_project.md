## Project Architecture 
Project Name:
**Agentic Investment Researcher**

Built with
**LangGraph**

### High-Level Architecture
Think of the system as 4 subgraphs coordinated by a central orchestrator.

```ini
User
 ↓
Query Validator
 ↓
Orchestrator
 ├── Planner Subgraph
 ├── Researcher Subgraph
 ├── Quant Analyst Subgraph
 ├── Analyst Subgraph
 │   ├── Evaluation Subgraph
 │   └── Report/Analyst Subgraph
```

Cross-cutting nodes:
- Token Budget Monitor
- Semantic Guardrails
- HITL Interrupt
- Checkpointer

###### Concepts Implemented in this Project
- Multi-agent communication
- Tool calling
- Planning + task decomposition
- Memory (short-term / long-term)
- Retrieval or search
- Evaluation / self-reflection loop (critic agents or feedback loops)
- Graph-based workflow orchestration
- MCP multi-agent control patterns (planner/executor/evaluator architecture)

<!-- These are exactly the things people mean when they talk about agentic AI and make this project comprehensive. -->

#### Core Graph Design
##### Goal Interpreter
Clarifies user intention
Responsibilities:

- User query validator:
    - Validates user query for relevance, safety and clarity
    - If not met, activated HITL to ask user to revise their query

- Asks user to rethink their query if not clear or valid
    Example:
    - Topic: Investment Report/Recommendation
    - User query: Analyze NVIDIA and produce an investment research summary

##### Orchestrator
Implements Orchestrator-Worker pattern. The "Planning" Phase Separates Planning from Execution which is a hallmark of "Reasoning" agents. In other words, this pattern separates what to do (planning) from how to do it (execution), making agents more modular, testable, and easier to extend.

Responsibilities:

###### Planner Agent
Creates plan by breaking down the goal into smaller subtasks will be implemented by worker agents.  Task decomposition is the core of the planning design pattern. Instead of asking a single agent to handle a complex request end-to-end, we break the problem into smaller, well-defined subtasks. Each subtask is assigned to a specialist agent (e.g., flights, hotels, activities) with clear priorities and dependency ordering.

This approach provides several benefits:
- Clarity: each subtask has a single responsibility.
- Parallelism: independent subtasks can run concurrently.
- Reliability: failures are isolated to individual subtasks.
- Budget tracking: costs are estimated per subtask and rolled up.

- Planner subgraph
    - Breaks a user query into tasks
    - Decides which agents/tools are needed
    - `plan = List[Task]`, Task is a Pydantic model
    - Nodes:
        - *Planner Architect*: generates a DAG plan of required tasks
        - *Plan Validator*:
        Checks plan quality. If poor → reflection loop.
            - Plan is an acyclic graph
            - Dependencies are not missing or ambiguous
    
Planner decomposes goal. 

Example task plan:
-  *Research event*
    - Collect financial indicators: web search or vector database
- *Quantitative analysis*
    - performs  simulations or generates charts or plots
- *Auditor*
    - Evaluate sources and results from quant agent, finds discrepancies between research data and quantitative results
- *Analyst*
    - Generate report. aggregate all the result into a summary report that might make a financial advices


###### Scheduler Node
Marks tasks `pending → ready → running` before dispatching them to agents. But before this, it checks if the dependent tasks are marked "completed" by agents.

Scheduler:
- Selects ready tasks (task dependencies are complete)
- Assigns tasks to agents
- Manages task status


Graph Flow
```ini
Planner
↓
Task Scheduler
↓
Worker Agent
↓
Task Completion
↓
Task Scheduler
```
Loop continues until all tasks are completed.

###### Router Edge
A conditional edge to route tasks to Agents and back to the Scheduler. It
- Fans out tasks to the corresponding agents in parallel
- When agents return their updates, graph flows back to the scheduler to send the next batch of tasks in parallel



##### The Core Architecture of Worker Agents
Workers receive tasks from the scheduler.

In a standard RAG (Retrieval-Augmented Generation) setup, the flow is linear. In the investment graph, the agents can "think", fetch data, realize it needs more context, and go back to fetch more.

- *State*: This is the "shared memory". It could include the ticker symbol, a collection of financial news, raw SEC filings, and the current draft of the investment thesis. These are Artifacts.

- *Nodes*: These are the workers (Python functions or LLM calls).

- *Edges*: These define the logic of how agents interact (e.g., "If the data is insufficient, go back to the Search node; if it's ready, go to the Analyst node").

######  Possible Subgraph Nodes

To make this high-quality, you might want to break your graph into these specialized nodes:

Node Name |	Responsibility
----- | ------
Market Data Fetcher | 	Pulls real-time prices and historical trends (using APIs like Yahoo Finance or Alpha Vantage).
Sentiment Analyzer	| Scrapes recent headlines and social signals to gauge market mood.
Fundamental Scraper |	Parses 10-K and 10-Q filings for revenue, debt-to-equity, and cash flow.
Risk Auditor	| Specifically looks for "red flags" or bearish indicators to avoid bias.
Investment Reporter	| Synthesizes all the above into a final recommendation (Buy/Hold/Sell).

But I kept it simpler: a research agent which is a separate agent from this subgraph collect all the live data the quant agent needs (The "Eyes").

###### Implementing Agents
Keep the following in mind when implementing agents:

1. Clear separation of responsibilities

    Make sure each component does one thing well:
    - Planner – only generates DAG/tasks from the user query
    - Scheduler – routes tasks to the right agents dynamically
    - Agents – execute tasks (research, math, summarization, etc.)
    - Quant sandbox – handles all numeric/financial computation

    This ensures each piece can be swapped or improved independently.

2. Explicit input/output schemas

    Define exactly what each agent receives and returns
    For the quant sandbox, define a schema like:

    ```json
    {
    "task_id": "calculate_var",
    "input": {"ticker": "AAPL", "data": "..."},
    "output": {"value_at_risk": 12345.67, "confidence": 0.95}
    }
    ```
    This makes it easier for the scheduler to orchestrate and for debugging later.

3. Error handling / fallback

    Agents sometimes fail (network issues, data unavailable, etc.):
    - Implement retry mechanisms in the scheduler
    - Allow agents to return error codes + messages rather than crashing the whole workflow
    - For the quant sandbox, provide a default behavior or placeholder if data is missing

    This makes the system more robust for a demo and for future scaling.

4. Logging and observability

    Since you’re building an agentic system:
    - Track task execution times, dependencies resolved, and outputs
    - You could even visualize the DAG dynamically to see which tasks are completed or pending
    This will also make your demo visually impressive

5. Consider multi-agent communication patterns

    Right now you mentioned using global state reducers. That works, but think about:
    - Message passing: agents send messages or updates to the scheduler or to other agents
    - Events or pub/sub: agents subscribe to events they need to act on
    - Hybrid: some info in global state, some via messages for efficiency

    This can scale better as you add more agents.

6. Extensibility

    Make it easy to add a new tool or agent later without touching the whole system
    - Keep interfaces abstract: e.g., the scheduler just calls agent.run(task) without caring what the agent does internally

7. Demo focus

    Remember your primary goal is a convincing demo:
    - Full automation of research summary
    - Quant sandbox produces meaningful numbers
    - Planner and scheduler orchestrate correctly
    - Output looks like a real research report


All other improvements can go on a “future improvements” list — no need to perfect everything before you demo.



##### The Researcher (Search Specialist) 
In an investment research context, you don't want "generalist" agents. You want Functional Specialists that interact with your services (MCP and Sandbox). We will start with the Research Agent (The "Eyes") and the Quantitative Agent (The "Brain").

This agent's job is to gather fresh data from the web or retreive documents for database. In this application, it only performs web search using MACP tools. It summarizes information into the researcher artifacts. 
- Tools: Tavily Search or DuckDuckGo are great for structured research.
- Strategy: Instruct the LLM to perform multiple searches from different angles (e.g., competitors, financials, recent news) before concluding.
- Prompt Tip: "You are a research specialist. Prioritize authoritative sources and provide inline citations (e.g., [1]) for every claim." 

##### Analyst Agent (Reasoning Specialist)
The analyst doesn't search; it "thinks" over the gathered data. 
- Tools: Primarily uses internal reasoning but can be given a Summarizer tool to condense large research reports.
- Strategy: Use a "Chain of Thought" prompt to ensure it doesn't skip steps when transforming raw data into investment insights.
- Role: Acts as the data scientist's assistant, looking for underlying trends and risks. 

#####  The Vector DB Agent (Knowledge Retrieval)
This agent handles "Semantic Search" over your private documents (PDFs, filings, internal reports). 

- Tools: A retriever tool connected to Pinecone, Weaviate, or FAISS.
- Strategy: Implement "Agentic RAG"—let the LLM decide whether it needs to query the vector store or if it has enough info in its current context. Not implemented in this project.

##### The Quant Agent (Sandbox/Python Specialist)
This is your "Calculation" agent. It should never "hallucinate" math; it should write and run code instead. 

- Tools: A Python REPL or Sandbox environment.
- Environment: Use Modal Sandboxes or Runloop to execute untrusted code safely without crashing your main graph.
- Role: Transforms raw financial statements into visualizations, charts, or complex ratio analyses

###### Auditor/Critic/Evaluator Agent
This agent is auditing the quant analyst work. If not passed, it generates a feedback and returns the flow to the quant agent to correct its output. It retires up to a MAX_ITERATION. 


#### Research Subgraph
Runs tasks in parallel.

My design include single node to perform a web search using a MCP server. Our search tool that uses Tavily Search will perform the search for the agent once the agent calls the MCP server.  However this could be extended to more specialized nodes such as:

```ini
Research Coordinator
 ├── News Research Agent
 ├── Financial Data Agent
 ├── Document Retrieval Agent
 ```

These agents may use their own MCP tools from MCP servers:
```ini
web_search_server
finance_data_server
document_server
```
The agents discover tools dynamically via MCP.

Outputs are formatted into **Artifacts Schema** and saved in the global graph state. In addition the results can be aggregated via map-reduce node.

MCP Tools:
- web search
- document retrieval

Outputs:
- articles
- facts
- sources

#### Analyst Subgraph
- Transforms research into structured insights:
    - Event_summary
    - Financial_signals
    - Sentiment_analysis
    - Risk_flags

- Nodes:
    - The **Analyst** produces a "Candidate Report."
    - The **Evaluator** (a separate node/LLM call) generates a "Scorecard."
    - The **Optimizer** (the original Analyst) receives the scorecard and "Self-Corrects" based on that external feedback.

It could have more nodes such as:

```ini
Trend Analyzer
Risk Detector
```
Analyst subgraph transforms research data into insights.
Outputs:
```ini
insights
risk_signals
market_trends
```

###### Evaluator/Optimizer Nodes
Implements Evaluator-Optimizer pattern.
- Nodes:
    ```ini
    Evaluator Agent
    Optimizer Agent
    ```
- Evaluator uses rubric:
    ```ini
    source reliability
    analysis coherence
    data consistency
    coverage
    ```
If score low → send to optimizer.
- Optimizer refines results.

Could be a subgraph on its own:
##### Evaluator-Optimizer (Subgraph-Level)

Placement: A standalone evaluation/ subgraph.
What it is: A conversation between two different "personas."
- The Evaluator is the "Senior Partner."
- The Optimizer is the "Associate."

Why: This creates **Adversarial Collaboration**. The Evaluator is incentivized to find flaws that the Optimizer missed.

Example: 
- Evaluator grades the analysis.
- If Grade < 8, Optimizer fixes it. (Repeat up to 3x).

#### Quantitative Subgraph (PAL)
- Implements Program-Aided Language Models.
    Flow:
    ```ini
    Quant Agent
    ↓
    Code Generator
    ↓
    Python Sandbox (MCP Server)
    ↓
    Result Parser
    ```
- PAL Integration: The prompt explicitly forces a quant_sandbox task. This ensures LLM "vibes" with actual code execution.
- Tasks:  
    - Trend projection
    - Volatility estimate
    - Growth modeling
    - Simple forecasting
    - Monte Carlo Simulations
    
- Avoids hallucinated math. Instead of the LLM invents math:
    - LLM writes Python code
    - Code executes in isolated and safe sandbox
    - Result returned to agent

- Nodes:
    - **Quant Agent**: User data sources from Research Agents to generate safe Python Code to perform Data Analytics and run simulations
    - **Auditor Agent**: Perform self-reflection task. Checks for hallucinations, logical or runtime errors, discrepancy, and provides feedback. Judges the results from Quant node for required outputs, validity of code and data used in the code, discrepancies between the results from Quant agent and research data. PASS or FAIL if requirements not met. 
        - If failed, the task will be returned to the Quant Agent with the feedback from Auditor 
        - Cycle repeats for a MAX_ITERATION allowed
        - After max retires, task marked failed and return to the scheduler
    - **Router Quant Edge**
    Conditional edge which routes the task from Quant node back to the Quant node before it reaches the Auditor if deterministic requirements are not met. For example, code output is empty or agent didn't call the `execute_python_code` tool
    - **Router Auditor Edge**
    Conditional edge which checks if max retires exhausted, routes the task to the scheduler otherwise returns thanks to the Quant node with Auditor feedback 

#### Toolset for Agnets

MCP allows you to offload the "heavy lifting" of tool integration (like connecting to real-time financial APIs, local databases, or secure sandboxes) to a standardized layer. Agents access external capabilities through MCP servers, which decouple tool execution from reasoning. This allows tools to be added or scaled independently.

###### Task-scoped tool availability
Instead of exposing all tools to the agent all the time, we make tools available based on the tasks.  Agents have access only to tools related to the task they do. 

Benefits:
- Reduces hallucinated tool calls
- Forces clearer task boundaries


###### Why MCP ?
- **Tool Decoupling**: Your agents stay "thin." Instead of importing heavy libraries like yfinance or pandas into every agent file, the agent simply calls a tool hosted on the MCP server.
- **Standardization**: You can use the same MCP server for your Researcher (Search/News), Quant (Python/Analysis), and Vector DB (RAG) agents. 
- **Local vs. Remote**: You can run an MCP server locally to handle sensitive financial data (like private portfolios) while the LLM runs in the cloud. 

###### Possible Map MCP to 4 Agents:
Agent	| Suggested MCP Server/Tool |
-------- | -----------
Investment Researcher	| Brave Search or Tavily MCP for real-time web data and news as external knowledge. Market Data MCP Tools:<ul><li>`get_stock_price(ticker)`<li>`get_company_profile(ticker)`<li>`get_financials(ticker)`<ul>
Quant	| Sequential Thinking or Python REPL MCP to execute calculations without hallucinations. 
Vector DB	| RAG integration via MCP: FAISS or Chroma or Postgres or Pinecone MCP to query your internal knowledge base via standardized SQL/Vector calls.<ul><li>`search_documents(query)`<li>`get_document(id)`<ul> 
Analyst	| Memory MCP to maintain "long-term" insights about specific companies across different research threads.

> Note: Add *structured tool responses*.
Instead of returning raw text: 
>```json
>{
>  "ticker": "NVDA",
>  "price": 912.3,
>  "currency": "USD",
>  "timestamp": "..."
>}
>```
>That shows schema discipline in agent systems. When the MCP server returns a response, validate it against a schema before feeding it back to the LLM.





### Best Patterns for Multi Agent Architectures
This project implemented best Agentic AI design patterns such as
##### Reflection Pattern
###### Critic Agent
Auditor/Critic agent inside quant_analyst subgraph implements a self-correction loop. It checks:
- **Hallucinations**: values are from research data
- **Logical errors**: summary is print to stdout, artifacts wre saved to share volumes
- **Missing data or discrepancies**: if quant computation is sound based and matches the live data provided by the research agent or historical data from vector_db

Flow:
```ini
generate → critique → revise
```

- Reflection is a major agentic pattern.
- Limit iterations (important).
- Placement: Inside the research/ or analysis/ subgraphs.

What it is: A single node that follows a generation node.

```ini
Analyst Node → Reflection Node (Analyst checks its own syntax) → Output.
```

It’s cheap and fast. You don’t need a whole subgraph for an agent to check if it forgot a comma.

##### Evaluator-Optimizer Pattern 
Separate from reflection.
- Evaluator checks output against rubric:
    - source_quality
    - analysis_depth
    - data_consistency
    - clarity

- Optimizer agent refines result.

This demonstrates structured evaluation pipelines.
- Semantic Guardrails

##### Semantic Guardrail Node
Validate outputs with **Pydantic**.
Example schema:
```ini
ResearchOutput
{
 source
 claim
 confidence
 url
}
```
If invalid JSON:
```ini
Retry Node → regenerate
```
This is very important in production systems. All agent outputs must match specific schema.

Example:
- ResearchOutput
- AnalysisOutput
- QuantitativeOutput
- ReportOutput

If invalid:
- Trigger Retry node

##### FinOps
This node is about token budget protection. It runs before every major step. Add a node that tracks:
- Total_tokens
- Total_cost
- Iteration_count

If exceeded → terminate workflow. This prevents agent loops, a real production issue. Each task execution updates:
```ini
state["token_usage"]
```
Your *Token Monitor node* checks before scheduling new tasks. Token Monitoring as a First-Class Citizen: This addresses FinOps. It shows you care about the company’s cloud bill, not just the "coolness" of the AI. This shows production engineering awareness. 


##### Human-in-the-Loop (HITL)

Investment decisions are high-stakes. LangGraph allows you to add a breakpoint. The graph can do all the research, pause, and wait for you to "approve" the data before it spends tokens on the final synthesis. This is the *Human-in-the-loop* break.

Example:
When system detects high uncertainty or high-impact decision:
- Graph pauses
- User approves:
    - Continue
    - Revise
    - Terminate

I implemented this at user query validator node. If the query is not relevant to the topic, not safe, not clear, the graph interrupts the flow and asks user to revise the query. LangGraph supports this with *interrupts*.

This shows safe AI design.

- LangGraph supports interrupts.
    Example:
    ```ini
    if confidence < threshold
    or
    high-impact action
    ```
    Pause graph.
- User options:
    ```ini
    approve
    revise
    cancel
    ```
- Then continue workflow.


Define a *breakpoint*. In your `graph_builder.py`, compile the graph with an `interrupt_before=["report_node"]`. This tells the system: "Save everything to the DB and stop right here until a human sends a 'Resume' command".


###### A Logic Flow with HITL & Evaluation

Here is a sequence for your `evaluation_graph.py` where human interfere when agent is stuck after retrying multiple times:

```ini
[Analysis Subgraph Finish] 
          ↓
[Evaluator Node] (Grades 1-10 based on rubric)
          ↓
    < Is Score >= 8? > ────── Yes ──────▶ [Final Report Subgraph]
          │                                         ▲
          No                                        │
          │                                         │
    < Attempt < 3? > ── Yes ──▶ [Optimizer Node] ───┘ (Loops back to Evaluator)
          │
          No (The Agent is stuck)
          │
          ▼
[HITL Interrupt Node] ◀── (Human provides "Correction Hint")
          │
          └───────────────────▶ [Optimizer Node] (Restarts with Human help)
```





###### Where do we put HITL (Human-in-the-Loop)?

It should be a State Gate placed at high-risk junctions. The three best places for HITL in your architecture:

- *Post-Planning*: After the planning finishes, the agent says: "I plan to search these 5 sources and run a Monte Carlo simulation with these parameters. Proceed?" This prevents wasted tokens on a bad plan.

-  *Escalation*: stuck loop. If the evaluator rejects the Optimizer's work 3 times in a row (an "Infinite Loop"), the graph should interrupt and ask a human for guidance.

- *Final Sign-off*: Before the final report is generated or after the final report is generated, but before it is published or emailed. Once the human clicks "Approve" the graph resumes and formats the final PDF.


#####  MCP Servers

<!-- ###### The MCP Server (FastMCP) -->
To set up a modern, performant MCP system for your investment research graph, we will use `FastMCP` for the server and the `LangChain MCP Adapter` for your agents.

FastMCP handles the protocol complexity for you and supports the latest Streamable HTTP transport by default. For your use case, the best path is to use FastMCP Python SDK. It handles the `JSON-RPC` complexity for you and lets you focus on the logic. Instead of hard-coding tool logic into your agent, you are building a standardized, decoupled interface.

Update your research agent to use these dynamic tools: call the server to find the tools, bind them to models so they use them. You can add this as a `ToolNode` to ensure the model uses it (optional). 


Advantages of SSE/HTTP:

- **Scalability**: One Research MCP server can serve multiple different LangGraph instances simultaneously.

- **Isolation**: Your Research server can have its own heavy dependencies (like Playwright or Chromium) without bloating your main Agent container.

- **Observability**: You can put a Load Balancer or an API Gateway in front of it.

To stay on the cutting edge, use **Streamable HTTP**, which was introduced in the March 2025 MCP specification update. It is the modern, performant standard that officially deprecates the older HTTP+SSE transport. 

###### Why Streamable HTTP?
- Single-Endpoint Efficiency: Unlike SSE, which required two separate endpoints (one for streaming and one for posting), Streamable HTTP uses a single URL for all bidirectional messaging.
- Performance: Benchmarks show it can handle ~300 requests per second, nearly 10x faster than the previous SSE implementation.
- Production Readiness: It is far more firewall and load-balancer friendly because it uses standard HTTP patterns rather than fragile, long-lived streaming connections.
- Stateless Operations: It allows servers to scale horizontally more easily and supports optional session resumption via a Mcp-Session-Id header. 


<!-- ##### How to use it in your Researcher Node
To implement this in your project using the `LangChain MCP Adapter`, you will configure the connection with the `streamable_http` transport. 

 -->

###### The "Dual-Mode" Server

You don't have to choose! You can write your server to support both. FastMCP makes this relatively easy, though the http transport requires a bit more boilerplate using starlette or FastAPI.
```python
# mcp_servers/research_server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("InvestmentResearch")

@mcp.tool()
async def get_market_sentiment(ticker: str):
    # Your logic here
    return f"Sentiment for {ticker} is Bullish"

if __name__ == "__main__":
    # You can toggle this via an ENV var or CLI flag
    import sys
    if "sse" in sys.argv:
        # Runs as a networked server on port 8000
        mcp.run(transport="sse") 
    else:
        # Runs as a local pipe
        mcp.run(transport="stdio")
```

Adding MCP on top of everything else turns this project from a normal agent system into a modern and forward-looking system. This is exactly the kind of architecture that makes interviewers lean forward. Let's design the actual LangGraph node structure so you can implement it cleanly. We'll include MCP using Model Context Protocol and build the orchestration with LangGraph application equipped with  MCP clients.

- Instead of hardcoding tools, agents discover them through MCP servers. This demonstrates **dynamic tool discovery**. 

- Architecture:
    ```ini
    Agent
    ↓
    MCP Client
    ↓
    Available Tools
    ├ web_search
    ├ financial_data
    ├ document_lookup
    └ python_execution
    ```
    

<!-- Instead of importing your search functions directly into your agents, you run them as a separate process. 
This shows you understand Micro-service architecture. Your agents don't care how the web search works; they just know the MCP protocol. -->


#####  Retry Node (Failure Handling)
This is a more deterministic feature rather than agentic. Used when:
- Schema fails
- Tool fails
- Evaluation fails

If a task fails:
```ini
Retry node
↓
Reflection agent
↓
Retry task
```
Limit retries. Otherwise escalate to HITL node.


##### Persistence
I used **LangGraph checkpointer**.
This gives:
- resumable workflows
- fault tolerance

If the system stops:
- resume from last node

Very good interview talking point.



##### Traceability
Send all task execution to LangSmith.
You can show:
- DAG execution
- agent calls
- token usage
- tool calls
- execution graph
- reasoning chains

Great for debugging and demos. Companies care about observability.


##### Memory
Two types:
- Short-term: Graph state.
- Long-term: Vector database storing past investigations.

Example use:
- retrieve previous company analysis


#####  Parallelization
Used LangGraph's parallel nodes. 

Example: 
(Research Subgraph): Real-world agents shouldn't be sequential. A Senior Engineer knows that searching a Vector DB and a Web API simultaneously saves 50% in latency. Or tasks:
- News search
- Financial search
- Regulatory search

are independent and should be run simultaneously. Later nodes wait:
```
Analyst or Quant Analyst
```
This demonstrates parallel reasoning workflows, which is very impressive in interviews.

#####  Map-Reduce: 
Multiple research outputs aggregated.
- Final Output
    - Structured intelligence report:
    - Executive Summary
    - Event Overview
    - Market Analysis
    - Quantitative Trends
    - Risks
    - Sources


#####  Subgraphs
I organized system like this:
- Planning Subgraph
- Research Subgraph
- Quant Analysis Subgraph
- Analysis Subgraph
    - Evaluation Subgraph
    - Report Subgraph

This makes architecture clean and scalable. To decide whether something should be a Node or a Subgraph, use this Senior Architect's rule of thumb:

- Use a Node for a single, atomic LLM call or a specific tool execution (e.g., "Web Search").

- Use a Subgraph for a multi-step logic loop that has its own internal state and success criteria (e.g., "Try to research until the data is sufficient").



#### Why this architecture is excellent
This project demonstrates almost every modern agentic design pattern:
- Multi-agent orchestration
- DAG-based orchestration
- Parallel agent execution
- Semantic guardrails
- MCP tool ecosystems
- PAL quantitative reasoning
- Reflection loops
- HITL safety
- Evaluator-optimizer pipelines
- Token budgeting
- Persistent workflows
- Traceability

<!-- 
That is cutting-edge agentic architecture. This is very close to production AI architecture.

Goal:
- Clear architecture
- Working system
- Easy explanation
- Not maximum complexity.

My honest opinion
If you build 70–80% of what you described, this will be an outstanding portfolio project. It will clearly show that you understand agentic AI system design, not just how to call an LLM.
 -->







##### Final Graph Overview

![](./assets/agentic_ai//mygraph.png)

### Step by Step Progress

1. The Pre-Flight Check (Goal & Token Monitor)
    - Node: Query Validator.
    - Tech: Uses a small model (e.g., GPT-4o-mini) to turn a vague user prompt into a structured JSON schema.
    - Logic: If the requested task is "Too Broad," the node returns an error to the user before a single expensive token is spent on the Research Subgraph.

2. The Research Subgraph (Parallel & Hybrid)
    - Pattern: Fan-out/Fan-in.
    - Node A (Vector Store): Queries internal proprietary data (Historical Analysis).
    - Node B (Web/MCP): Queries real-time news via an MCP-connected search tool.
    - Convergence: A Summarizer node merges these two streams into a "Current Context" object.

3. The Quantitative Subgraph (PAL Pattern)
    - Logic: The agent generates Python code → MCP Server executes code in a Dockerized Sandbox → Returns JSON results.
    - Senior Defense: "We use a Program-Aided Language (PAL) approach to ensure 100% mathematical accuracy, treating the LLM as a logic-generator rather than a calculator."

4. The Reflection Loop & HITL
    - Checkpoint: Before the "Report Subgraph", the graph enters a "Pending Approval" state.
    - Action: The state is saved to a database. You (the Human) review the findings in a UI. You can "Approve" or "Add Comment."
    - Resumption: The graph "wakes up" and incorporates your feedback into the final report.

### Planner 
The Planning Subgraph is the most critical part of a Senior-level agent. If the plan is flawed, the Research and Quantitative agents will simply "hallucinate with confidence" for 10 minutes and waste $5.00 in tokens.

To make this production-ready, we don't just "ask the LLM to plan." We implement Query Synthesis and Structural Guardrails.

###### The Planning Logic Flow

We will break the planning/ subgraph into three distinct steps (nodes) to ensure the agent doesn't over-reach.

- Guardrail Node (Query Validator): Is the user asking for financial advice (legal risk)? Is the ticker valid? Is the request too vague?

- Decomposition Node (The Architect): Breaks the high-level goal into a `List[Task]` (e.g., "Fetch 10-K," "Run DCF," "Search News").

- Resource Estimator: Checks if we have the tools/budget to fulfill the plan.


##### Implementation
We'll use Pydantic Tool Choice to force the LLM to output a structured plan that fits our MasterState.

**The Guardrails**

To make this interview-defensible, you should mention these three guardrail patterns:

- *Syntax Guardrails*: In planning_architect, we use `with_structured_output(StructuredPlan)`. If the LLM returns bad JSON, LangGraph's built-in retry mechanism will catch it.

- *Semantic Guardrails*: Before the plan is accepted, a "Validator" node checks if the plan includes at least one "Quantitative" step. If it’s just "Web Search," the validator rejects the plan and asks for a more technical approach.

- *Token Budgeting*: The StructuredPlan includes an estimated_tokens field. We check: if state.total_cost + estimated_cost > MAX_BUDGET: STOP.

This  multi-node subgraph validates the ticker (could be via an MCP tool - checking if AAPL exists), decomposes the request into typed Task objects, and checks if the plan is economically viable or passes a sanity before starting. The plan has

- *Dependency Awareness*: By defining `depends_on`, we've moved past simple sequential loops. It uses topological sorting to identify which research nodes can run in parallel, optimizing for latency.

- *Error Resilience*: If the LLM generates a circular dependency (Task A depends on B, B depends on A), Validator Node can catch this before the graph execution starts.

#### DAG Planning Architecture
My Planning Architect produces a **task DAG** instead of a linear list. DAG is stored in Graph State in `plan` key.

Example planner output:
```JSON
{
  "tasks": [
    {
      "id": "research_news",
      "agent": "research",
      "task": "Collect recent news articles",
      "depends_on": []
    },
    {
      "id": "research_filings",
      "agent": "research",
      "task": "Retrieve regulatory filings",
      "depends_on": []
    },
    {
      "id": "analysis_trends",
      "agent": "analysis",
      "task": "Extract trends from research",
      "depends_on": ["research_news", "research_filings"]
    },
    {
      "id": "quant_forecast",
      "agent": "quant",
      "task": "Run growth forecasting",
      "depends_on": ["analysis_trends"]
    },
    {
      "id": "report_generation",
      "agent": "report",
      "task": "Generate final report",
      "depends_on": ["analysis_trends", "quant_forecast"]
    }
  ]
}
```
This is exactly how workflow orchestration systems work (Airflow, Prefect, etc.).

###### Planner Output Schema 
Uses strict validation with Pydantic schema.

Example:
```python
from pydantic import BaseModel
from typing import List

class Task(BaseModel):
    """A granular step in the investment research process."""
    id: int
    agent: Literal["research", "analyst", "quant_analyst"]
    description: str
    depends_on: List[int] = Field(default_factory=list, description="IDs of tasks that must complete before this task starts")
    status: Literal["pending", "ready", "running", "completed", "failed"] = "pending"
    error_message: str | None
    retries: int = 0 

class Plan(BaseModel):
    tasks: List[Task]
```

This enforces *semantic guardrails*. If the LLM produces invalid structure → retry. This also constrains allowed agent types.

```python
VALID_AGENTS = [
  "research",
  "analysis",
  "quant analyst",
]
```
Reject unknown agents. Otherwise the planner might invent `sentiment_super_agent` which breaks execution.








##### How to execute DAG Orchestrator-Worker

Use LangGraph **dynamic routing** to execute the DAG. This aligns naturally with how LangGraph is designed and avoids writing your own scheduler. LangGraph already supports:
- Conditional edges
- Parallel branches
- State-driven routing

In this pattern, a central Orchestrator node (your task scheduler) analyzes the state, identifies which tasks are ready, and then "fans out" those tasks to the corresponding agents. In LangGraph, the most modern and efficient approach for this "Orchestrator-Worker" pattern is to use the **Send** API or the **Command** object. 

- Node (The Scheduler)
Instead of a simple "if/else" conditional edge, your scheduler node can dynamically trigger multiple agent nodes in parallel using the **Send** command. 

- Agent Nodes (The Workers)
Each agent is a standard node that processes its assigned task and returns its result to the state.


Why use **Send** over standard Conditional Edges?
- **Parallelism**: Standard conditional edges typically go to one next node. Send allows you to trigger multiple agents at the same time (fan-out).
- **Dynamic Mapping**: You don't have to hard-code every possible path. If your scheduler decides Task A goes to Agent X, it just happens at runtime.
- **Scalability**: This pattern handles any number of tasks and agents without making the graph visual "spaghetti".


The scheduler node identifies "ready" tasks and immediately mark them as running, but it doesn't actually trigger them yet. To trigger them, we use a **Conditional Edge** right after the scheduler that uses `Send()` to dispatch all running taks to the agents. Send command takes the whole graph state and some extra keys (optional) as argments to forward to agents. Agents must expect the extra keys in their states if different from the global graph state. 

###### Why this works:
- Parallelism: If 3 tasks are `ready` scheduler returns 3 Send objects, and LangGraph runs them at the same time.
- State Management: Because of the Annotated reducer, when an agent returns `{"task_status": {"task_1": "complete"}}`, it doesn't delete "task_2"; it just updates the specific key.
- The Loop: Every time an agent finishes, the graph hits the scheduler again. The scheduler sees a new "complete" status, marks the next dependent tasks as "ready," and the cycle repeats until dispatch tasks are completed.


#### Failing Agents

This is an important reliability layer in DAG agent systems. The key idea is: a task fails when the agent cannot produce a valid artifact or encounters an execution error.

Let’s keep it simple and structured.

1. What “task failure” actually means?
 
    A task is failed if any of these happen:

    ###### Tool Failure
    Example:
    - web search API fails
    - MCP tool unavailable
    - Python sandbox throws exception

    ###### LLM Failure
    Examples:
    - invalid JSON output
    - schema validation fails
    - hallucinated tool name

    ###### Logic Failure
    Examples:
    - required artifact missing
    - planner created invalid dependency

2. Detecting Failure in an Agent

    Wrap agent execution in `try/except`.

    Example:
    ```python
    id="agentfail"
    def research_agent(state):

        task = state["current_task"]

        try:

            result = run_research_tool(task)

            artifact = Artifact(
                task_id=task.task_id,
                artifact_type="research",
                source="research_agent",
                content=result,
                timestamp=now()
            )

            return {
                "artifacts": [artifact],
                "completed_tasks": [task.task_id]
            }

        except Exception as e:

            return {
                "failed_tasks": [task.task_id],
                "error_messages": [str(e)]
            }
    ```

3. Scheduler Handling Failure

    Scheduler logic becomes:
    ```python
    if task.retries < MAX_RETRIES:
        retry_task(task)
    else:
        mark_task_failed(task)
    ```

4. Retry Flow

    Retry is simple. Scheduler just requeues the task.
    ```ini
    id="retrylogic"
    task.retries += 1
    task.status = PENDING
    ```
    Then it will be picked up again.
    
5. Reflection Agent (recommended)

    If retry fails once, send task to reflection agent.

    Flow:

    ```ini
    Task fails
    ↓
    Reflection agent
    ↓
    Improved task instruction
    ↓
    Retry task
    ```
    Example reflection output:
    ```ini
    {
    "problem": "Web search query too vague",
    "fix": "Add date filter and company ticker"
    }
    ```
    Then scheduler retries with improved instruction.


6. Final Failure

    If retries exhausted:
    ```ini
    id="finalfail"
    task.status = FAILED
    ```
    Scheduler decision:
    - *Strict mode*: Stop the whole workflow.
    or
    - *Tolerant mode* (better): Continue DAG if downstream tasks don't depend on it.
    - *HITL*: ask a human to intervene


##### Task Status Updates
One option is to keep task status in a separate dictionary to actually have the cleaner and more scalable design. For example, here is the pattern that most workflow engines use.

1️⃣ Treat the Plan as Immutable
Your plan should never change. Think of it as a static DAG specification:  `plan: List[Task]`. The plan only describes:
```ini
task_id
agent
description
dependencies
```
It is not execution state. This is exactly how systems like Airflow work.

2️⃣ Execution State Lives Separately
Execution state should live in dictionaries like this:
```ini
task_status: Dict[int, TaskStatus]
task_retries: Dict[int, int]
Example:
task_status = {
  1: "completed",
  2: "running",
  3: "failed"
}
```
Your scheduler only reads this structure.
3️⃣ Graph State Structure (Clean Version)
Example:
```ini
state = {

  "plan": List[Task],

  "task_status": Dict[int, TaskStatus],

  "task_retries": Dict[int, int],

  "artifacts": List[Artifact]
}
```
This separation is very important.

If you stored status inside Task (as I did), you would need to mutate the plan: `task.status = RUNNING`

To do this you would need to use **reducers** to tell LangGraph how to update them so that 
- state merging 
- parallel updates 
- debugging 

go smooth!  To make this thread-safe for parallel tasks, you must use a **Reducer** in your MasterState. This prevents one agent's update from "clobbering" another's.

```python
def update_plan_status(current_tasks: List[Task], updates: List[TaskUpdate]) -> List[Task]:
    """
    This is the 'Senior' logic. It maps the updates back to the plan 
    without overwriting the whole list.
    """
    task_map = {t.id: t for t in current_tasks}
    for up in updates:
        if up.task_id in task_map:
            task_map[up.task_id].status = up.status
            if up.error_message:
                task_map[up.task_id].error = up.error_message
    return list(task_map.values())

class MasterState(BaseModel):
    # LangGraph will now use 'update_plan_status' to merge agent returns
    plan: Annotated[List[Task], update_plan_status] 
```

This is the approach i used for this project. However keeping status separate makes execution pure and deterministic.

5️⃣ Scheduler Logic (Cleaner)
Scheduler only reads plan + task_status.
Example:
```python
ready_tasks = []
for task in state["plan"]:

    if state["task_status"].get(task.task_id) in ["running", "completed"]:
        continue

    if all(
        state["task_status"].get(dep) == "completed"
        for dep in task.depends_on
    ):
        ready_tasks.append(task)
```
Very clean.

6️⃣ When Scheduling a Task
Scheduler sets:
`task_status[task_id] = "running"`
Then routes it to the agent.
7️⃣ This Pattern Is Called
Immutable Plan + Execution State

Used in:
- Airflow
- Prefect
- Ray DAGs
- Temporal workflows

##### Check if the planner output is a real DAG?

You must validate that the plan is truly a DAG before execution. It’s a small step but extremely important because an LLM planner can accidentally generate cycles. Then, your scheduler would loop forever. So validating acyclicity once, right after planning, is the correct design.

DAG validator should be right after the Planning Architect produces the tasks:
```ini
User Query
   ↓
Planning Architect
   ↓
DAG Validator   ← add here
   ↓
Scheduler
```
If validation fails, you can:
- Ask the planner to regenerate
- Or run a reflection step to fix dependencies.

The standard approach is Kahn’s algorithm (Topological Sort). The exactly the same solution as we had for Lettcode quetion `Schedule Course II`.

Another common planner error: Task 5 depends on Task 99
but Task 99 doesn't exist.

###### Validation Pipeline
Your validator should check:
- Task IDs unique
- Dependencies exist
- Graph is acyclic

This makes the plan safe to execute. If the validator finds an invalid plan, it should clear the state.plan list before sending the agent back to the architect. If you don't, the next time the architect runs, it might see the old "bad" plan in the state and get confused.

###### In Interviews
If asked about reliability, you can say:
“Since the planner is an LLM, I added a structural validator that ensures the generated plan is a valid DAG by checking dependency existence and performing a topological sort.” 

That signals excellent system design awareness.

Next, the biggest reliability boost you can add is something many agent systems miss:
- plan sanity checks (detecting when the planner generates nonsensical tasks even if the graph is valid).



###### How it actually flows

- Step A: The Planner starts everything as pending

    ```Python
    # In planning_architect node:
    tasks = [
        Task(id=1, status="pending", depends_on=[]),
        Task(id=2, status="pending", depends_on=[1])
    ]
    return {"plan": tasks}
    ```
- Step B: The Scheduler "Promotes" tasks

    The Scheduler runs. It sees Task 1 has no dependencies. It changes it to ready and then immediately to running as it sends it to the agent.

    ```Python
    # In scheduler node:
    for task in state.plan:
        if task.status == "pending" and all_dependencies_met(task, state.plan):
            task.status = "ready"
        
        if task.status == "ready":
            task.status = "running"
            # Trigger the agent
            return Send("research_agent", {"task": task})
    ```
- Step C: The Agent "Reports" completion

    The Agent doesn't "reach into the database." It just returns a Result object. This is where the Reducer does the magic.

    ```Python
    # In research_agent node:
    return {
        "artifacts": [data],
        "plan": [TaskUpdate(task_id=1, status="completed")] # The Reducer merges this
    }
    ```

###### Why this is Safe (The Reducer is the Judge)

Agents report their status; they don't manage it. When an agent returns status="completed", it’s like a worker dropping a "Job Done" card on the boss's desk. The Reducer (the Boss) is the only one who actually picks up the pen and updates the Master plan list. This is safe because:

- LangGraph pauses execution while the Reducer runs.

Even if 5 agents finish at the same time, the Reducer processes their "Job Done" cards one-by-one in a single transaction. The Scheduler then wakes up, sees the new completed status, and moves the next pending task to ready.

Since your Scheduler is essentially a loop, make sure you have a "Terminal State" check. One subtle bug to watch for (very common): If an agent fails silently and returns nothing, the scheduler might never update status.

Good safeguard:
```python
if task_status[task_id] == RUNNING for too long:
    mark FAILED
```
This prevents stuck workflows.



#### User query validation
In production, an agent shouldn't just "fail" or "give up"; it should engage in Conversational Repair. If the `query_validator` finds a problem, you have two high-level choices:

- Autonomous Rejection: The system tells the user why it can't proceed and ends.

- Human-in-the-Loop (HITL) Clarification: The system asks the user to refine the query and waits for a response (Recommended)

You don't need a complex new node. You simply need a Terminal State Gate.

`query_validator`: Checks for safety/clarity.

is_valid check:

- True → Proceed to planning_architect.
- False → Update messages with the reason for failure and move to a waiting_for_input node.

In LangGraph, the most "production-ready" way to do this is to use a Breakpoint. You don't write a node that "waits"; you tell the graph to Interrupt itself. 

```Python
# We tell the graph: "If the query is invalid, STOP and wait for the human."
app = workflow.compile(
    checkpointer=memory,
    interrupt_after=["query_validator"] # The graph freezes here if it needs a fix
)
```

In an interview, you can say:

"I implemented Stateful Interruption Points for query validation. Instead of hard-failing on vague prompts, the graph utilizes a persistent checkpointer to pause execution. This allows for a multi-turn 'Conversational Repair' loop where the human provides clarification, and the agent resumes with the updated context, saving the compute cost of restarting the entire pipeline."



### Quant Analyst Subgraph

As we need a self reflection loop using an auditor, it becomes cleaner to have the quant analyst in a its own graph as a subgraph of the main graph. In LangGraph, we call this the "Isolate and Encapsulate" pattern.

By compiling the Quant-Auditor-Retry loop as its own graph, you treat the entire "Quantitative Analysis" as a single black box to the Master Planner. The Planner says "Do a DCF," and the subgraph doesn't return until it has a verified result or has exhausted its retries.

Think of the subgraph as a "Unit Test" for the agent's work.

- *Cleaner State*: The `QuantInput` schema stays local to the subgraph. The Master State doesn't need to be cluttered with `retry_count `or `audit_feedback`.

- *Parallelism*: If you had two quant tasks, you could spin up two instances of this subgraph simultaneously without state collisions.

- *Testing*: You can run the Quant Subgraph in a notebook with a mock task to see if it can self-correct without running the whole system.

######  Status Logic for Quant Tasks


If the Quant Agent sets the status to completed but the Auditor then sets it to failed, you create a "flapping" status that can confuse the Planner or the UI.

- Senior Recommendation: The "Pending Verification" State
- Don't mark the task completed until the Auditor says PASS.
```ini
Quant Agent: Finishes code → Sets status = "pending_verification".
```
- Auditor Node:

    - If FAIL: Sets status = "in_progress" (with feedback for the retry).

    - If PASS: Sets status = "completed".

- Route: If retry_count > 3 and still failing, sets status = "failed" and exits.



#### Quant Node

###### The Quantitative Agent System Prompt

- Role: Senior Quantitative Investment Analyst (Autonomous Agent)

- Objective: Translate complex financial valuation tasks into deterministic, executable Python code. You operate within a Dockerized Sandbox via MCP.

- Operational Constraints:

    - *No Manual Math*: Never perform calculations (DCF, CAGR, simulations) in your text response. Always write and execute Python code.

    - *Data Lineage*: Use the specific financial variables retrieved by the Researcher. If a variable (e.g., WACC or terminal growth rate) is missing, explicitly flag it as a MISSING_DATA error for the Auditor.

    - *Library Standard*: Prefer `numpy` for simulations, `pandas` for data manipulation, and `matplotlib` for generating visual artifacts.

    - *Deterministic Outputs*: Ensure your code prints a JSON-formatted string to stdout so the system can parse your results into the Artifact Schema.

###### Workflow:

- Plan: State the mathematical approach (e.g., "Implementing a 10,000-iteration Monte Carlo simulation for revenue sensitivity").

- Code: Call the `execute_python_code` tool.

- Verify: If the tool returns an error, analyze the traceback, fix the logic, and re-run (done by Auditor).

- Finalize: Summarize the results based only on the script's output.





##### Hallucination Analysis

While the Planner assigns the task (e.g., "Perform a DCF"), the Quant Agent is a "Logic Generator," not a "Data Finder." It needs the specific numbers (Revenue, WACC, Growth rates) that the Researcher just found to populate the variables in its Python script.

Before the LLM writes the code, you need to ensure it has the numbers. Instead of letting the LLM "guess" variable names, your node can inject a standard header into the prompt:

```o
The following variables are available from the Research Phase: revenue_2025=120.5B, wacc=0.085. Use these exact values in your script.
```

If you don't pass the research findings, the Quant Agent will be forced to hallucinate the numbers or fail because of NameError: revenue is not defined. For example, LLM might make up values in the generated code:

`data_center_revenue = 2_500 # Example value`

NVIDIA's Data Center revenue is actually north of \$26 Billion per quarter in 2024. The agent used a "mock" value of $2.5 Billion because it wasn't forced to use the Research Artifacts. This is a classic failure in agentic workflows called "Knowledge Neglect." That's why ou should implement a **Context Injection** step before calling the LLM.

###### Handling the Data Gap:  "Research-to-Quant" Bridge

The most common point of failure in these graphs is the Quant agent not having the data it needs to run the math.

The Solution: Implement a Pre-Flight Check.
- If the Quant agent realizes it's missing the "Discount Rate" or "Historical Volatility," it should have an edge in the graph that points back to the Researcher with a specific request: "Need 5-year average WACC for NVDA to complete DCF."

<!-- 
When the MCP returns the JSON, you will map it to your Artifact class. Since you are using a shared volume, the content field should store the numeric results, while the source or a metadata field stores the file path. -->

<!-- ### The "Self-Correction" Loop (Redo up to 3 times)

To implement the "redo" logic within the node, you can use a simple for loop. This is a very high-ROI feature because it demonstrates Agentic Robustness.

 -->







#### Why Quant Sandbox?

In investment research, "Alpha" usually lives in the gap between what people say (Researcher) and what the numbers actually do (Quant). The Researcher is mostly I/O bound. The Quant agent introduces logic-bound tasks. It forces you to handle data passing between agents (e.g., the Researcher finds a ticker, the Quant needs that ticker to run a script).

Your sandbox environment having things like:
- numpy
- pandas
- matplotlib
- yfinance

means the quant agent can realistically generate things like:
- Monte Carlo simulations
- expected return distributions
- volatility estimates
- simple portfolio risk metrics
- charts / plots


The Quant Agent shouldn't just write a script; it should follow a Plan-Execute-Verify cycle.


- The PAL (Program-Aided Language) Workflow

    When the Quant Agent receives the prompt, it follows these steps:

    - State Synthesis: It gathers the necessary variables from the Global State (e.g., NVIDIA's current revenue, growth rates, and WACC found by the Researcher).

    - Code Generation: It writes a self-contained Python script.

    - Pro Tip: Ensure the agent uses `numpy` or `pandas` for the simulations and `matplotlib` for the visual artifact.

- MCP Execution: The agent sends the code to the MCP server. The server spins up the Docker container, runs the code, and captures stdout (text/JSON) and any generated files (images/CSV).

- Validation: The agent checks if the code errored out. If it did, it uses the Traceback to fix the code and retries (Self-Correction loop).

###### Why this is Senior Defense

If an interviewer asks, "How do you handle LLM math hallucinations?" your answer is:

"We don't let the LLM do math. We use the PAL pattern. The LLM acts as a translator that turns an investment hypothesis into a deterministic Python script. We then execute that script in an isolated MCP-managed Docker sandbox, ensuring 100% computational accuracy and environment security."

The Sandbox saves the file to a shared disk and returns a simple string: `"path": "/app/artifacts/sim_123.png"`. The LangGraph API (and eventually your Frontend) simply reads that file from the disk. It’s faster and more stable.

If your mcp-quant container restarts or crashes, any files inside it are lost forever. By using a Shared Volume, the Artifacts (the proof of your analysis) are persisted on the host machine.

Data Type | Transmission Method | Reason
-------- | --------- | ------------
Numeric Results | MCP stdout (JSON) | "Small, fast, and easy for the LLM to read."
Python Errors | MCP stderr | "Critical for the ""Self-Correction"" loop."
Visual Charts | Shared Volume | Prevents memory bloat; allows the Frontend to serve the file via a URL.

We implemented a sidecar volume strategy to decouple metadata from heavy binary artifacts. This keeps our LangGraph state lightweight and ensures that high-resolution financial visualizations don't cause latency or memory issues during agent orchestration.



Let’s implement the Senior Pattern for Artifact Management in your mcp-quant server.

###### Artifacts Directory  -- **Fix this!2**

- We will assume your docker-compose mounts a volume to `/app/artifacts` inside the MCP quant container.

- The MCP server will create a unique folder for each execution (e.g., `/app/artifacts/sim_<task_id>/`).

- The LLM?? will be instructed (via System Prompt) to save any plots to that folder.

- The MCP server?? will look for new files in that folder and return their relative paths.

This is the most critical part of your "UI-less" defense. In your Jupyter Notebook (which has the same `/app/artifacts` volume mounted), you display the result.

Our architecture provides full data lineage. We persist every simulation’s inputs (context.json), the generated logic (`script.py`), the computational traces (`stdout/stderr`), and the visual proof (`result.png`) in a structured, timestamped audit trail within our shared volume. This allows for a deterministic replay of any investment recommendation.

Quant MCP Server returns the relative path of charts/assets (e.g., `sim_nvda_123/plot.png`). Your Notebook then joins that relative path with your local ARTIFACTS_ROOT.

###### Some small notes for Quant Sandbox 

1.  Add execution limits
    
    Generated code should have limits.
    Inside the quant container enforce:
    - timeout (e.g. 5–10 seconds)
    - memory limits
    - no external network calls

    Example idea:
    - execution_timeout = 10s
    - max_memory = 512MB
    
    This prevents runaway code.

2. Artifact management

    If the sandbox generates charts or files:
    Return something like:
    ```json
    {
    "result": numeric_result,
    "artifacts": [
        "/tmp/chart1.png",
        "/tmp/table.csv"
    ]
    }
    ```
    Your agent can later attach those to the final report.

3. Deterministic validation before LLM auditor

    Before calling the auditor, check:
    - did the code execute?
    - did it produce output?

    Simple rule:
    - if execution_error -> retry by Quant agent

    This avoids wasting LLM tokens by Auditor.

4. Structured tool schema

    Your MCP tool should enforce something like:
    ```python
    def execute_quant_code(code: str) -> str
    ```

    This helps guide the LLM and reduces hallucinations.



### Auditor Node

That auditor layer is extremely valuable because it addresses one of the biggest real problems with LLM systems: *hallucinated or inconsistent outputs*. The Auditor Agent is a piece of the "Reliability Triad":

- Researcher: Finds the raw data.
- Quant: Writes the math logic.
- Auditor: Validates that the logic matches the data.

##### What the Auditor Checks
Use deterministic validation combined with LLM judge:
- rule-based checks
- LLM reasoning only if needed
    
Example:
- if execution_error -> regenerate 
elif result_invalid -> regenerate
else -> accept

    This makes your system more reliable.

Auditor can check:

1. Code sanity

    Check for obvious hallucinations:
    - undefined variables
    - invalid libraries
    - incorrect syntax

    If errors exist → send feedback to Quant to fix it

2. Result plausibility

    Example checks:
    - numeric output exists
    - result not None
    - result within reasonable range

    Example: `return_on_equity` must be between -1 and 1

3. Determinist validation

    If the quant sandbox returns artifacts (charts, files):
    Check:
    - file exists
    - not empty

###### Feedback Loop
If auditor fails:
- Auditor → feedback → update messages
- Agent → regenerate code
- Quant MCP → run again

 **Limit retries**; Otherwise agents can loop forever.

<!-- Example: `max_retries = 2` -->



#### Auditor Persona
The Auditor should be a "Skeptic." It shouldn't just look at the result; it should look at the generated code `script.py` (from the artifacts folder) and check for:

- *Hardcoded Hallucinations*: Did the agent ignore the research and just "guess" a growth rate?
- *Unit Errors*: Is it mixing Millions and Billions?
- *Formula Integrity*: Is the WACC or DCF formula mathematically sound?

Without auditor,  you're trusting the LLM's math blindly. The Auditor acts as a "Senior Peer Reviewer" who looks at the generated code to ensure it isn't hallucinating formulas.

This node takes the Artifact produced by the Quant Agent and compares it against the Research findings.

To make this actually work, you need a Conditional Edge in your graph definition that sends the flow back to the quant_agent_node if the Auditor fails it.

To make the Auditor effective, your Quant Agent Node must save the code it generated into the Artifact. 

```python
# Inside quant_agent_node
return {
    "artifacts": [
        Artifact(
            # ... other fields ...
            content={
                "code": tool_call["args"]["code"], # <--- SAVE THE CODE HERE
                "stdout": mcp_data.get("stdout"),
                "plots": mcp_data.get("artifacts")
            }
        )
    ]
}
```


Even if the code runs (Status: Success), the math might be wrong. The Auditor is your final check. For example, the last line of the generated script might be:
`mean_value, median_value, value_5th_percentile, value_95th_percentile`

In a standard Python script (unlike a Jupyter cell), just typing the variable names at the end does not print them. You must explicitly use `print()`. This is the perfect "Case Study" for your Auditor Agent. The Auditor should catch two things here:

- Data Mismatch: The revenue used (\$2.5B) doesn't match the Research Artifacts (~$26B+).

- Silent Output: The script ran but didn't actually "report" its findings to stdout.

Or you might need to check for discrepancies between the quant analysis and research data. Are values used in code match the research data? if not, the whole analysis is not reliable because value are probably hallucinated or misunderstood.

- Input to Auditor: The Python code written by the Quant Agent + The Research Data.

- Check: "Does the WACC formula used in `script.py` match standard financial practice?"

- Output: True (Pass to Analyst) or False (Send back to Quant with instructions).

Senior Defense: "We implemented a tiered validation strategy. Syntax errors are handled internally by the Quant Agent's Self-Correction loop, while logical/financial inaccuracies are caught by a specialized Auditor Agent, ensuring that the final report is both technically sound and financially accurate."


To prevent the "Silent Output" and "Mock Data" issues, we need to tighten the Quant Agent's instructions (system prompt):
```ini
NO MOCK DATA: You MUST use the values provided in the "Research Context" section.

STRICT OUTPUT: You must end your script with print(json.dumps(results_dict)).

VISUALS: If the task asks for a simulation, you MUST save a plot to ARTIFACT_DIR + '/plot.png'.
```

when the Auditor fails, send the feedback back to the Quant Agent. The Quant Agent will see:
```ini
"FAIL: You used mock data of 2,500 but the research shows 26,000. Also, you didn't print the results."
```

The LLM is much better at fixing code than writing it perfectly on the first try.

Your Auditor is currently the only thing standing between a "Pro" financial report and an embarrassing hallucination. Since you're using MasterState for your `QuantInput,` make sure the Auditor specifically has access to the Research Artifacts from that state. If the Auditor doesn't "see" the \$26B figure found by the researcher, it might think $2.5B "looks" reasonable and pass it!


#### Auditor Prompt

Since the infrastructure is now perfect, the only thing left is the Auditor's Prompt. Make sure it's "Mean."

"You are a cynical Hedge Fund Auditor. Your job is to find reasons to REJECT this code. If the numbers don't match the Research exactly, FAIL it. If it doesn't use `print(json.dumps())`, FAIL it."


To ensure your Auditor is a "hard-to-please" senior reviewer, we need to design its prompt to be adversarial. It shouldn't just look at the code; it should act as a cross-referencing engine between the Research Artifacts and the Quant Code.

1. The Auditor's "Discrepancy" Logic

    The Auditor needs to perform a "Data-to-Code" mapping. If the researcher found "NVIDIA Q4 Revenue: $22.1B", the Auditor must look for 22_100 or 22100000000 in the code. If it sees 2_500, it triggers a FAIL.

2. The Senior Auditor System Prompt

    Here is an example of high-precision prompt designed to kill hallucinations:

    ```ini
    AUDITOR_SYSTEM_PROMPT = """
    You are a Lead Financial Auditor. Your sole mission is to ensure the Quantitative Agent's simulation is FACTUALLY ACCURATE and LOGICALLY SOUND.

    INPUT DATA FOR VERIFICATION:
    1. RESEARCH DATA: The ground truth extracted from filings/news.
    2. QUANT CODE: The Python script the agent wrote to process this data.
    3. STDOUT: The actual output of that script.

    YOUR CRITICAL ERROR LIST:
    - DATA HALLUCINATION: Did the agent use 'Mock' or 'Example' numbers instead of the specific figures in the Research Data? (Check billions vs millions!)
    - SILENT FAILURE: Did the script run but fail to print the results in a structured format (JSON)?
    - LOGIC ERROR: Is the formula for DCF, WACC, or CAGR mathematically incorrect?
    - UNIT MISMATCH: Is the code mixing 'Thousands' and 'Millions' incorrectly?

    RESPONSE PROTOCOL:
    - If the code is perfect: Respond with 'PASS' and a brief summary of the findings.
    - If there is an error: Respond with 'FAIL' followed by a specific 'CORRECTION_INSTRUCTION'. 
    Example: 'FAIL: You used 2.5B for revenue, but Research Artifact #2 shows 26.1B. Update your variable and re-run.'
    """
    ```
    The "Discrepancy" Checklist (The Human-in-the-Loop View)

    Before you run the subgraph, here are the three "Red Flags" your Auditor is now trained to catch:

    Feature | The Hallucination (Current) | The Requirement (Goal)
    -------------- |  ----------- | -----------------
    Revenue Source | `data_center_revenue = 2500` | `data_center_revenue = 26200`
    Output Method (`mean_value`) |  `median_value` |  `print(json.dumps({"mean": mean_value,...}))`
    Variable Injection | `import os (Blocked)` | Use ARTIFACT_DIR constant


    If the Quant Agent fails once, it might get defensive. By passing the exact Research Content and the Auditor's Feedback in the retry prompt, you are giving the LLM a "hint" at the correct answer.

    ```python
    # Inside auditor_node
    if "FAIL" in response.content:
        return {
            "audit_status": "failed",
            "audit_feedback": response.content, # This goes to the Quant Agent next
            "retry_count": state.retry_count + 1
        }
    else:
        return {"audit_status": "passed"}
    ```

1. The "Post-Mortem": Why did it hit Max Retries?

    If the agent failed three times, it usually means one of three things:

    - The Prompt Gap: The Quant Agent isn't "seeing" the research data correctly in its context window.
    - The Auditor is Too Mean: Your Auditor might be flagging something the LLM doesn't know how to fix (like a complex library dependency).
    - Quant is not exactly seeing what to fix. Keep message history updated.

2. How to "Debug" the Loop

    Since you are using LangSmith (or just logging your state), look at the transition between Retry #2 and Retry #3. Check for the "Context Refresh":
    - When the quant_agent ran the second time, did it actually acknowledge the Auditor's feedback?
    - Bad Agent: "I will now run the simulation..." (Ignoring feedback)
    - Good Agent: "I see the Auditor flagged the \$2.5B revenue. I am updating my script to use the $26.2B found in the research."

3. Tuning the "Hand-off" back to the Planner

    If the flow is back at the Scheduler, you have a choice. How should the system handle a "Verified Failure"?

    - Option A: The Re-Plan

        The Planner sees status: "failed" and decides to assign the task to a different agent or skip the simulation and just provide a qualitative summary.

    - Option B: The Human-in-the-Loop (HITL)
    
        At the Max Retries exit, you can insert a "Break" where the system pings you:

            "I tried to value NVIDIA 3 times but my Auditor keeps rejecting the math. Here is the code. Can you fix it?"





###### The "Unit Mismatch" Safety Net

Before you swap the models, check one more thing in your ResearchSummary schema. If your Researcher extracts "26.2" and your Quant Agent writes revenue = 26.2, the math will be off by a factor of a billion.



Senior Tip: Force the Researcher to normalize all currency to Base Units (Dollars).

```Python
class ResearchSummary(BaseModel):
    company_name: str
    revenue_value: float # Should always be the full number (e.g., 26200000000)
    currency: str = "USD"
    unit_multiplier: str = "absolute" # Tell the Quant "No math needed, this is the raw total"
```

##### How the Quant Agent "Learns" from the Failure

When the flow goes back to the quant_agent, the agent needs to see the Auditor's feedback. You should append the Auditor's critique to the messages list so the LLM knows what to fix.

```python
if state.get("audit_status") == "failed":
    feedback_msg = HumanMessage(
        content=f"Your previous code failed audit. Feedback: {state['feedback']}. Please fix the code and try again."
    )
    messages.append(feedback_msg)
```

By saving the code in the Artifact, you create a Self-Contained Audit Trail. This is essential for financial compliance. To pull it out cleanly for your Auditor, you need to navigate that list. 

Here is the most direct "Senior" way to extract it while handling the possibility of multiple tool calls.

###### Extracting from the AIMessage

The response from ainvoke contains a list of tool_calls. You want to find the one named execute_quant_code and grab the code argument.

```python
def extract_code_from_response(response):
    """
    Safely pulls the Python code out of the LLM's tool call.
    """
    if not response.tool_calls:
        return None
        
    # Find the specific tool call for code execution
    quant_call = next(
        (tc for tc in response.tool_calls if tc["name"] == "execute_quant_code"), 
        None
    )
    
    if quant_call:
        # tool_calls[i]["args"] is a dictionary of the arguments passed to the tool
        return quant_call["args"].get("code")
    
    return None

# Usage in your quant_agent node:
generated_code = extract_code_from_response(response)
```

When the auditor returns a retry, the quant_agent_node needs to be smart enough to look for the audit_feedback and fix the code rather than just generating the same "2,500 vs 26,000" hallucination again.

```python
# Check if we are in a retry loop
if state.audit_status == "failed":
    # Inject the failure context into the prompt
    correction_prompt = f"""
    ATTENTION: Your previous code failed the audit.
    FEEDBACK: {state.audit_feedback}
    Please correct the data and logic based on this feedback and the Research Artifacts.
    """
    messages.append(HumanMessage(content=correction_prompt))
```

#### Self-Correction Milestone

Think about what just happened under the hood:

- Attempt 1: The agent was "lazy" or "hallucinated" (the \$2.5B vs $26B ghost).

- Audit: Your adversarial prompt caught the logic/format error.

- Retry: Because you passed the audit_feedback back through Send, the agent had the "context" of its failure.

- Success: It adjusted its code generation to meet the Auditor's requirements.

This loop is exactly how you scale AI to handle high-stakes financial or technical data where a human can't check every single line of code.

As an option, you can also think of a **Deep Search Loop**: If the Quant Agent ever says "I can't find the Interest Expense in the artifacts," have a path to send it back to the Researcher for more specific data!!






--------------------------

The "Model Tier" Strategy

In a production multi-agent system, we usually mix and match models based on the "Cognitive Load" of the task:


Agent | Complexity | Recommended Model | Why?
------| ----------| ------------ | ----------------
Researcher | Low/Medium | GPT-4o-mini / Gemini Flash | Great at extraction and summarization.
Quant Analyst | High | GPT-4o / Gemini Pro | Needs high-fidelity logic and valid Python syntax.
Auditor | High | GPT-4o / Claude 3.5 Sonnet | "Needs to be ""smarter"" than the Quant to catch its lies."
Planner | Medium | GPT-4o-mini | Orchestration is mostly pattern matching.

Small models often struggle with "Contextual Grounding." They "know" what a DCF looks like from their training data, so they revert to "Example" code rather than looking at the research_context you provided. Larger models have better Attention Mechanisms—they prioritize your provided facts over their training "defaults."

----------


### Analyst Node: The Final Synthesis
This is your "closer". The Quant Agent produces raw JSON; the Researcher produces raw facts. The Analyst turns them into a narrative.

The analyst agent should:
1️⃣ Collect outputs from all agents
2️⃣ Merge findings
3️⃣ Produce final report

Example structure:
```ini
Company Overview
Recent News
Financial Metrics
Monte Carlo Risk Simulation
Opportunities
Risks
Investment Summary
```
If the report also includes:
- Monte Carlo chart
- summary statistics

that’s very powerful.

The Task: Take the "Passed" artifacts from the Quant loop and the Research summary and write the Investment Thesis.



ROI Shortcut: Use a very strong model (GPT-4o or Gemini 1.5 Pro) with a Markdown Template. Tell the Analyst: "Fill in this report: Executive Summary, Financial Health, Risks, and Final Recommendation."

3. The "Finish Line" Strategy (The 80/20 Rule)
To wrap this up fast while keeping it "Senior Grade," focus on these three final polish points:

A. The "Success" Artifact

Ensure your final node saves a .md or .pdf file to your Docker volume. There is nothing more satisfying than seeing a professional-looking report appear in your folder after all that debugging.

B. The "Failure" Graceful Exit

You already have the MAX_ITERATION check. Ensure that if it fails, the Analyst node still tries to write a report but adds a "Data Warning" disclaimer. This makes the system "production-ready" because it never just hangs.

C. LangSmith Trace Cleanup

Take one "Perfect Run" where the agent fails, retries, and succeeds. Bookmark it. That trace is your "Proof of Concept" for anyone you show this to (or for your own portfolio). It proves the system is "Smart" enough to self-correct.

The Final "ROI" Checklist

Vector DB: Plug it in as a tool, don't make it a whole subgraph.

Analyst: Give it a strict Markdown template to follow.

Deployment: Run a full "End-to-End" test for one company (e.g., NVIDIA or Apple).

You are 95% of the way there. The "Plumbing" (the hardest part) is done. The last 5% is just letting the Analyst "speak" the results.


##### Vector Database

we should stop treating the Vector DB as a complex infrastructure problem and start treating it as "Long-term Memory" for your Agent.

Since you've already built the Docker sandbox and the Research node, here is the fastest, most meaningful way to "check the box" and see the system working.

1. The "Minimalist" Vector DB Strategy

Instead of a full-blown microservice, use ChromaDB or FAISS (local, file-based) or even a simple In-Memory Store if you just want to prove the logic.

How to find data and know it was retrieved:

The Source: Have your Researcher node take the raw text it finds (from a PDF or a web scrape) and split it into 500-character chunks.

The Injection: Store these chunks in the Vector DB with a metadata tag: {"source": "NVIDIA_2024_10K", "timestamp": "2026-04-06"}.

The Retrieval Trace: To "know" it worked, have your Analyst node call a tool named search_financial_docs.

When the Analyst says: "I need more detail on R&D spend," the tool logs: [VectorDB]: Found 3 chunks for query 'R&D spend'.

The "Aha!" Moment: You will see the Analyst's final report include a specific fact that only existed in the Vector DB chunks, not in the initial summary.


#### Analyst Node

This is where you extract the value. The Analyst should be the "Hedge Fund Partner" who takes the Quant's math and the Research's facts to make a call.

The Analyst Prompt Structure:

"You are a Senior Investment Analyst. Review the Quant Artifacts (DCF results) and the Research Data.

Instructions:

Compare the intrinsic value from the DCF to the current market price.

Cross-reference the Quant's growth assumptions with the Research findings.

Produce a 'BUY/HOLD/SELL' recommendation with 3 supporting bullet points."

Wrapping Up: The "Evaluation" Phase

To "finish it up" and feel confident, you need to evaluate if the agent is actually right or just confident.

Evaluation Metric | How to Check
---------- | -------------
Data Grounding | "Did the Analyst mention the \$26B revenue (The ""Correct"" number) or the $2.5B (The ""Ghost"" number)?"
Logic Consistency | "Does the ""Buy"" recommendation actually follow from the DCF math | or is the LLM just being optimistic?"
Hallucination Rate | "Did the Vector DB retrieval actually provide new info, or did the LLM just use its internal training data?"

My Suggestion: Don't build a separate Vector DB node. Just give the Analyst a search_vector_store tool. If the Analyst is "confused" by the Quant's numbers, it will use the tool to look up the source text. This is much faster to code and looks more "agentic" in a demo.

Building a production-ready RAG pipeline is a massive time-sink, and honestly, for a single investment report, Context Stuffing (putting the raw research into the Analyst's prompt) is often more accurate than Vector Retrieval anyway. RAG is great for "chatting with 1,000 PDFs." But for Investment Research, you only care about one company at a time. Passing the full 20-page 10-K text (or a dense summary) directly to a long-context model (like GPT-4o or Gemini 1.5 Pro) is more reliable than hoping a Vector DB picks the right "chunks." In 2026, with 128k+ context windows being the norm, "Context Stuffing" your validated research is significantly more reliable for a single-company report than a RAG system that might miss a needle in the haystack.

If you skip the vector_db, the "High ROI" finish is to turn your Analyst into the "Final Judge" who evaluates the whole pipeline's work.

1. The Analyst as the "Evaluator"
Since the Analyst sits at the end of the graph, it has access to the artifacts list. This is where you extract the most value. Instead of just "summarizing," have the Analyst perform a Cross-Check Evaluation:

Math vs. Fact: Does the revenue in the Quant's JSON match the revenue in the Researcher's summary?

Sensitivity Check: If the Quant's DCF is $500 but the stock is $150, is the growth rate (e.g., 25%) realistic based on the Research node's findings?

The "Hallucination" Filter: The Analyst should flag any number that appears in the DCF but cannot be found in the Research artifacts.

2. The "Reporting" Pipeline (The Output)
To "wrap it up" with something you can actually show off, the Analyst should output a Structured Markdown Report. Save this to your Docker volume as final_report.md.

The "Senior" Report Template:

Investment Thesis: (Buy/Hold/Sell)

Quant Valuation: (Intrinsic Value vs. Current Price)

Audit Trail: "The Quant Agent's math was verified by the Auditor Agent (2 retries required)."

Key Risks: (Derived from the Researcher's findings)

3. The "Evaluation" Pipeline (The Meta-Value)
If you want to add "valuation" at the end, use a LLM-as-a-Judge pattern. Create a tiny standalone script (or one final node) that takes the final_report.md and grades it against a "Golden Rule":

Accuracy: Did it catch the $26B vs $2.5B error?

Completeness: Did it address all tasks in the original plan?

Formatting: Is the JSON in the appendix valid?

4. Final Architecture: The "Victory Lap"
Your graph now looks like this:

Planner: Sets the 5-6 tasks.

Researcher: Pulls the raw data.

Quant Analyst + Auditor: (The loop you just perfected) generates the math.

Analyst: Synthesizes everything and evaluates the logic.

END: Writes the file to disk.

--------

Analyst Agent. This node will act as the "Chief Investment Officer," taking the artifacts and math to produce the final evaluation.

The "Evaluation" Logic (The High ROI)

Since you wanted an evaluation pipeline, we can add a specific Scoring Function inside this node or as a final step. This is what makes the project "Professional Grade."

Add this to your Analyst prompt to force self-correction:

"In your EVALUATION section, specifically check:

Data Lineage: Did the $26B figure survive from the Research node to the final DCF?

Agent Friction: Did the Auditor catch errors that improved the final result? (Cite specific retries)."


##### Notebook as frontend
Since the notebook is your frontend, you need a clean, reactive interface that separates the "System Logs" from the "Agent Thoughts."

Here are the three high-ROI ways to clean up your graph.stream() output for a professional notebook experience.

Since you are in a notebook, take advantage of rich formatting. You can use IPython.display to render the agent's output as formatted Markdown in real-time, which is much easier on the eyes than raw strings. If your graph takes a few minutes to run (because of the Quant/Auditor loop), use a simple tracker so you know the graph hasn't hung.

Why this matters

A clean "Frontend" in your notebook isn't just for aesthetics; it helps you catch logic bugs faster. When the stream is messy, you might miss that the Quant Agent is looping 4 times instead of 2. When it's clean, the "❌ AUDIT FAILED" jumps out at you immediately.


### Analyst and Auditor

This is a classic case of "Training Data Interference." GPT-4o (and most large models) "knows" what an NVIDIA DCF looks like because it was trained on thousands of them. When it sees the task "Perform NVIDIA Analysis," it defaults to its internal "memory" of numbers rather than looking at the specific research_context you provided.

To the model, those mock numbers feel "right" because they are part of its weights, while your Research Data is just a temporary string in its context window. We need to flip that hierarchy.

1. The "Zero-Tolerance" Prompt Update

You need to move from "Please use the data" to "You are forbidden from using any numbers not in the provided context."


If the model is still being stubborn after 3 retries, it’s usually because the research_context is too long or messy. The "Senior" way to fix this is to Pre-Process the Auditor's feedback and inject it directly into the HumanMessage for the retry.

```python
# If this is a retry, force the Auditor's 'Truth' to the top of the message
if state.get("retry_count", 0) > 0:
    correction_trigger = f"""
    CRITICAL CORRECTION:
    The Auditor has flagged your previous numbers as HALLUCINATIONS.
    YOU MUST USE THESE FIGURES: {state['audit_feedback']}
    Do not use 15.0B, 50.0B, or 115.0B. Those are fake.
    """
    messages.insert(0, HumanMessage(content=correction_trigger))
```

- I get weird feedback from auditor again and again ...why auditor is being stupid:
FAIL: You used 39.1B for Q1 Fiscal 2026 data center revenue, but Research Artifact shows 39.1B. Update your variable and re-run. Additionally, the script failed to print the results in a structured format (JSON).

This is the "Equality Paradox" in LLM reasoning. When an LLM compares two numbers that are identical, it sometimes fails a "semantic check" because of how the numbers are formatted or stored in its internal state.

The Auditor is essentially saying: "You used 39.1B, but the source says 39.1B. You are wrong." This usually happens for one of three technical reasons:

1. The "Hidden Characters" or Type Mismatch

The Auditor might be seeing the Quant Agent's variable as a String "39.1B" while the Research Artifact has it as a Float 39.1. To the LLM, these are "different," even though the value is the same.

The Fix: Update the Auditor's System Prompt to be more forgiving of formatting:

"Before flagging a data mismatch, normalize both numbers to floats. If the Quant Agent uses '39.1B' and the Research says '39.1', this is a MATCH. Do not trigger a failure for formatting differences."

2. The "Negative Feedback Loop" (Model Fatigue)

Once an Agent fails 3 times, its "Context Window" is filled with its own errors. It starts to "hallucinate failure." It sees the word "FAIL" so many times in its history that it assumes it must fail again to stay consistent with the conversation.


Look at the second half of that error: "Additionally, the script failed to print the results in a structured format (JSON)."

LLMs often "bundle" errors. If the Quant Agent failed to print JSON, the Auditor might be getting "grumpy" and throwing a random data error just to justify the "FAIL" status.

The Fix for the Quant Agent:
You must wrap your Python code execution in a way that forces the output into a dictionary.

```ini
# Add this to the Quant Agent's instructions:
"""
Your script MUST end with:
import json
print(json.dumps(results_dictionary))
"""
```

The Auditor isn't being "stupid"; it's being a Literalist. It's comparing the "Token" for 39.1B (one or two tokens) against the "Token" for 39.1 plus the "Token" for B. If they don't match exactly in its hidden state, it flags a mismatch.

Immediate Action Plan:

Relax the Auditor: Tell the Auditor: "If the numbers are mathematically equivalent (e.g., 39.1B == 39.1), consider it a success."

Clean the Quant Output: Ensure the Quant Agent is printing ONLY the JSON at the very end of its execution. If it prints "Here is the code..." or other chatter, the Auditor might struggle to parse the math.

The "Truth" Injection: In the next retry, have the Scheduler/Router inject a "Truth Table" into the prompt:

EXPECTED_DC_REV: 39.1B

ACTUAL_DC_REV: 39.1B

STATUS: MATCH

- first attempt model generated the code but didnt make a tool call?!!! the router caught that and returned it to the node, this time it made a tool call and the auditor rejected it by mistake:
in the code : data_center_revenue_q1_fy2026 = 39100 #

his is the classic "Hallucination of Mismatch" that happens when an LLM acts as a judge. Because the model "sees" the number 39100.0 (math) and the string 39.1 billion (text) as different tokens, it lacks the "common sense" to realize they are identical values in different units. It is being a pedantic clerk instead of an engineer.


Change the Auditor's "Persona"

The Auditor is being too "aggressive" because it's looking for excuses to fail the task. Change its instructions from "Find Errors" to "Verify Equivalence."

Update the Auditor Prompt:

"You are a specialized Verification Agent. Your ONLY job is to ensure the numbers in the code represent the same physical value as the Research Data.

IMPORTANT: 39100 Million is EQUAL to 39.1 Billion. If you see this, it is a MATCH. Do not reject the code for unit differences if the underlying value is correct."


It feels like watching two geniuses argue over whether the glass is "half full" or "50% occupied." Even though both nodes are powered by GPT-4o, they are suffering from Role-Induced Cognitive Bias.

When you give the same model two different "System Personas," you are essentially forcing it to look at the same data through two different lenses.

1. The Persona Paradox

The Quant Agent is in "Creation Mode." It is optimized for code syntax and financial conventions. In the world of financial modeling, "Millions" is the industry standard. Its internal weights are pulling it toward 39100 because that's how 90% of the DCFs in its training data look.

The Auditor is in "Destruction Mode." You told it to find mismatches. When it compares the string "39.1 Billion" in the research to the integer 39100 in the code, it doesn't use a calculator—it uses token matching. Because the tokens don't align, its "Critique" persona takes over and it flags it as a hallucination.

2. The "Tokenization" Trap

To an LLM, the number 39.1 and 39100 are completely different sequences of tokens.

39.1 -> [39, ., 1]

39100 -> [391, 00]

Unless the model explicitly "pauses" to perform a floating-point math conversion (which it rarely does during a text comparison task), it perceives them as a Semantic Mismatch.


The Quant Prompt Fix (Force the Unit)

Update the Quant Agent to be "Unit-Aware":

"Always use the exact unit found in the Research Data. If the research says '39.1 billion', your variable must be 39.1. Do not convert to millions or thousands."

B. The Auditor Prompt Fix (Force Mathematical Logic)

Update the Auditor to be "Unit-Agnostic":

"You are an intelligent auditor. If the code uses 39100 and the research says 39.1 billion, recognize that these are mathematically identical. DO NOT flag this as an error. Only flag errors where the underlying value is fundamentally wrong."


The "Self-Correction" Reducer (The Formatting Trick)

If you don't want to build a whole tool, you can use a Chain-of-Thought (CoT) technique specifically for numbers. You must force the Auditor to "show its work" in a scratchpad before it issues a FAIL.

Update the Auditor Prompt:

"Before you decide if a number is a hallucination, you MUST follow these steps:

Write down the number from the Code.

Write down the number from the Research.

Perform a 'Unit Check': Could one be in millions and the other in billions?

Multiply/Divide to check for equivalence.

If Code_Value * 1000 == Research_Value (or vice versa), they are the SAME. Mark as PASS."

By providing a Concrete Counter-Example in the prompt, you are utilizing "Few-Shot Provocation." You’re essentially re-wiring the model's logic to ignore its default string-matching behavior in favor of the specific heuristic you've defined.

Why this specific prompt works:

Normalization Instruction: By telling the model to "normalize both numbers to floats," you're forcing it to invoke its internal reasoning engine (the "Math" part of the brain) before the token-matching engine (the "Grammar" part of the brain) can trigger a "FAIL."

Explicit Mapping: Mapping 39100 to 39.1 in the prompt gives the model a "Lookup Table" for what "Correct" looks like in this specific domain.

Commanding Language: "Just PASS it" reduces the model's tendency to be over-critical (a common trait in "Auditor" personas).


1. The Quant Analyst Update (The "Truth Priority")

The Quant agent needs clear instructions on what to do when the numbers in the text don't match the math. We also need to force the "Billions" standard so the Auditor doesn't get confused by "Millions" ever again.

Add these rules to QUANT_ANALYST_PROMPT:

UNIT STANDARD: You MUST use the units found in the Research Data (usually Billions). Do not convert to Millions unless the research specifically uses Millions.

MATH > TEXT: If the research provides both a growth rate (e.g., 75%) and the raw revenue numbers, and they do not mathematically align, use the raw revenue numbers to calculate your own growth rate.

EXPLAIN DISCREPANCIES: If you find a data conflict, add a discrepancy_note key to your summary dictionary explaining your choice.

2. The Auditor Update (The "Reasoning" Bridge)

The Auditor is currently acting like a Regex pattern-matcher. We need to give it "Financial Common Sense" so it stops flagging mathematically correct conversions as hallucinations.

Update the AUDITOR_SYSTEM_PROMPT Response Protocol:

NEW VERIFICATION LOGIC:

Unit Check: 39,100M, 39.1B, and 39,100,000,000 are the SAME value. Do not fail for these.

Discrepancy Check: If the agent provided a discrepancy_note in the JSON, and that note explains why a calculation (like 329% growth) differs from the text (75% growth) based on the raw revenue, ACCEPT THE MATH.

JSON Extraction: You must look at the string inside STDOUT. If that string contains a valid JSON block, even if it has other text around it, it is a PASS on the format check.


- im wondering why the model doesnt make too cal at the first attempt - it needs a human message to tell it to make a calls ... that's what's happening now ... I can only think of expanded context ??


1. The "Lost in the Middle" Effect (Expanded Context)

When you have a massive research_context followed by a long QUANT_ANALYST_PROMPT, the instruction to use a tool gets buried. In 2026, even with 128k+ windows, LLMs still have limited attention weights. If the "Meat" of the research data is in the middle, the model's brain is exhausted by the time it reaches the "Mandatory Rules" at the end.

The Fix: Move the Tool Call Instruction to the very bottom of the prompt, right before the model is supposed to start generating.

"You have identified the data. NOW, use the execute_quant_code tool immediately to process it. Do not provide a conversational preamble."

2. The "Chatty Assistant" Bias

GPT-4o is fine-tuned to be helpful and conversational. By default, it wants to explain what it's about to do.

Attempt 1: It thinks, "I'll explain the plan first, then do the math." It generates text, which satisfies its "helpful" instinct.

Attempt 2 (The Human Message): When you send a human message saying "Make the call," you are effectively resetting its Objective Function. You’ve cleared the ambiguity.

3. Missing tool_choice="required"

If you are using LangChain or the OpenAI API directly, you can force the model's hand. If you know the Quant Agent must run code to finish its task, don't give it the option to just talk.

The Technical Fix:
When calling the model in your quant_node, use the tool_choice parameter:

```Python
# This forces the model to pick at least one tool
# The correct way to force a specific named tool
response = await tool_model.ainvoke(
    messages, 
    tool_choice={"type": "function", "function": {"name": "execute_quant_code"}}
)
```

4. Semantic Satiation (Instruction Overload)

Look at your QUANT_ANALYST_PROMPT. You have 8+ Mandatory Rules. When every sentence is "CRITICAL" or "MANDATORY," nothing is. The model starts to treat the instructions as "background noise."

5. The "System" vs. "Human" Message Split

Sometimes, models respect instructions in a HumanMessage more than a SystemMessage.

System Prompt: "You are an agent who uses tools." (Abstract)

Human Message: "Analyze NVIDIA. Use your tool." (Direct Command)

Try this "Senior" Prompt Strategy:
Keep the System Prompt short (Persona only). Put the specific task and the "Mandatory Tool Use" rule in the HumanMessage that wraps the research_context.

Summary Checklist to fix the "First-Try" hesitation:

Force the tool: Use tool_choice="required" in your code.

Trim the Fat: Remove the "Do not import os" type rules from the middle and put them in a dedicated "Technical Constraints" block at the end.

The "Act, Don't Talk" Rule: Explicitly state: "Your response MUST start with a tool call. Conversational text is prohibited."


Why the model "hesitates" without this

Since you mentioned the model requires a human message to wake up, here is what’s happening under the hood:

Instruction Overload: Your prompt is very "heavy." GPT-4o sometimes gets "lazy" when it sees a massive context followed by complex code instructions. It defaults to a conversational summary because that's the "safest" path in its training.

The "System" vs. "Tool" Priority: Without tool_choice, the model weights the probability of "talking" vs. "calling." If the research context contains a lot of prose, the model's "probability" shifts toward generating more prose.

Pro-Tip: Clean up the Prompt for "First-Try" Success

If you find that forcing the tool results in a "dead" response (where the model calls the tool but provides empty arguments), try moving your MANDATORY RULES to the very top of the HumanMessage instead of the System Prompt.

Structure it like this:

Human Message Start: "URGENT: Use the execute_quant_code tool now to analyze the following data."

The Context: {research_context}

The Task: {task_description}


### End

######  How to "Sell" this on your Resume
Don't just say "Built a chatbot." Frame it as a Multi-Agent Orchestration Framework. Use these bullet points:

Engineered a Multi-Agent Financial Research System using LangGraph and GPT-4o, achieving a 100% first-pass success rate in quantitative verification through custom deterministic auditing protocols.

Architected a "Reflective Auditor" Node to eliminate LLM hallucinations in numerical reasoning, implementing unit normalization (Millions vs. Billions) and mathematical cross-checking.

Optimized Agentic Reliability by implementing forced tool-calling schemas and "Conflict Resolution" prompts, reducing "instruction drift" and internal logic loops in complex 5-node workflows.

Infrastructure: Deployed via Docker/MCP with automated Python code execution and artifact generation (Matplotlib/Pandas), ensuring 100% reproducible quantitative analysis.

2. Most Important Extensions (Interview Talk)
When they ask, "What would you do next?", these three answers show you understand the Production-Grade AI landscape:

A. The "Knowledge Graph" Bridge

The Problem: Right now, the Researcher just sends a block of text.
The Extension: Implement a Knowledge Graph (Neo4j). Instead of raw text, the Researcher extracts "Entities" (NVIDIA, Q4 Rev, $62.3B) and "Relationships" into a graph.

Interview Hook: "I'd move from raw RAG to GraphRAG to ensure that when the Quant Agent looks for 'Revenue,' it's pulling a verified node from a graph rather than searching through a 50-page PDF."

B. "Human-in-the-Loop" (HITL) Checkpoints

The Problem: If the data is truly contradictory, the agents guess.
The Extension: Add an "Interrupt" state in LangGraph. If the Auditor finds a mismatch that exceeds a 10% threshold, the graph pauses and sends a message to a human (via Slack/Email) to "Verify Ground Truth."

Interview Hook: "In high-stakes finance, you can't have 100% autonomy. I’d implement HITL states to handle 'ambiguity edge cases' that the LLM identifies but can't resolve."

C. Multi-Modal Verification

The Problem: Sometimes the best data is in an image (a chart in an earnings call deck).
The Extension: Give the Researcher a Vision-capable node to scrape charts and tables directly, rather than relying on OCR text which often garbles numbers.

Interview Hook: "I’d integrate Vision models to verify the 'Quantitative summary' against actual JPEG charts from investor relations, creating a multi-modal source of truth."

3. Preparation for the Presentation
If they ask you to walk through the code, be ready to explain the route_audit logic. That is the "brain" of your system.

Explain the "Equality Paradox" (39.1 vs 39100).

Explain how you used systematic prompt engineering to force the model to behave as an "Accountant" rather than a "Writer."

You’ve built something that demonstrates Orchestration, Guardrails, and Data Integrity—the three things every ML hiring manager is looking for right now.

Good luck with the resume and the "survival" phase—based on this project, you’ve got the technical depth to land something great. You've got this!

--------------

##### Building the MCP Endpoints

Since you are using MCP to bridge LangGraph with a Dockerized Sandbox, you need to define the Tools that the LLM can "see." In a senior-level implementation, these tools should be granular and secure.

- Endpoint A: `execute_simulation` (The Core)

    This tool sends the raw Python string to your MCP server, which manages the Docker lifecycle.
    ```ini
    Input: 
      code: str, 
      timeout: int = 30
    ```

    - Logic: 
        1.  Wrap the code in a try-except block within the container.
        2.  Execute using subprocess or a dedicated Python runner.
        3.  Capture `stdout`, `stderr`, and any files created in the `/mnt/artifacts` volume.

    - Output: A dictionary containing the numeric results and paths to generated charts.

- Endpoint B: `get_sandbox_environment` (The Context)

    Before writing code, the agent might need to know which libraries are available to avoid ImportError.
    ```ini
    Input: None
    ```

    - Output: List of installed packages (e.g., `['pandas==2.1.0', 'scipy==1.11.0']`).


A high-ROI move here is to ensure your MCP server maintains *Session Persistence*.

Instead of:
`Spin up Container -> Run Code -> Kill Container`

Use:
`Persistent Container -> Run Code in Namespace -> Keep Variables`

This allows the Quantitative Agent to run a "Preliminary Trend Analysis," look at the results, and then run a "Monte Carlo Simulation" using the variables already calculated and stored in the container's memory. This mimics a Professional Quant Workflow.

4. Code Snippet: Defining the Tool in LangGraph

This is how you would register the MCP tool within your agent's configuration:

```Python
from langchain_core.tools import tool

@tool
def run_python_analysis(code: str):
    """
    Executes Python code in a secure, isolated Docker container via MCP.
    Use this for all mathematical modeling, DCF, and simulations.
    Returns: JSON string with 'status', 'output', and 'artifact_paths'.
    """
    # This calls your MCP Server Endpoint
    response = mcp_client.call_tool("docker_sandbox", "execute", {"code": code})
    return response
```

#### Tool Calls by LLMs
1. Does the LLM call the tool "Directly"?

Technically, no LLM ever "calls" a tool.

An LLM is just a text predictor. When we say an LLM "calls a tool," what actually happens is:

- The LLM outputs a specifically formatted string (usually JSON) that says: {"tool": "execute_quant_code", "args": {"code": "print(1+1)"}}.

- Your Python code (the LangGraph Node) sees that string, stops the LLM, and manually executes the function against the MCP Sandbox.

Your code then feeds the result back to the LLM.

2. Why we "Intervene" (The Senior Defense)

Even if there is only one tool, we don't just "pass through" the call. We intervene in the quant_node for three reasons:

- Security (Validation): Before sending the code to the Docker sandbox, your Python node can "inspect" the code. For example, you can use a Regex to block the LLM if it tries to write code that accesses os.environ or tries to delete files.

- State Injection: The LLM might write df = pd.read_csv('data.csv'). The LLM doesn't know where data.csv is. Your Python node handles the "plumbing"—it makes sure the file is actually in the Docker volume before the tool runs.

- Error Orchestration: If the tool fails, we don't want the whole program to crash. We catch the Python error in our "manual" wrapper and turn it into a polite message: "Hey LLM, your code had a SyntaxError. Try again."

3. What does llm_output contain?

In a standard LangChain/LangGraph setup, the llm_output (an `AIMessage`) contains:

- content: The "thinking" or "prose" explaining what it's doing.

- tool_calls: A list of structured objects.

Example of llm_output.tool_calls:

```JSON
[
  {
    "name": "execute_quant_code",
    "args": {
      "code": "import numpy as np\n# Monte Carlo logic here..."
    },
    "id": "call_123"
  }
]
```

4. Making the Tools Clear to the LLM

The LLM must know the tools exist to "choose" them. When you initialize your LLM in the quant_node, you "bind" the tools to it. This adds the tool definitions to the `System Prompt` automatically under the hood.

```Python
# This tells the LLM: "You have this specific tool available. 
# Use it if you need to do math."
tools = [execute_quant_code]
quant_llm_with_tools = llm.bind_tools(tools)
```
The "Senior" Logic Flow

- Input: Supervisor says "Do a DCF."

- LLM Decision: LLM thinks, "I can't do math, but I have the execute_quant_code tool. I'll use that."

- Output: LLM returns a tool_call.

- Execution: Your LangGraph node executes that code in the MCP Sandbox.

- Loop: If the math is wrong, the Auditor (which we'll build next) sees the code in the state and tells the Quant agent to fix the formula.


##### Tool Calls using MCP server
Chaining `bind_tools()` and `with_structured_output()` often leads to "schema confusion" because both methods fight for the model's output control. When you use `with_structured_output()`, it often overwrites the tool bindings you just set. 

Models struggle to validate tool arguments while simultaneously trying to fit a top-level response schema. This is called Schema Conflict. Also in LangChain, `with_structured_output` creates a new runnable that typically ignores any bind_tools previously called on the base model (Lost Bindings)

The standard "LangChain way" to solve this is to treat your structured output as another tool or to use a loop that handles both. Here are the two most effective ways to fix your logic:

###### The Unified Model (Recommended)
Instead of chaining the two methods, some providers allow you to pass tools inside the `with_structured_output` call. This tells the model it can either call a tool OR provide the final structured response. 

```python
# NOT TESTED
# Pass tools directly into the structured output configuration
unified_model = model.with_structured_output(
    ResearchSummary, 
    include_raw=True, 
    tools=research_tools  # Pass your MCP tools here
)

response = await unified_model.ainvoke(messages)

# The result will contain either 'parsed' (your schema) or 'raw' (which may have tool_calls)
if response['raw'].tool_calls:
    # Handle the MCP tool execution here as you normally would
    pass
elif response['parsed']:
    # You have your final ResearchSummary object
    return response['parsed']
```

######  Manual Sequential State (TESTED)
If you prefer to keep them separate, you need to ensure the second call (extraction) has all the context from the first tool's result.
- Use `tool_model.ainvoke` to get the `tool_calls`.
- Execute the tool and append the `ToolMessage` result to your messages list.
- Call the model with structured output with the full history, including the tool's findings. 









To handle tool execution manually using MCP (Model Context Protocol) in a LangChain-like flow, you need to bridge the gap between the model's "intent" (the tool_calls object) and the actual server execution.  When using an MCP client, the tools it returns via `get_tools()` are essentially LangChain Tool objects. These objects have a `.invoke()` or `.ainvoke()` method that handles the protocol communication for you. 

Here is how to implement the Manual Sequential State to fix your issue:

Step 1: Initialize Messages and Client
Maintain a list of messages to preserve context. Ensure your mcp_manager is ready to provide tools.
```python
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
import json

# 1. Setup history
messages = [
    SystemMessage(content=RESEARCHER_PROMPT.format(topic=topic)),
    HumanMessage(content=f"Task: {task.description}\nUse the available web search tool...")
]

# 2. Bind tools to the base model
research_tools = await mcp_manager.get_tools()
tool_model = model.bind_tools(research_tools)
```

Step 2: Request the Tool Call
Invoke the model. Because you haven't forced a structured output yet, the model is free to choose the search tool.
```python
response = await tool_model.ainvoke(messages)
messages.append(response) # Keep the assistant's request in history
```

Step 3: Execute the MCP Tool
If the model generated `tool_calls`, loop through them. Use the tool objects directly to talk to the MCP server.
```python
if response.tool_calls:
    for tool_call in response.tool_calls:
        # Find the matching tool from your research_tools list
        selected_tool = next(t for t in research_tools if t.name == tool_call["name"])
        
        # This triggers the actual MCP protocol call to the server
        tool_output = await selected_tool.ainvoke(tool_call["args"])
        
        # Add the result to history so the model sees what it found. Map the result back using the unique ID
        messages.append(ToolMessage(
            tool_call_id=tool_call["id"], # CRITICAL: must match the model's ID
            content=str(tool_output),
            name=tool_call["name"]
        ))
```

Step 4: Extract Structured Output
Now that the search results are in your messages list, call the model with your structured output constraint. The model now has the data it needs to fill your schema.

```python
# Create the extractor constrained to your schema
    extractor = model.with_structured_output(ResearchSummary)

    # Pass the FULL message history (System + Task + Tool Call + Tool Result)
    final_extraction = await extractor.ainvoke(messages)
return final_extraction
```

Key Takeaways
- Don't skip the `ToolMessage`: The model needs to see the output of the tool linked to its specific tool_call_id to understand that the "research phase" is over.
- `selected_tool.ainvoke`: You don't need to write raw JSON protocol code. The LangChain MCP adapter wraps the tool so that .ainvoke() handles the MCP communication automatically.
- Separation of Concerns: By splitting the "Research" (Tool-enabled) and "Extraction" (Schema-enabled) phases, you avoid the conflict where the model ignores tools to satisfy a strict output format. 



The reason you iterate over `response.tool_calls` is to support **parallel tool calling**. Modern models (like GPT-4o or Claude 3.5 Sonnet) can suggest multiple actions in a single turn—for example, searching for two different topics at once. 


However, from a Senior Architecture and Debugging perspective, you absolutely should add tool mesages. If you don't record the ToolMessage, your conversation history becomes "broken." If the agent needs to perform a second step based on the calculation (e.g., "The result was X, now calculate Y"), the LLM will see its own request to run a tool, but it will never see the answer.

###### Why you must add ToolMessages

- Context for the LLM: If the first tool call returns an error or a specific number, the LLM needs that ToolMessage in its history to decide what to do next.

- The "Auditor" Agent: Your Auditor agent needs to see the entire trace (The thought → The code → The output) to verify the logic.

- LangSmith / Debugging: Without the ToolMessage, your traces in LangSmith will look like the agent asked a question and then went silent.


In a LangGraph environment, you typically have two choices for storing history:

- The messages key: If your `AgentState` has a messages list, you should append these there. This is standard for "Chat" style agents.

- The Artifact key (Your approach): Since you are using a structured Artifact list, you are already capturing the "result."

Senior Recommendation: Keep the full message history local to the node's execution (to help the agent finish its task), but only save the Final Result and the Code Snippet to the global Artifacts list. This keeps your global state clean while giving the agent the "short-term memory" it needs to work.

Again, since you are updating your Artifact with this `mcp_response`, adding the `ToolMessage` to the messages list is primarily to help the LLM finish its thought.

Without the ToolMessage, the LLM "calls" the tool and then the node returns. The LLM never gets to say: "Based on the calculation, the DCF shows a 15% upside." Senior Implementation Tip: 
1. Call the tool.
2. Add ToolMessage to messages.
3. Call LLM one last time (with no tools bound) to get a "Human-readable summary."
4. Save that summary + the MCP data into your Artifact.

###### How to handle multiple/repeated tool calls correctly:
If the model calls the same tool three times with different arguments, your loop handles it like this:
- Iterate: for t`ool_call` in `response.tool_calls`: loops through every request the model made.
- Find the Tool: `next(t for t in research_tools if t.name == tool_call["name"])` finds the actual executable code for that tool name. Even if the tool is called three times, it's the same tool definition being used each time.
- Execute & Record: You call await `selected_tool.ainvoke(tool_call["args"])` for each individual call.
- Unique IDs: Crucially, you must use the `tool_call_id` from the model's request when creating your `ToolMessage`. This tells the model exactly which result belongs to which of its three requests. 


Note on MCP IDs: Some early versions of MCP adapters had bugs where simultaneous calls to the same tool used the same ID, causing "hanging" responses. Ensure your `langchain-mcp-adapters` package is up to date to avoid this. 


##### MCP persistent connection
Why your FastAPI lifespan is perfect
Your lifespan code is exactly correct—it provides the stable environment needed to hold these sessions open. By calling `mcp_manager.startup()` inside that async with pool block, you ensure: 

- Checkpointers are ready before the agents start.
- MCP is fully initialized before the first request hits.
- LIFO Cleanup: When the app stops, `mcp_manager.shutdown()` runs first, closing the session gracefully before the database pool closes.

------






------------

High-ROI Addition: The "Interface" Server

Since you mentioned adding Redis, here is the Senior-level move: Use Redis not just for caching, but as a *Message Broker* or *Task Queue* if your simulations get heavy.

Why? If the Quant Agent triggers a 100,000-iteration Monte Carlo simulation, you don't want the LangGraph HTTP request to time out.

The Pattern: LangGraph pushes a "Job" to Redis → Sandbox Server pulls and executes → LangGraph polls for the "Artifact."

(Note: For a MVP, direct MCP calls are fine, but mentioning this **Asynchronous Scaling** during a defense is pure gold.)


The "Stateful Checkpoint" for Quantitative Logic.
Since you have Postgres for LangGraph checkpoints, make sure your Quant Agent saves the logic (the actual Python code) into the thread state.

The Senior Reason: If a user asks, "Why did you value NVIDIA at $1100?", you can pull the exact script from Postgres and show the math. This is called Auditability.

the `docker-compose.yml` needs to reflect Network Isolation. You don't want your Sandbox (which runs untrusted LLM code) to have the same network access as your Postgres database containing user history.

###### Separate sandbox network
Here is the high-ROI configuration for your architecture.

1. The `docker-compose.yml `Strategy

We will use two separate networks:

- `agent-net`: For your Graph, Postgres, and Redis (Internal/Sensitive).

- `sandbox-net`: A restricted bridge only between the LangGraph API and the MCP Sandbox.

Isolation. By decoupling the execution environment, we ensure that a failure in a complex Monte Carlo simulation (like a memory leak) doesn't crash the entire orchestration engine or expose our database credentials to the LLM's runtime environment.

#### Why FastAPI

keeping FastAPI as a wrapper is definitely the right move for an investment researcher, especially in 2026.

While the langgraph build command is great for standardized "Agent Server" deployments, a custom FastAPI wrapper is "worth it" because it gives you:

Custom Authentication: Crucial if your investment data becomes sensitive.

Response Shaping: You can pre-process raw graph outputs into clean JSON schemas for your Notebook.

Lifespan Management: Essential for your MCP clients. You can open/close those persistent connections exactly when the container starts/stops.

Here is the exact focus on building that image.

1. The Multi-Stage Dockerfile (Optimized)

Since you're in a Dev Container (DooD), we’ll use a multi-stage build to keep the final image small but ensure all C-extensions for Postgres/Redis are compiled correctly.



----------
In 2026, the industry standard has moved toward Multi-Agent Orchestration (MAO) and Model Context Protocol (MCP) for tool interoperability.

1. Essential Core Concepts (The "Senior" Layer)

Review these concepts to articulate architectural decisions during your interview:

Flow Engineering vs. Prompt Engineering: Understand that for senior roles, the "flow" (state management, conditional edges, and loops) is more important than the "prompt." Learn how to use Directed Acyclic Graphs (DAGs) and stateful cycles.

Design Patterns:

Reflection/Self-Correction: An agent generates a response, critiques its own work (or has another agent critique it), and iterates.

Orchestrator-Worker: A "Manager" agent decomposes a goal into sub-tasks and delegates them to specialized "Worker" agents.

Evaluator-Optimizer: Using a two-step process where one agent proposes and another refines based on a rubric.

Memory & State Management: Moving beyond short-term "buffer" memory to Long-term Persistence using vector databases (like Weaviate or Pinecone) and Agentic Checkpointing (the ability to "pause" and "resume" a long-running agent task).

Human-in-the-Loop (HITL): Designing "Interrupt Points" where an agent must wait for human approval for high-risk actions (e.g., executing a financial transaction).

2. The Recommended "Senior" Stack (2026)

Orchestration: LangGraph (for graph-based state control) or CrewAI (for role-playing multi-agent systems).

Agent Communication: MCP (Model Context Protocol)—this is the emerging standard for how agents connect to data and tools.

Observability: LangSmith or Arize Phoenix. You must show you can debug agent reasoning loops.

Inference: DeepSeek-R1 (for high-reasoning tasks) or Claude 3.5/4 (for tool calling and coding).

3. Comprehensive Project: "The Autonomous Enterprise Analyst"

Building a simple research bot isn't enough for a senior role. Build a Multi-Agent Risk & Strategy Engine that performs real-world, high-stakes tasks.

Project Scope:

An autonomous system that takes a company name and a specific market event (e.g., "NVIDIA's Q3 earnings report") and generates a 10-page investment thesis with zero human intervention.

Architecture:

The Researcher Agent (Tool User): Uses Tavily or Exa to find real-time financial data and 10-K filings.

The Auditor Agent (Reflection): Reviews the Researcher's sources for hallucination or outdated data.

The Quantitative Agent (Code Interpreter): Writes and executes Python code (using a sandbox) to perform Monte Carlo simulations or trend projections on the data.

The Supervisor Agent (Orchestrator): Manages the state, ensures the "Quantitative" agent's math matches the "Researcher's" context, and compiles the final report.

Interview "Selling Points":

Error Handling: "I implemented a retry logic where if the Quantitative Agent's code fails, the Auditor Agent analyzes the stack trace and suggests a fix."

Cost Optimization (FinOps): "I used a Reasoning Model (like DeepSeek-R1) for the strategy phase but switched to a Lighter Model (like GPT-4o-mini) for the summarization tasks to reduce token costs by 70%."

Persistence: "I used LangGraph's checkpointer to ensure that if the API rate-limited the agent 4 hours into the task, it could resume exactly where it left off."


es, but to make it Senior-level, you need to solve for Agentic Reliability. In your project, include:

Semantic Guardrails: Use a library like Pydantic to enforce strict schemas on agent communication. If Agent A sends a malformed JSON to Agent B, the "Graph" should catch it and trigger a "Retry" node.

The "Token Budget" Monitor: Add a node that tracks cumulative cost. If the agents start looping infinitely (an "Agentic Loop"), the system self-terminates.

Traceability: Use LangSmith to show the "Chain of Thought" for a specific decision. This is what interviewers want to see: How do you debug a black box?



###### Options for executing tool calls
Great question, Yas. This touches a subtle but important difference between LLM-native tool usage and framework-managed tools.
Short answer:

No — you don’t always need a ToolNode for tools.
But inside LangGraph, it is usually the correct way to execute tools.

The reason schema validation tools worked automatically is different.

1️⃣ Why schema validation tools worked automatically
When you wrote something like:
model.with_structured_output(ResearchSummary)
LangChain internally creates a special tool for the model.
Conceptually it becomes:
```ini
LLM
  ↓
tool: schema_validation
  ↓
validated JSON
```
But this happens inside the model wrapper, not inside your graph.

2️⃣ Why MCP tools behave differently
Your MCP tool comes from:
`load_mcp_tools(session)`
That returns LangChain tools.
The LLM can request them:

`AIMessage(tool_calls=[...])`
But the model cannot execute Python functions.
Something must run:

`tool.ainvoke(...)`
That executor is either:
- AgentExecutor
- ToolNode
- manual code

Since you are using LangGraph, the executor should be ToolNode.


Three different tool execution patterns exist
Pattern 1 — Structured output (what you saw)
```ini
LLM
 ↓
internal schema tool
 ↓
parsed object
Used by:
with_structured_output()
```
No ToolNode required.
Pattern 2 — Classic LangChain agents
```ini
LLM
 ↓
tool_call
 ↓
AgentExecutor
 ↓
tool
 ↓
LLM
```
AgentExecutor runs the tool.
Pattern 3 — LangGraph agents (your system)
```ini
LLM
 ↓
tool_call
 ↓
ToolNode
 ↓
tool
 ↓
LLM
```
ToolNode runs the tool.
4️⃣ Why LangGraph strongly prefers ToolNode
*LangGraph is built around state machines*.
A tool call is essentially a graph transition.
Example:
```ini
research_agent
      ↓
tool_call?
   /     \
yes      no
 ↓        ↓
ToolNode   next node
 ↓
research_agent
```
Without ToolNode, you must write that logic manually.
5️⃣ Why your schema validation tool didn’t need it
Because it happened inside the model call, not the graph.
You effectively ran:
`model.invoke()`
and the wrapper handled everything.
6️⃣ When you don’t need ToolNode
If you call the tool explicitly in Python, e.g.:
`data = await search_market_data.ainvoke(...)`
Then the LLM isn’t deciding anything.
Your code is.
But that removes the agent behavior.
7️⃣ For your system the best architecture is
Planner → Scheduler → Agents → ToolNodes.
Example:
```ini
Planner
   ↓
Scheduler
   ↓
ResearchAgent
   ↓
ResearchToolNode
   ↓
ResearchAgent
   ↓
Artifact
```
This lets the research agent run multiple search iterations if needed.

| Tool Type          | Needs ToolNode? |
| ------------------ | --------------- |
| structured output  | ❌ no            |
| manual python call | ❌ no            |
| LangChain tools    | ⚠️ usually      |
| LangGraph agents   | ✅ yes           |
| MCP tools          | ✅ yes           |

-------

✅ Why Keep the Notebook Outside Microservices
Notebooks aren’t designed for long-lived async streams.
SSE / streaming HTTP in Jupyter often stalls because the event loop can get blocked or starved.
Containers and services handle concurrency better.
FastAPI / LangGraph in a container can run multiple async tasks safely, restart cleanly, and manage SSE connections without depending on a notebook.
Clean separation of concerns.
Notebook → only orchestration, triggers jobs, inspects results.
FastAPI container → handles LangGraph agent + MCP interactions.
MCP server → tools, research, external APIs.
External APIs → Tavily, OpenAI, etc.
Easier logging and monitoring.
If the notebook hangs, you can’t debug easily. In containers, you can stream logs, restart failed agents, and monitor retries.
🏗 Suggested Architecture Diagram
```ini
+-------------+      HTTP       +------------------+      SSE / MCP API      +------------------+
|  Notebook   |  ------------>  |  LangGraph API   |  -------------------->  |   MCP Server     |
| (Jupyter)   |                 |  (FastAPI +      |                         |  (FastMCP + Tools|
|   Trigger   |                 |   Agent Node)    |                         |   like Tavily)   |
+-------------+                 +------------------+                         +------------------+
         |                                                                          
         |--- GET/POST results / status ---->
```                                      
- Notebook never directly opens SSE streams.
- LangGraph API container keeps persistent MCP sessions.
- MCP server handles tools and research pipelines.
- Multiple notebooks / users can trigger jobs safely via HTTP without touching the SSE layer.

🔑 Key Practices for This Setup
- Singleton MCP session per API container.
- One persistent SSE connection per container.
- Auto-reconnect logic if the stream dies.
- Async locks for tool usage.
- Prevent multiple concurrent load_mcp_tools() calls from colliding.
- Structured output always serialized.
Pydantic → .model_dump() → JSON in Artifact.content.
Notebook only triggers jobs and polls results.
Example: POST /run-research-task
Example: GET /task-status/{task_id}
Optional: Background task queue
Use FastAPI + Celery/RQ to run heavy extraction asynchronously.
Notebook doesn’t block waiting for SSE streams.

🏎 Bonus Advantage
You can scale LangGraph containers horizontally: multiple users / notebooks can trigger independent jobs.
MCP server stays isolated; even if one LangGraph container dies, MCP and other containers remain stable.
If you want, I can sketch a working Docker Compose setup for this exact architecture, with:
1 FastAPI LangGraph container
1 MCP server container
Notebook triggers via HTTP
Optional Redis / task queue for async results
It’ll be production-ready and extremely stable, no more hanging requests.


Local Notebook
       │
       │ HTTP POST /run-research-task
       ▼
LangGraph API container (Docker Compose)
       │
       │ persistent MCP SSE session
       ▼
MCP server container (Docker Compose)
       │
       ▼
External APIs (Tavily, OpenAI, etc.)



-----
#### MCP and FastAPI connection
MCPManager Refactor — Persistent App-Lifetime Session

MCP session is persistent and never closed per-request → ClosedResourceError goes away.
Async context (AsyncExitStack) is now tied to app lifecycle, not lazy per-request.
No more async coroutines in f-strings → no subtle race conditions.
Tools are cached globally (get_tools) → avoids repeated loading.


Make MCPManager concurrency-safe (10 lines)
The key improvement is to protect the session initialization and tool loading with an asyncio.Lock. This ensures that multiple requests hitting get_session() in parallel don’t race, which is critical when you run multiple research tasks at the same time.


Where the MCP Tools Belong
Correct location:
Agent Node
   ↓
MCP Manager
   ↓
MCP Server
Your architecture already matches this:
research_agent()
    ↓
mcp_manager.get_session()
    ↓
load_mcp_tools()
This is exactly how tool-driven LangGraph agents should work.


we just need to merge MCP startup into your lifespan so the order is guaranteed and you don’t rely on @app.on_event("startup") anymore. This way, checkpointer and MCP session are fully ready before the app yields, and agents can safely call mcp_manager.get_session() without race conditions.


1. Core Services Layer
MCPManager
Single global instance
Manages a single async session to the MCP research server
Caches loaded tools for efficiency
Startup/shutdown tied to FastAPI lifespan
LangGraph Checkpointer
AsyncPostgresSaver for storing checkpoints
Connection pool lives for app lifetime
Ensures runs can be resumed after crashes
Postgres + Redis
Postgres → persistent checkpointing & metadata
Redis → ephemeral cache for parallel task states, queues, or rate-limiting
2. Agents Layer
Each agent is isolated, reentrant, and parallelizable.
Research Agent
Inputs: AgentInput task, cached tools from MCPManager
Outputs: ResearchSummary schema
Can run multiple tasks concurrently
Uses checkpointing to resume long-running searches
Vector DB Agent
Inputs: query embeddings, cached tools
Outputs: relevant documents or embeddings
Checkpoints allow resume of partial retrievals
Uses Redis for parallel query state management
Quant Sandbox Agent
Inputs: structured data, possibly ResearchSummary
Outputs: calculations, simulations, financial metrics
Uses MCP tools for data enrichment
Checkpoints allow long-running simulations to resume
Analyst Agent
Inputs: outputs from Research + VectorDB + Sandbox
Outputs: final analysis report or actionable insights
May use TrustCall or advanced schema serialization for nested objects
Parallelizable per user/task
3. FastAPI Runtime Layer
Lifespan Context
Initializes: Postgres pool + checkpointer, MCP session, Redis connection
Cleans up gracefully on shutdown
Endpoints
/run-graph → triggers agent runs, supports:
Thread memory / agent state
Resume runs via checkpoint
Parallel users
Agent checkpoints
Logging / SSE
Stream updates for real-time progress
Pprint or custom serializers to handle nested schemas
4. Parallelism & Safety
All agents share MCPManager session and cached tools but maintain per-agent tool bindings if needed.
Checkpointing + async Postgres + Redis ensures parallel execution with crash recovery.
Avoid any global mutable state inside agents—everything should be task-specific.


###### FastAPI endpoint

Step 2: Design /run-graph endpoint skeleton
We can make FastAPI support:
Thread memory → track conversation or research state per thread ID
Resume runs → pick up an existing state from memory or DB
Parallel users → isolate thread_ids
Agent checkpoints → store intermediate artifacts


API Endpoint
app/api/routes/run_graph.py
from fastapi import APIRouter
from app.services.langgraph.runtime import graph

router = APIRouter()

@router.post("/run-graph")
async def run_graph(req):

    inputs = {
        "task_id": req["task_id"]
    }

    config = {
        "configurable": {
            "thread_id": req["thread_id"]
        }
    }

    result = await graph.ainvoke(inputs, config=config)

    return result
The endpoint:
• does NOT know about agents
• does NOT know about tools
• only runs the graph
Perfect separation.


With Postgres  checkpoints added:

Feature	| Status
---| -----
Parallel users	| ✅
Thread memory| 	✅
Agent checkpoints	| ✅
Resume runs	| ✅
SSE streaming | 	✅
MCP tools	| ✅

You basically built a mini LangGraph server.

#### Redis

Optional Redis Layer
Since Redis already runs in your cluster, you can use it for tool caching.
Example use cases:
cache research results
cache expensive API calls
rate limiting
temporary agent memory
Example service:
app/services/cache/redis_client.py
But Redis is optional right now.

-------------

Parallel users
The graph runtime already supports this as long as you pass:
configurable.thread_id
You already do this 👍


Agent checkpoints
This requires a checkpointer backend:
Examples:
• SQLite
• Postgres
• Redis
Then LangGraph automatically stores:
state after each node

The single biggest improvement to your system now would be adding a LangGraph checkpointer.
It turns your server from:
Graph runner
into a full:
LangGraph runtime server


LangGraph will not automatically use Postgres or Redis.
You must explicitly attach a checkpointer to the graph.
LangGraph is intentionally explicit about persistence so you can choose:
Backend	Purpose
SQLite	local dev
Postgres	production checkpoint store
Redis	ephemeral memory / caching
For agent checkpoints + resume + thread memory, the correct backend is Postgres.
Step 1 — Install LangGraph Postgres checkpoint backend
You likely need this package:
pip install langgraph-checkpoint-postgres
It provides the class:
PostgresSaver
Step 2 — Create a Checkpointer
Create a new file.
app/services/langgraph/checkpointer.py
Example:
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://postgres:postgres@postgres:5432/langgraph"

checkpointer = PostgresSaver.from_conn_string(DB_URI)
Important:
The hostname must match your docker-compose service name.
Example compose:
services:
  postgres:
    image: postgres:16
Then your connection host is:
postgres
not localhost.
Step 3 — Attach Checkpointer to the Graph
Where you compile the graph:
graph = workflow.compile()
Change it to:
graph = workflow.compile(
    checkpointer=checkpointer
)
That's it.
LangGraph now automatically stores:
• node state
• agent outputs
• graph transitions
• thread memory
inside Postgres.
Step 4 — Use Thread IDs (you already do this)
Your endpoint already passes:
config = {
    "configurable": {
        "thread_id": "1"
    }
}
This becomes the primary key for graph memory.
Example behavior:
thread_id	Result
1	resume previous research
2	separate conversation
3	another user
Step 5 — Resume Runs Automatically
Now when you call:
await graph.ainvoke(inputs, config=config)
LangGraph will:
1️⃣ load the last checkpoint for thread_id
2️⃣ restore graph state
3️⃣ continue execution
No extra code needed.
Step 6 — Streaming Still Works
Your existing SSE endpoint:
async for event in graph.astream_events(...)
still works perfectly.
The difference is:
before → ephemeral runs
after  → persistent runs
Step 7 — Agent Checkpoints
Now every node execution is checkpointed.
Meaning if a run crashes at:
planner → researcher → analyst → quant
and crashes during analyst:
Next call with the same thread_id resumes from:
analyst
instead of starting over.
Step 8 — Parallel Users (Already solved)
LangGraph parallelism comes from:
thread_id isolation
Example:
User A → thread_id = "nvidia"
User B → thread_id = "tesla"
User C → thread_id = "apple"
All run simultaneously.
Step 9 — What Redis Is Useful For
Since you already have Redis running, you can optionally use it for:
• tool result caching
• vector search cache
• rate limiting
• agent scratch memory
But Redis is not the primary checkpoint store.
Postgres is.
What your system now becomes
Your stack is evolving into something pretty serious:
FastAPI
   ↓
LangGraph Runtime
   ↓
Agent Graph
   ↓
MCP Tool Network
   ↓
Postgres Checkpointer
   ↓
SSE Streaming
This is very close to production LangGraph architecture.
One more thing (important)
When using PostgresSaver, you should run the migration once:
checkpointer.setup()
This creates the checkpoint tables.
Next improvement I recommend
Once checkpoints are enabled, the next big improvement is adding:
Run IDs
so each execution has:
thread_id
run_id
This allows:
• replaying runs
• debugging agents
• timeline inspection
Which is extremely powerful when debugging research DAGs like yours.

| Layer       | Responsibility        |
| ----------- | --------------------- |
| API         | HTTP / SSE            |
| Runtime     | executing graphs      |
| Graphs      | workflows             |
| Nodes       | agent logic           |
| Tools       | external capabilities |
| Persistence | memory + checkpoints  |

-------

PydanticSerializationUnexpectedValue(Expected `none` - serialized value may not be as expected ...)
NotImplementedError in LogStreamCallbackHandler.on_llm_end
Your ResearchSummary object has nested structures (FinancialMetrics, lists, strings, etc.).
Pydantic’s default .dict() or serializer doesn’t know how to serialize custom objects unless you tell it.
The LLM logging callback (LogStreamCallbackHandler) tries to serialize the ResearchSummary to log or send it over SSE → it fails → NotImplementedError.
✅ Basically: the graph worked, but the callback/logging can’t fully serialize your rich Python object.

TrustCall (LangGraph’s utility for nested/complex schema serialization) exists for this reason:
Why the warning still happens
LangGraph’s LogStreamCallbackHandler attempts to .dict() or .json() everything it sees.
Even though ResearchSummary is Pydantic, it’s wrapped inside the LLM output object, which has metadata and extra fields (lc, type, id, etc.) that aren’t Pydantic.
So Pydantic raises a NotImplementedError because it doesn’t know how to serialize that container.
Basically:
LLM output → LogStreamCallbackHandler → Pydantic serialization → fails on “not implemented” nested object
What TrustCall does
TrustCall is designed to wrap LLM outputs and recursively serialize nested Pydantic schemas safely.
It handles lists, dicts, and nested objects like ResearchSummary without hitting NotImplementedError.
With TrustCall, you won’t have to change your models, and logging will work cleanly.


When containers share a bridge network:

DNS Resolution: Docker provides an internal DNS. The hostname langgraph-api automatically resolves to the internal IP of that container.

Port Isolation: You use the internal port (usually 8000) rather than the one you mapped to your host (like 8123).

In a notebook, you can "watch" your agents work in real-time. This is better than logs because you can inspect the Pydantic objects directly.


#### Senior Debugging Tip: "Time Travel"

One of the best features of the official LangGraph server is Checkpoints. In your notebook, if a run fails, you can pull the state of that exact moment:


##### 5. Troubleshooting "Connection Refused"

If your notebook says it can't find the server:

Check Network: Run !docker network inspect agent-net in a cell to ensure both your Dev Container and the LangGraph API container are listed.

Internal Port: Ensure you are using the port the server listens on inside the container (8000), not the one on your laptop (8123).

Environment: Ensure LANGCHAIN_TRACING_V2=true is set in the langgraph-api service so you can see the results in LangSmith simultaneously.

###### The "External" Network Trap

If you created the network manually via docker network create agent-net, you must mark it as external in your docker-compose.yml. Otherwise, Docker Compose will create a new network called [folder-name]_agent-net and your containers won't find each other.

Update your docker-compose.yml:
```yaml
networks:
  agent-net:
    external: true  # Tells Compose to use the existing network you created
```

```sh
docker network ls
# Then see which network your current container is on:
docker inspect <your-container-id> | grep NetworkMode
```

When you mark a network as external, you are taking responsibility for its lifecycle. Docker Compose will not create it for you; it will only try to link your services to it.

1. The Immediate Fix (Manual Creation)

Open a terminal on your host (or in your Dev Container's terminal if it has host Docker access) and run:

```Bash
docker network create agent-net
```

Once created, your docker-compose up command will find the network and proceed.

How to verify this inside your Notebook

Run these commands in a cell to see the "house" your notebook is living in:

```Python
# 1. See your own hostname (It should match your 'orchestrator' container ID)
!hostname

# 2. See what network interfaces you have
!ip addr

# 3. Check if you can resolve the other 'room' in the neighborhood
import socket
print(f"IP of LangGraph API: {socket.gethostbyname('langgraph-api')}")
```

To bridge the gap so your Notebook can talk to langgraph-api, you have two choices:


- If you don't want to change your `docker-compose.yml`, you must manually attach your running Dev Container to the network.

    - Find your Dev Container ID: `docker ps`

    - Run this in your host terminal:

            ```Bash
            docker network connect agent-net <your_dev_container_id>
            ```

    Now try the ping: Inside your notebook, `!ping langgraph-api` should work.

- The "Flush and Reset" Fix

    Run these three commands in your host terminal (not inside the notebook) to force a clean handshake:

    ```Bash
    # 1. Disconnect it forcefully
    docker network disconnect -f agent-net e8e6ee79eda6

    # 2. Reconnect it fresh
    docker network connect agent-net e8e6ee79eda6

    # 3. Restart the LangGraph API container (to refresh the DNS cache)
    docker-compose restart langgraph-api
    ```
- Verify the Connection (The "Senior" Way)

    Once you've reconnected, run this inside your Notebook to see if the network "bridge" is actually carrying traffic:

    ```Python
    import socket

    # Try to resolve the IP address of the service
    try:
        ip = socket.gethostbyname("langgraph-api")
        print(f"Success! langgraph-api is at {ip}")
    except Exception as e:
        print(f"DNS still failing: {e}")

    # If DNS fails, check if you can see ANY other hosts
    !cat /etc/hosts
    ```
Some udeful commands:

```sh
docker-compose up -d langgraph-api
# 1. Kill the specific container and its volumes
docker-compose down langgraph-api
# 2. Start it fresh (this triggers the entrypoint/install again)
docker-compose up -d langgraph-api
# 3. Watch the logs IMMEDIATELY
docker-compose logs -f langgraph-api

docker exec langgraph-api ls -la /app
```


### The Infrastructure Challenge: Persistent MCP Connections

Since you mentioned the MCP server has a persistent connection in Docker Compose, your Scheduler needs to be able to route tool calls to the right MCP endpoint based on the Agent type.


Explain this ...


##### Developing using DinD or DooD

By decoupling the Orchestrator, the MCP Server, and the Quant Sandbox into separate containers, you solve the "Dependency Hell" problem and simulate a real-world Kubernetes deployment.


Service | Responsibility | Network Port
--------| ---------- | ---------
orchestrator | "LangGraph, MasterState, Scheduler."| 8080
mcp-research,| "Web search, SEC filings, News API. (FastMCP SSE)" | 8000
quant-sandbox | "Python REPL, Jupyter Kernel, or E2B-style executor." | 8888
redis/postgres | Persistent Checkpointer for LangGraph. | 6379/5432

2. Implementation: The MCP SSE Server

Since you are using HTTP/SSE, your Research Server needs to be reachable via a URL. Using FastMCP, you can wrap it in a lightweight web server (Uvicorn).

3. The Quantitative Sandbox (The "DinD" Logic)

This is where it gets interesting. Instead of just running exec(), your agent calls a separate Sandbox Service.

Why this is "Senior" level:
If a "hallucinating" agent writes import os; os.system("rm -rf /"), it only destroys the temporary Sandbox container, not your Orchestrator or your Dev Environment.


4. How to Handle the Network in Dev Containers

Since you are in a Dev Container, you need to ensure your docker-compose.yml defines a custom bridge network so the Orchestrator can find the MCP server by its service name.

5. Why this Architecture is "Interview-Defensible"

When you present this to a hiring manager, you aren't just showing an "AI script." You are showing:

Security Obfuscation: The "Brain" doesn't have direct internet access; only the MCP service does.

Resource Isolation: Heavy data-processing tasks in the Sandbox won't starve the Orchestrator of CPU.

Horizontal Scalability: You can spin up 5 mcp-research containers to handle high-volume scraping without touching the main logic.


###### Why FastAPI
- In a complex agentic system with custom Pydantic states and microservices, FastAPI gives you total control over middleware (Auth, Logging, CORS).

- Explicit schemas make it easier to generate OpenAPI/Swagger docs for your team.

- Integration with your quant-sandbox and mcp-research is cleaner when you manage the HTTP client sessions yourself.

--------- 

#### Agent to Agent Communication:

1️⃣ What multi-agent communication is
In agentic AI, sometimes you have multiple agents that need to talk to each other, not just execute tasks individually. Examples:
- Agent A finishes task → needs Agent B to continue based on its output
- Agent C is a “critic” that checks Agent B’s reasoning and sends feedback
- Agent D is a “coordinator” that merges results from multiple agents

To do this, you need a protocol: rules and a system for agents to communicate, share data, and avoid conflicts.

2️⃣ Why it’s complex
A multi-agent protocol adds a bunch of challenges:
- Synchronization – making sure agents don’t step on each other when working on shared data
- Dependency tracking – making sure tasks happen in the right order across multiple agents
- Error handling – what if Agent B fails? How does Agent C know?
- State management – tracking what each agent knows, what’s finished, and what’s pending
Basically, it’s like turning your planner + scheduler + single agent system into a mini distributed system.

3️⃣ When it matters
- In your current project, you have one agent per task, scheduler orchestrates the DAG, which is already a strong architecture. You’re already showing exactly the right level of reasoning. Adding it would explode complexity with little extra resume value.
- Multi-agent communication protocols are mostly needed for systems where multiple agents collaborate or compete on the same task simultaneously.

##### Alternative ways

Using a global state (with reducers or similar) to let agents communicate implicitly is a common pattern in multi-agent systems. Let me break it down carefully and also show where it could evolve.
1️⃣ What you have: global state + reducers
- Global state = a shared data structure that all agents can read/write.
- Reducers = functions that take the current state and an agent’s output, then produce the new state.

Effectively:
Agent output → Reducer → Update global state → Next agent sees updated state

✅ Pros of this:
- Very simple to implement
- Keeps agents loosely coupled (they don’t call each other directly)
- Scheduler can observe all state changes, useful for DAG orchestration
- Makes debugging easier than ad-hoc messaging

2️⃣ Other common communication patterns
Depending on the system complexity, there are other ways agents communicate:

- a) Direct messaging / RPC between agents
Agent A sends structured messages to Agent B directly
    - Useful if agents need real-time collaboration or peer negotiation
    - Slightly more complex: you need message queues, error handling, timeouts
- b) Event-driven pub/sub
    - Global “event bus” where agents publish messages (events)
    - Other agents subscribe to relevant events
    - Good for highly asynchronous or distributed systems

- c) Shared database / vector memory
    - Agents write outputs to a shared DB
    - Other agents read from DB when needed
    - Similar to global state but can scale beyond single memory
    - Useful when you later add persistent memory

3️⃣ How to think about it for your project
Right now, your global state + reducers pattern is perfect:
- Simple, understandable
- Shows modularity and system design
- Works well for task DAGs
- Easy to extend if you later want persistent memory or multiple agents

You don’t need to add anything else unless you want:
- Real-time agent-to-agent negotiation
- High parallelism with distributed nodes
- Persistent memory across sessions

For a portfolio project, your current architecture is already impressive.

4️⃣ Optional tweaks that look strong in interviews
- State versioning / history — keep snapshots of state after each task
- State validators — ensure reducers produce consistent, schema-compliant outputs
- Logging — show exactly what each agent did and how it updated state

Even small things like that show robust system thinking.


---------------------------------
## About LangGraph

### States:

For a senior-level, production-ready system in 2026, the industry standard is Pydantic for the internal Graph State. Use Pydantic models for each section to enforce semantic guardrails.

While TypedDict is lightweight, it only provides type-hinting. Pydantic provides runtime validation. In a multi-agent system, if the Researcher agent outputs a malformed string instead of a list of URLs, a TypedDict will let it pass (causing a crash later), whereas Pydantic will catch the error at the node boundary, allowing your Reflection logic to trigger a retry.

1. The "State of the Art" State Strategy

We will use a Hierarchical State pattern.

- MasterState: The global state shared across the main graph.

- SubgraphStates: Local states for specific logic (like the Evaluation loop) that "sync" back to the MasterState.

2. Why this is "Production Ready"

- Runtime Safety: If the Quantitative agent tries to inject a float into the plan (which expects a List[str]), Pydantic will raise a ValidationError immediately.

- The Reducer Pattern (Annotated): Notice operator.add or merge_lists. This is crucial for Parallel Research. If your Web Search and Vector Search finish at the same time, LangGraph uses this function to combine their results into the MasterState without data loss.

- Traceability: Because these are Pydantic objects, they serialize perfectly into JSON for LangSmith or a SQL Database, making your "Human-in-the-loop" resume/pause functionality seamless.

In LangGraph, the state object passed into each node is an instance of the schema you defined (often a TypedDict). For example, if your schema is inherited from `StateMessage`, you can access them with dict methods (`state["messages"]` works) not `.` method for Pydantic schemas (`state.messages`: Error).


### Send command
The error occurs because when you use `Send`(node_name, payload), LangGraph passes that entire payload dictionary as the first positional argument to your node function.
Your analyst function is likely defined to expect a task_id string, but LangGraph is trying to hand it the whole state dictionary (the payload).

You need to update your agent nodes definition to accept the state (the payload you sent) and then extract the task_id from it.

1. The Send Logic: When you do Send("analyst", {"task_id": "123"}), LangGraph treats {"task_id": "123"} as the local state for that specific branched invocation of the analyst node.
2. The Function Signature: LangGraph always calls a node by passing *the state* as the first argument. If your function header is def analyst(task_id), Python thinks the dictionary {"task_id": "123"} is the value for `task_id`. It then looks for a second argument that doesn't exist, leading to the "missing 1 required positional argument" confusion.

To recieve `task_id` which is not a key in global state, we need to make it clear for recievers, foe example by doing:

```python
class AgentInput(MasterState):
    task_id: int

async def analyst(state: AgentInput):
    ...
```

but note these 2 things:
1. The Send call must include everything

When you use Send, LangGraph does not automatically copy the global state into the new branch. You have to manually "spread" the current state into the dictionary you are sending:

```python
return [
    Send(task.agent, {
        **state,           # This passes all MasterState fields (messages, topic, etc.)
        "task_id": task.id # This adds the specific task_id for the analyst
    }) 
    for task in state.plan if task.status == "running"
]
```

### Updating Tasks by Reducers
By default, LangGraph overwrites a state key with whatever a node returns. If your analyst node returns an update but doesn't include the full plan, or if multiple analysts return updates simultaneously, they might overwrite each other's data, leaving the list empty or incomplete. 

In LangGraph, the **Reducer** is called every time a node returns data for that key. When  `planning_architect` finishes, it returns a list of Task objects, but your reducer is expecting a list of `TaskUpdate` objects to merge.

LangGraph often passes state updates as dictionaries rather than Pydantic objects during the reduction phase. If you see errors about `.id` not existing, change `up.id` to up["id"] or ensure you are casting them back to objects inside the reducer.

### Send
LangGraph expects nodes to return a dictionary that maps to your state keys. In LangGraph, even if you are only performing a "fan-out" (branching), the node must return a dictionary where the **Send** objects are the value for a specific key (usually the one that triggers the next step) OR you simply return the list of Send objects as the entire return value if that node is specifically a **Command** or a functional router.

Wrap it in a Dictionary If your scheduler is a standard node in the graph, it must return the Send objects inside a dictionary.

```python
return {
    "plan": [Send(task.agent, {**state, "task_id": task.id}) for task in tasks_to_run]
}
```

In LangGraph, Send objects can only be returned from a Conditional Edge function, not from a Standard Node.

```python
return [Send(task.agent, {**state, "task_id": task.id}) for task in tasks_to_run]
```
 
Why it is better to use conditional edge for fanning out:
- Mermaid Visualization: Because you explicitly mapped the names (e.g., "analyst": "analyst"), the graph will show a clear fan-out from the scheduler to all agents.
- Decoupling: Your scheduler node focuses on data, while your edge focuses on flow.
- No "Ghost" Invocations: Since there are no direct edges, your agents will only ever be called when the router explicitly sends them a task_id.

Why this works:
- The Fan-Out: The scheduler (via route_to_agents) sends tasks to multiple agents at once.
- The Collection: As each agent finishes, it returns its `TaskUpdate`.
- The Loop: The edge takes the agent back to the scheduler. The scheduler looks at the updated plan, sees which tasks are now "ready" (because their dependencies met), and fans out again.
- The Exit: When `route_to_agents` sees no more tasks to run, it returns END.



--------------------------------




































































-----------

##### Mental Help

Yas… I hear you, and that’s a very real and valid fear. What you’re describing is not just about technical ability, it’s about credibility in a hyper-competitive, high-stakes environment — and it’s a new kind of pressure that didn’t exist even a few years ago.
Let me unpack a few things here:

1️⃣ The “AI / ChatGPT effect”
- Many candidates can now generate code, diagrams, project ideas, or reports quickly using AI.
- That raises skepticism in interviewers: “Did you really do this yourself?”
- Even when you actually built it, you can feel like you’re under constant suspicion.
- That’s a market-level pressure, not a personal failure. The rules changed, and everyone is adjusting — especially people like you who genuinely invest time learning and building projects.

2️⃣ Interviewers as “grillers”
You described interviews where:
- Minor misrepresentations are seen as intentional lies
- New technologies are expected instantly
- Any small mistake feels magnified

That’s actually a systemic problem in technical interviewing, not a reflection of your worth or abilities. Many skilled engineers fail interviews simply because of interviewer style or unrealistic expectations, not lack of competence.

3️⃣ Why this feels worse for you
You already carry regret, career gaps, and age-related concerns
You’re also investing serious energy learning cutting-edge AI systems
So the moment you walk into an interview, your brain amplifies the stakes and judgment
It’s natural that it feels heavier than it used to.

4️⃣ What actually works against this skepticism
Even in today’s environment, there are ways to signal credibility:
- Concrete project demos
- Don’t just claim “I did X”; show a working system, even minimal.
- Screenshots, recorded video of tasks, notebooks, or live demo.
- Documented process
- Include design notes, architecture diagrams, Git history.
- Show how the project evolved and decisions you made.
- Explain your reasoning, not just outputs

Interviewers often grill people to test thinking.
- Walk through why you chose FastMCP, global state reducers, planner/DAG, etc.
- Set expectations upfront in interviews
Example: “I built a minimal prototype for the agentic research system, focusing on modular MCP servers and task orchestration. Here’s a demo and code snippet.”

This frames you as honest and grounded, reducing the “skepticism pressure.”

Yas, the pressure you feel is understandable and real, but it’s mostly external noise, not a reflection of your actual skills. You’re thinking at a level most entry-level engineers never do — planners, DAGs, schedulers, modular agents. That’s already strong, even if the interview world is skeptical.

Apply broadly — not only ML titles
This is important.
You should apply to roles like:
- ML engineer
- AI engineer
- backend engineer
- data engineer
- platform engineer

Your architecture work overlaps with backend systems and distributed workflows, which many companies need.


6. Avoid the trap you described

You said something very insightful earlier:
interview → learn new thing → interview → learn another thing
- Try to stop chasing every new trend.
Instead:
- Focus on one strong system and learn it deeply.
- Your current project is perfect for that.

7. Use your maturity as an advantage
At 46 you have something younger candidates often lack:
persistence
- depth
- ability to understand complex systems
- Many companies actually value that when they see it.



Resume entry
Something like:
Agentic AI Investment Research System
- Designed multi-agent architecture using LangGraph
- Planner generates DAG of tasks with dependencies
- Scheduler dynamically routes tasks to agents
- Implemented research agent using web search tools
- Built quant sandbox for financial analysis
- Integrated external tools via MCP servers
- Generated automated structured research reports

This sounds very strong.



The 5-minute story you should be able to tell
Something like:
“I built an agentic AI system that automatically generates an investment research report for a company.
The system uses a planner that decomposes a user query into a DAG of tasks.
A scheduler then dynamically routes those tasks to specialized agents.
The research agent gathers information using web search tools, while a quant sandbox performs financial calculations.
All tools are exposed through MCP servers, allowing the agents to call external capabilities in a structured way.
The system finally synthesizes the results into a structured investment research summary.”
That alone already sounds very strong.



Clear README

Explain the system clearly:
- Problem: automated investment research
- Architecture: planner → agents → sandbox → auditor → analyst
- Technologies used (LangGraph, MCP, Docker, etc.)
- Example output (report + Monte Carlo simulation)

2️⃣ One strong resume bullet (maybe two)
Focus on impact and architecture, not just tools.
Example idea:
Built an agentic AI research system using Python and LangGraph that plans tasks via a DAG, orchestrates research and quant agents, executes generated Python code in an MCP sandbox, and validates results with an automated auditor before producing a final investment analysis report.
That kind of sentence shows systems thinking, which hiring managers like.

3️⃣ Then return to applying.

Exactly as you said: you don’t want to get stuck in endless projects. This one is enough to demonstrate the capability.





A small practical tip for your notes
Don’t try to prepare everything. Just prepare answers to a few key questions like:
1️⃣ What problem does your system solve?
2️⃣ What is the architecture?
3️⃣ What was the hardest technical challenge?
4️⃣ How did you handle hallucinations or incorrect outputs?
5️⃣ What would you improve next if you had more time?
If you can answer those calmly, you are already in a very strong position.