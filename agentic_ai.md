Phase 1 (Week 1–2): Mental Model Solidification
Goal: Vocabulary + clarity + architecture confidence.

You will master:
- ReAct pattern (reason + act loop)
- Tool calling mechanics (JSON schemas, function calling)
- Memory types:
    - Short-term state
    - Long-term vector memory
- Planning patterns:
    - Single-agent loop
Planner + executor split
Guardrails:
Input validation
Output schema enforcement
Tool permissioning

- Evaluation:
    - Task success metrics
    - Failure logging
    - Human-in-the-loop feedback
- Deliverable:
    - You can whiteboard an agent system confidently.
    - No code obsession yet.

Phase 2 (Week 3–4): Build ONE Structured Agent
Not toy. Not messy.
One clean architecture.
Example (aligned with your profile):
Option A: Enterprise Document Intelligence Agent
Upload financial reports
Retrieve context (RAG)
Decompose query into subtasks
Use tools:
Calculator
SQL
Document retriever
Produce structured report
Log intermediate reasoning (without exposing chain-of-thought)
Store evaluation metrics
Option B: Multi-step Research Agent
Query planning
Web retrieval simulation
Source ranking
Structured synthesis
Confidence scoring
Key: clean architecture.
Include:
Tool abstraction layer
Logging
Observability hooks
Failure recovery
This becomes your demo story.

Phase 3 (Week 5–6): Production Framing
You add:
Latency analysis
Cost analysis
Failure cases
Guardrail discussion
Security concerns
Scaling considerations
Evaluation framework
Now you’re no longer “I built an agent.”
You are:
“I designed a reliable multi-step LLM system for enterprise.”
That’s interview gold.


Why This Is High ROI For YOU Specifically
Because you are not:
A prompt engineer
A researcher chasing papers
A frontend tinkerer
You are:
Systems thinker
ML depth
Pipeline-oriented
Deployment-aware
Agentic AI without system depth is fragile.
Agentic AI with system design is rare.
That’s leverage.



They’re looking for someone who can answer:
How do you structure multi-step reasoning?
How do you prevent tool hallucination?
How do you evaluate task success?
How do you control cost?
How do you log failures?
How do you scale this safely?
If you can confidently design and discuss those, you are ready.



What “Agentic-Ready AI Engineer” Actually Means in 2026
It means you can confidently do these five things:
1️⃣ Design a Multi-Step LLM System
You can explain and implement:
Goal decomposition
ReAct-style loops
Tool invocation
State tracking
Error handling
You can whiteboard it under pressure.
That alone puts you ahead of most candidates.
2️⃣ Integrate Tools Safely
You understand:
JSON schema enforcement
Deterministic tool interfaces
Tool permission boundaries
Output validation
Retry logic
This is engineering maturity — not hype.
3️⃣ Add Memory Properly
You know the difference between:
Context window memory
Structured state
Vector retrieval memory
When NOT to use memory
This is critical. Most people misuse it.
4️⃣ Implement Evaluation
You can define:
Task success metrics
Regression tests
Logging
Failure categorization
Cost vs performance trade-offs
This is where senior-level credibility comes from.
5️⃣ Frame It as Production System Design
You can discuss:
Latency bottlenecks
Token cost control
Observability
Scaling
Security concerns
Guardrails
At that point, you’re not “learning agents.”
You’re operating as an AI systems engineer.





<!-- 


## 🔥 Project Concept (High-ROI, Resume-Level)
##### Enterprise AI Research & Decision Support Agent
Think:

“An autonomous multi-step AI system that analyzes enterprise documents, retrieves knowledge, invokes tools, performs calculations, and produces structured decision-ready reports — with evaluation and guardrails.”

This is not a chatbot.
This is a system.

#### 🧱 High-Level Architecture
##### 🏗 Project Headlines (Bounded but Comprehensive)
We’ll refine later. For now, structure only.
###### 1️⃣ Ingestion Layer
- Upload PDFs / financial reports / policy documents
- Chunking strategy
- Embedding pipeline
- Vector DB (FAISS locally or Pinecone/Weaviate in cloud)
- Metadata indexing

Concepts covered:
- Chunking strategies
- Embedding models
- Indexing trade-offs
- Recall vs precision

###### 2️⃣ Retrieval-Augmented Generation (RAG Core)
- Hybrid retrieval (semantic + keyword optional)
- Query rewriting
- Context filtering
- Source ranking
- Confidence scoring

Concepts covered:
- Dense vs sparse retrieval
- Context window budgeting
- Hallucination mitigation
- Citation enforcement

###### 3️⃣ Agent Orchestration Layer (The Agentic Core)
Single-agent with structured planner OR planner + executor split.

Capabilities:
- Task decomposition
- Tool selection
- Multi-step reasoning loop (ReAct style)
- Controlled iteration limit
- Tool failure recovery

Tools example:
- Calculator
- SQL-like structured query
- Document retriever
- Risk scoring function

Concepts covered:
- Tool schema design
- Deterministic interfaces
- Retry logic
- Loop termination strategies

Framework choice (pick one only):
- LangGraph (recommended for structure)
OR
- Clean custom orchestrator (stronger engineering signal)

###### 4️⃣ Memory & State Management
- Short-term state object
- Structured conversation memory
- Long-term vector memory
- When memory is NOT used

Concepts covered:
- Stateless vs stateful agents
- Memory bloat problems
- Token cost control

###### 5️⃣ Guardrails & Safety
- Input validation
- Output schema enforcement (Pydantic)
- Tool permission boundaries
- Prompt injection mitigation strategy
- Role-based tool access (optional advanced)

Concepts covered:
- Security
- Safety design
- Attack surfaces in LLM systems

###### 6️⃣ Evaluation & Observability (Critical)
This is what makes it senior-level.
Add:
- Task success metric definition
- Automated test queries
- Regression evaluation set
- Logging of:
  - tool calls
  - failures
  - latency
  - token usage
- Cost tracking

Optional:
- Basic dashboard (Streamlit / FastAPI + logging view)
Concepts covered:
- Offline evaluation
- Online monitoring
- Performance drift
- Reliability engineering

###### 7️⃣ Deployment Layer
Two options:
-  Option A: Local Production Simulation
   - Dockerized service
   - FastAPI endpoint
   - Local vector DB
   - Structured logging

- Option B: Cloud
  - AWS/GCP
  - Managed vector DB
  - API Gateway
  - Containerized backend

Concepts covered:
- Scalability
- Infra decisions
- Rate limiting
- Cost estimation

##### 🎯 What This Project Signals in Interviews
You can confidently say:
- “I designed a multi-step agentic system.”
- “I implemented structured tool orchestration.”
- “I added evaluation loops and regression testing.”
- “I analyzed cost-performance tradeoffs.”
- “I implemented guardrails and failure recovery.”
- “I deployed it containerized.”

That’s AI Systems Engineer level.
Not tutorial level.

📚 About the IBM Certificate
Use it selectively.
Skip or skim:
Basic GenAI intro
Basic RAG intro
Focus on:
Fundamentals of Building AI Agents
Agentic AI with LangChain / LangGraph
MCP concepts (important vocabulary)
Multi-agent orchestration overview
Treat it as vocabulary + terminology reinforcement.
Not your main project.
Your project is the centerpiece.

###### 📅 Time Scope (Realistic)
6–8 weeks total:
Weeks 1–2 → RAG + ingestion clean
Weeks 3–4 → Agent orchestration
Weeks 5–6 → Evaluation + guardrails
Weeks 7–8 → Deployment + polish
Bounded.
No scope creep.
 -->



----------------------
----------------
----------

### Foundational Models
- tempreture: higher -> more creative, lower -> more deterministic
- max_tokens: limits the number of tokens in the response
- timeout: max time to wait for response for the model before canceling the request
- max retires: max amount of times to retiry your request if that request fails

Now you create an agent with a custom model but its not very useful unless you taylair to to your specific usecase.

The easiest way to customize the performance of a chat model is with system prompt. For example, when you ask a LLM the capital of the moon, it correctly tells you there isnt one.  

```python
from langchain.chat_models import init_chat_model

model = init_chat_model(model="gpt-4.1-nano")

response = model.invoke("What's the capital of the Moon?")
response
```
```o
AIMessage(content='The Moon does not have a capital, as it is a natural satellite and not an independent nation or political entity.', additional_kwargs={'refusal': None}, ....
```
or 

```python
response.content
```

```o
'The Moon does not have a capital, as it is a natural satellite and not an independent nation or political entity.'
```

#### Initializing an agent

```python
from langchain.agents import create_agent
from pprint import pprint

agent = create_agent("gpt-4.1-nano")

from langchain.messages import HumanMessage

response = agent.invoke(
    {"messages": [HumanMessage(content="What's the capital of the Moon?")]}
)

pprint(response)
```
```o
{'messages': [HumanMessage(content="What's the capital of the Moon?", additional_kwargs={}, response_metadata={}, id='a716c09b-5361-4f45-9c59-25258d7a3a5d'),
              AIMessage(content='The Moon does not have a capital, as it is a celestial body and not a country or city.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 21, 'prompt_tokens': 14, 'total_tokens': 35, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'gpt-4.1-nano-2025-04-14', 'system_fingerprint': 'fp_62b483d6f3', 'id': 'chatcmpl-DMzdR0Xby2J8jXEGttKprnaXUBZWI', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019d20d7-dde0-7613-8e69-817f590ba239-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 14, 'output_tokens': 21, 'total_tokens': 35, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})]}
```
with all the metadata etc. The last message is
```python
response['messages'][-1].content
```
```o
The Moon does not have a capital, as it is a celestial body and not a country or city.
```
How about this? Here, we are passing a chat history:
```python
from langchain.messages import AIMessage

response = agent.invoke(
    {"messages": [HumanMessage(content="What's the capital of the Moon?"),
    AIMessage(content="The capital of the Moon is Luna City."),
    HumanMessage(content="Interesting, tell me more about Luna City")]}
)

```
which returns a dict of messages including the final message (AIMessage):

```o
"Luna City is a fictional or conceptual settlement often referenced in science fiction and space exploration discussions as a proposed or imagined permanent human settlement on the Moon. It is envisioned as a hub for scientific research, resource extraction, and potentially a stepping stone for missions deeper into space, such as Mars.\n\nWhile Luna City doesn't currently exist in reality, the concept generally includes features like:\n- **Habitat modules** to support life in the Moon's harsh environment.\n- **Research facilities** for lunar geology, astronomy, and other scientific endeavors.\n- **Resource utilization centers** for mining lunar materials like water ice and regolith-based minerals.\n- **Transportation infrastructure** for moving between different parts of the Moon and potentially to Earth.\n\nIn speculative terms, Luna City represents humanity's vision of establishing a sustainable presence on the Moon, fostering advancements in technology and enabling future exploration missions. Various space agencies and private companies, such as NASA, ESA, and SpaceX, are exploring concepts and plans that could make such settlements a reality in the coming decades."
```

#### Streaming Output

One of the issues with agents is latency. Software systems may measure the response time in milliseconds, we'll be running agents with response times of seconds even minutes if we hand in few messages. One way ti improve percieved latency is to use `stream` to invoke agents which streams tokens to users rather than printing answers all at once. 

```python
for token, metadata in agent.stream(
    {"messages": [HumanMessage(content="Tell me all about Luna City, the capital of the Moon")]},
    stream_mode="messages"
):

    # token is a message chunk with token content
    # metadata contains which node produced the token
    
    if token.content:  # Check if there's actual content
        print(token.content, end="", flush=True)  # Print token
```

This method is used by all major chatbots to improve preceived latecny and user experience. 

### Prompt 

Now you have a custome model. The easiest way to imporve the perfromance of the model to taylor it for your specific usecase is **system prompt**.

```python
from langchain.agents import create_agent
from langchain.messages import HumanMessage

agent = create_agent(model="gpt-4.1-nano")

question = HumanMessage(content="What's the capital of the moon?")

response = agent.invoke(
    {"messages": [question]}
)

response['messages'][1].content
```
```o
The Moon does not have a capital, as it is a natural satellite of Earth and does not have a government or administrative divisions.
```

But if you include a system prompt, the answer will be different:

```python
from langchain.messages import SystemMessage

system_prompt = "You are a science fiction writer, create a capital city at the users request."

response = agent.invoke(
    {"messages": [SystemMessage(content=system_prompt)] + [question]}
)
```

```o
'The moon doesn\'t have an official capital, but if we imagine a futuristic lunar colony, a fitting "capital" could be **Lunos Prime** — a sprawling lunar city located in the crater of Shackleton, near the lunar south pole, serving as the central hub for governance, research, and resource management in this envisioned moon colony. Would you like me to elaborate on how Lunos Prime might look and function?'
```

#### Few-shot examples

We usually prefer the LLM response to be concise and even structured rather than long or unstructured. We can singal this to LLMs using a system prompt:

```python
system_prompt = """You are a science fiction writer, create a space capital city at the users request.
User: What is the capital of mars?
Scifi Writer: Marsialis

User: What is the capital of Venus?
Scifi Writer: Venusovia"""

response = agent.invoke(
    {"messages": [SystemMessage(content=system_prompt)] + [question]}
)

response['messages'][-1].content
```
```o
'Lunaris Prime'
```

#### Structured Prompts
We often desire agent response to be structured. One way to do this to add system prompt showing the model how to structure its response:

```python
system_prompt = '''
You are a science fiction writer, create a space capital city at the users request.

Please keep to the below structure.

Name: The name of the capital city
Location: Where is it based
Vibe: 2-3 weeks to describe its value
Economy: Main industries
'''
```

And now pose the same question. You will receive the naswer with just these 4 topics. 

```o
Name: Luaris Prime
Location: South plar region, perched on the rim of Shackleton Crater, Moon
Vibe: Icebound metropolis
Economy: ISRU ice mining and processinc
```

A good prompt helps LLMs to focus on their tasks and return a more high quality answers. It is more common to do theis using tools.

### Tools

What separates an agent from a standard chatbot is its ability to take actions and reat accordingly. ReAct agnets use this pattern since most of the industry coalesed around this matter, we'll just call ReAct agents, agents. The actions that an aent can take are defined by the tools that we provide to it. Tools can allow agents to access data, execute tasks, even call our agents, transforming it from a passive language model to the coordinator of a much more capable system. 

You can turn any function into a tool by adding `@tool` decorator, adding detailed description for the function which becomes the tool description. You can then use `.invoke(*arg, **kwarg)` to run the function as usual. This is exactly how agents run the tool. We usually specify a list of the tools for our LLM to use when creating agents. The agents understands to use the tools provided when appropirate:

```python
from langchain.tools import tool

@tool
def square_root(x: float) -> float:
    """Calculate the square root of a number"""
    return x ** 0.5

square_root.invoke({"x": 467})
```
```o
21.61018278497431
```
```python
from langchain.agents import create_agent
from langchain.messages import HumanMessage


agent = create_agent(
    model="gpt-4.1-nano",
    tools=[square_root],
)

question = HumanMessage(content="What is the square root of 467?")

response = agent.invoke(
    {"messages": [question]}
)

response['messages'][-1].content
```
The last message is very neat and exactl as we expected:
```o
The square root of 467 is approximately 21.61.
```
The model used the tool `square_root` automatically without even us adding any system prompt to possibly hint the model to use the tools. 
```python
response['messages']
```
```o
[HumanMessage(content='What is the square root of 467?', additional_kwargs={}, response_metadata={}, id='04faf488-d62d-4f63-810c-2e53fe1a3ed4'),
 AIMessage(content='', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 54, 'total_tokens': 68, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'gpt-4.1-nano-2025-04-14', 'system_fingerprint': 'fp_4ea5d69903', 'id': 'chatcmpl-DN0lTGWn8nRqv74IBXa6kYVBWWp5G', 'service_tier': 'default', 'finish_reason': 'tool_calls', 'logprobs': None}, id='lc_run--019d211a-1a98-7921-9de1-232ecabdf54b-0', tool_calls=[{'name': 'square_root', 'args': {'x': 467}, 'id': 'call_O63szKKflf2cqQRaNO38iyB9', 'type': 'tool_call'}], invalid_tool_calls=[], usage_metadata={'input_tokens': 54, 'output_tokens': 14, 'total_tokens': 68, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}),
 ToolMessage(content='21.61018278497431', name='square_root', id='d6ae2d6f-32da-4cc1-a860-6f87b8b03d78', tool_call_id='call_O63szKKflf2cqQRaNO38iyB9'),
 AIMessage(content='The square root of 467 is approximately 21.61.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 83, 'total_tokens': 97, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'gpt-4.1-nano-2025-04-14', 'system_fingerprint': 'fp_4ea5d69903', 'id': 'chatcmpl-DN0lUNtYCJT1ygeevP0YdIOhGNgyg', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019d211a-2493-7291-b7ac-9634bdb22144-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 83, 'output_tokens': 14, 'total_tokens': 97, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})]
```

You can see the model knows to use the tools. It creates an `AIMessage` with *no content* but containng a **tool call** part provifing the arguments the tool needs to run. The response is back to the model from the tool call message is called `ToolMessage`, which returns the result of applying the tools: `content='21.61018278497431'`.  Finally, the model polishes the final answer to the user request.

### Search Web
There are tools to add even more complex capabilities to LLMs such as searching the web. LLMs cant do that on their own.

```python
from langchain.messages import HumanMessage

question = HumanMessage(content="Who is the current mayor of San Francisco?")

response = agent.invoke(
    {"messages": [question]}
)

response['messages'][-1].content
```

```o
'As of October 2023, the current mayor of San Francisco is London Breed.'
```

which is incorrect! The current mayor at Mar 2026 is Daniel Lurie! Why the agent got is wrong??
```python
response = agent.invoke(
    {"messages": ["How up-to-date your training knowledge is?"]}
)
response['messages'][-1].content
```

```o
'My training includes information up until October 2023. If you have questions about events or developments beyond that date, I may not have the most current details.'
```
Model training knowldege is not up-to-date. `TavilySearch` API can help here.

```python
from langchain.tools import tool
from typing import Dict, Any
from tavily import TavilyClient

tavily_client = TavilyClient()

@tool
def web_search(query: str) -> Dict[str, Any]:

    """Search the web for information"""

    return tavily_client.search(query)

web_search.invoke("Who is the current mayor of San Francisco?")
```
The result is a list of search results from web related to the query.  Now let add this as a tool to the agent:

```python
agent = create_agent(
    model="gpt-4.1-nano",
    tools=[web_search]
)

question = HumanMessage(content="Who is the current mayor of San Francisco?")

response = agent.invoke(
    {"messages": [question]}

response["messages"][-1].content
```
```o
'The current mayor of San Francisco is Daniel Lurie.'
```
Now its correct. If we look at the detailed response we see that the model makes a tool call which returns the following  `ToolMessage`:

```o
ToolMessage(content='{"query": "current mayor of San Francisco", "follow_up_questions": null, "answer": null, "images": [], "results": [{"url": "https://en.wikipedia.org/wiki/Mayor_of_San_Francisco", "title": "Mayor of San Francisco - Wikipedia", "content": "The current mayor is Democrat Daniel Lurie.", "score": 0.94251215, "raw_content": null}, {"url": "https://apnews.com/article/san-francisco-new-mayor-liberal-city-81ea0a7b37af6cbb68aea7ef5cc6a4f0", "title": "San Francisco\'s new mayor is starting to unite the fractured city", "content": "San Francisco Mayor Daniel Lurie, a political newcomer and Levi Strauss heir, has marked his first 100 days with a hands-on, business-friendly approach.", "score": 0.8745175, "raw_content": null}, {"url": "https://www.sf.gov/departments--office-mayor", "title": "Office of the Mayor - SF.gov", "content": "Daniel Lurie is the 46th Mayor of the City and County of San Francisco.", "score": 0.8446273, "raw_content": null}, {"url": "https://en.wikipedia.org/wiki/Daniel_Lurie", "title": "Daniel Lurie - Wikipedia", "content": "Daniel Lawrence Lurie (born February 4, 1977) is an American politician and philanthropist who is the 46th and incumbent mayor of San Francisco, serving since", "score": 0.8156003, "raw_content": null}, {"url": "https://www.sf.gov/profile--daniel-lurie", "title": "Daniel Lurie - SF.gov", "content": "Chair, and Mayor of San Francisco. Disaster Council · Office of the Mayor. Mayor Daniel Lurie sworn in as the City\'s 46th mayor on Jan 8. See recent news. Learn", "score": 0.81524754,  --- continued
```

Thsi tool call enables the model to answer correctly.

### LangSmith
To get insight into how your agents are running, use LangSmith to trace all queries and observe latencies, token usage, tools called and their inputs and outputs. Connect to the API endpoint (with its API key) to debug your agents when things get a little bit more complex as you add more tools and the tasks are less deterministic. With the free tier, you have up to 5000 free tokens per month which is more than enough for development and side projects. 


### Meomory

The very basic feature expected from any chatbot is the ability to maintain the memory over the length of the conversation. The angent built so far dont have that ability. 

```python
from langchain.agents import create_agent
from langchain.messages import HumanMessage

agent = create_agent(
    "gpt-4.1-nano"
)

question = HumanMessage(content="Hello my name is Seán and my favourite colour is green")

response = agent.invoke(
    {"messages": [question]} 
)

response["message"][-1].content
```
```o
"Hello Seán! It's great to meet you. Green is a wonderful colour—so fresh and calming. Do you have a favourite thing that's green?"
```
Then ask:

```python
question = HumanMessage(content="What's my favourite colour?")

response = agent.invoke(
    {"messages": [question]} 
)

response["message"][-1].content
```
```o
"I don't have access to your personal information, so I don't know your favorite color. If you'd like to tell me, I'd be happy to chat about it!"
```

What's happening? In out LanChain aganets we are tracking *states* which you can think of it as the memory of our agent. The problem is that *the state is NOT being saved from one run to another run*. In fact the agent memory is being wiped! We need to save the states so agents remember previous messages. We do that by using 
- **checkpointers**: saves a snapshop of the state at the end of each run, and then groups them into the same **thread** of conversation, same thread id. 

`InMemorySaver` is the checkpointer we use in LangGraph:

```python
from langgraph.checkpoint.memory import InMemorySaver 
from langchain.messages import HumanMessage

agent = create_agent(
    "gpt-4.1-nano",
    checkpointer=InMemorySaver(),  
)

question = HumanMessage(content="Hello my name is Seán and my favourite colour is green")
config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [question]},
    config,  
)

response["message"][-1].content
```
```o
"Hello Seán! It's great to meet you. Green is a wonderful colour—calming and full of life. Do you have a favourite thing that’s green or a reason why you like it?"
```

```python
question = HumanMessage(content="What's my favourite colour?")

response = agent.invoke(
    {"messages": [question]},
    config,  
)

response["message"][-1].content
```
```o
'Your favourite colour is green.'
```

In fact you can see 4 messages in the response including the previous ones. It retained the memory of our conversation and appended it to its list of messages and they are all included in this repsonse because all the checkpoints are grouped by `thread_id=1`. Now we have memory.

##### Customized States
By default, the states track a list of messages only. But we can add custom fields like `user_id`, `langage` if we would like them to be tracked overtime. These fields dont even have to be text! Text is not the only type of inputs LLMs can receive these days. The states could include images or audios etc so agents can see or hear! Encode you image and audios in `Base64`. Thsi enables us to represent binary data and transmit text-based communication channels. 

In the following snippet, we have encoded a picture of moon with a urban scene and invoking the model with a query about this picture. The image `img_b64` is encoded as `Base64` before.  

```python
from langchain.messages import HumanMessage
from langchain.agents import create_agent

agent = create_agent(
    model='gpt-4.1-nano',
    system_prompt="You are a science fiction writer, create a capital city at the users request.",
)

multimodal_question = HumanMessage(content=[
    {"type": "text", "text": "Tell me about this capital"},
    {"type": "image", "base64": img_b64, "mime_type": "image/png"}
])

response = agent.invoke(
    {"messages": [multimodal_question]}
)

response['messages'][-1].content
```
```o
This image depicts a breathtaking extraterrestrial city set against a dramatic alien landscape. The towering spires and sleek structures suggest an advanced civilization, possibly centered around energy or technological innovation. The city appears to be built in harmony with the rugged terrain—its spires piercing the sky and blending seamlessly into the rocky environment.

In the background, a massive moon or planet dominates the sky, hinting at a neighboring ... --- continued
```

You can do the same thing for Audio files. Convert them to `Base64` and pass them to LLMs with and a query.

### Model Context Protocol (MCP)
MCP is defined by Anthropic the group who made it as an open protocol that standardizes how your LLM applications connect to and work  with your tools and data sources. It allows us to plug in various tools to our agent to add functionality just like USBs.

Creating tools and providing context to different model providers used to look a lot like this. A never ending web of API calls and databases to connect to your agents for every application you try to build. Thats why this universal model context protocol for model providers and tool builders to use.

The MCP host hosts an MCP client, which communicates to the MCP server. In our case, the host can be AI application or an agent. MCP servers can expose tools and resources (read-only data) and prompt templates or whatever our agent needs.

Once we have built an MCP server with our tools and context, its very easy to share with other projects and developers and streamlining future agent builds. There is s ahuge open source servers that other people have built which we can easily insert into our agent and other types of AI applications compatible with MCP servers like your favourite chat bots or IDEs.

The following code can start a MCP server which is defined in module `resources/2.1_mcp_server.py`. Setting up a MCP serve is very similar to that of a FastAPI server. The syntax is very familiar to me.  

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient(
    {
        "local_server": {
                "transport": "stdio",
                "command": "python",
                "args": ["resources/2.1_mcp_server.py"],
            }
    }
)
```
One important configuration is transport protocol. Thsi could be `STDIO` or `StreamingHTTP` depending on the server. For more info, check out MCP documentation. We can get the tools, resources and prompts available at this server:

```python
tools = await client.get_tools()

# get resources
resources = await client.get_resources("local_server")

# get prompts
prompt = await client.get_prompt("local_server", "prompt")
prompt = prompt[0].content
```
For example:
```python
tools
```
```o
[StructuredTool(name='search_web', description='Search the web for information', args_schema={'properties': {'query': {'title': 'Query', 'type': 'string'}}, 'required': ['query'], 'title': 'search_webArguments', 'type': 'object'}, response_format='content_and_artifact', coroutine=<function convert_mcp_tool_to_langchain_tool.<locals>.call_tool at 0x7fafc03c9d00>)]
```
We can create an agents with tools and prompts from our MCP server.

```python
from langchain.agents import create_agent

agent = create_agent(
    model="gpt-4.1-nano",
    tools=tools,
    system_prompt=prompt
)
```
Now our agent has access to the tools, prompts, resouces on our server. 

```python
from langchain.messages import HumanMessage

config = {"configurable": {"thread_id": "1"}}

response = await agent.ainvoke(
    {"messages": [HumanMessage(content="Tell me about the langchain-mcp-adapters library")]},
    config=config
)
```
If we see the response, we notice that the agent made a tool call to our tools in MCP server:

```o
tool_calls=[{'name': 'search_web', 'args': {'query': 'langchain-mcp-adapters library'}
```

#### Online MCP Servers
The biggest advantage of MCP servers is to connect your agent to other people's MCP servers. There are 100k+ MCP servers you could find online most are open sourced and free and some are not. You just find a config file for the server you need and insert it in `MultiServerMCPClient` arg field and there you have it. Its that easy to connect your agent to the server. The significance here is that you do not need to run your MCP API. The agent will call the server for you.

```python
client = MultiServerMCPClient(
    {
        "time": {
            "transport": "stdio",
            "command": "uvx",
            "args": [
                "mcp-server-time",
                "--local-timezone=America/New_York"
            ]
        }
    }
)

# get the tools from the client and pass them to the agent just like we did before

tools = await client.get_tools()
```
Now we can query time related questions:

```python
question = HumanMessage(content="What time is it?")

response = await agent.ainvoke(
    {"messages": [question]}
)

response['messages'][-1].content
```
```o
'The current time in New York is 20:49 (8:49 PM) on Tuesday, March 24, 2026. Would you like to know the time in a different timezone?'
```

Our agent made a tool call `tool_calls=[{'name': 'get_current_time', 'args': {'timezone': 'America/New_York'}` to find the accurate time.

Its highly recommneded to try the course [MCP: Build Rich-Context AI APPs with Antropic](https://learn.deeplearning.ai/courses/mcp-build-rich-context-ai-apps-with-anthropic/lesson/dbabg/creating-an-mcp-server)



---------------------------------------------------------

###  In-context Learning

In-context learning doesn’t require additional training. A new task is learned from a small set of examples presented within the context or prompt at inference time. 

###### Advantage
- No fine tuning needed
- Reduces the resources and time while improving performance

###### Disadvantage
- Limited to what can fit in-context 
- Complex tasks could require fine tuning for some models


#### Prompts 
Prompts are instructions or context given to an LLM designed to guide it toward generating an output. 

##### Elements of prompt templates
We ususally use prompt templates when we need to repeat a prompt for different inputs.

- Instructions 
  - Clear, direct commands that tell the AI what to do
  - Need to be specific to ensure the LLM understands the task
  - For example: "Classify the following customer review into neutral, negative, or positive sentiment."
- Context 
  -  Information that helps the LLM make sense of the instruction. 
  -  Can be data, any relevant details that shape the AI's response
  -  For example: "this review is part of feedback for a recently launched product"
-  Input data 
   -  the actual data the LLM will process and is different from prompt to prompt
   -  For example: "The product arrived late but the quality exceeded my expectations"
-  Output of model
   -  the part of the prompt where the LLM's response is expected. It's a clear marker that tells the AI where to deliver its analysis.

By combining these elements effectively, you can tailor LLMs to perform tasks ranging from answering queries and
analyzing data to generating content. 


##### Prompt engineering
Prompt engineering is 
- designing and refining the questions, commands, or statements to interact with the AI systems, particularly LLMs. 
- The goal is to carefully craft clear, contextually rich prompts (not just asking questions) tailored to get the most relevant and accurate responses from the AI. 
- This process is fundamental in fields ranging from customer service automation to advanced research and
computational linguistics. 


- directly influencing how effectively and accurately LLMs function. 
- ensures LLMs to generate precise and relevant responses to the context. 
- clearer prompts reduces misunderstandings. 


## Introduction to LangChain

LangChain is an open source Python library. It
- assists in integrating LLMs into their AI applications. 
- provides methods for responding to complex prompts by retrieving data and generating a coherent summary. 
- chains together the retrieval, extraction, processing, and generation operations from the large amounts of text
  

AI developers prefer this framework because of  its 
- **Modularity**: encourages component reuse, reducing development time and effort
- **Extensibility**: add new features, adapt to existing components, integrate with external systems. 
- **Decomposition capabilities (chain of thoughts)**: breaking down complex queries or tasks into smaller, manageable steps to make accurate inferences from context, resulting in relevant, precise responses. 
- **Integration with vector databases**: enables efficient semantic searches and information retrieval.


LangChain consists of several components:
- Documents
-  Chains
-  Agents
-  Language Model
-  Chat Model
-  Chat Message
-  Prompt Templates
-  Output Parsers. 

#### Language Models   
The Language Models in LangChain are the foundation of LLMs. It uses text input to generate text output and helps complete tasks and summarize documents. LangChain uses IBM, OpenAI, Google, and Meta as a primary language model. For example, to generate a response for a new sales approach using a language model, choose LLM model, ensure that the necessary dependencies, such as params are specified, customize the model by adjusting settings such as tokens and temperature. Now that you have a model object, you can generate response text for the inserted prompt. 

##### Chat Models
The next component of LangChain is the Chat Model. A chat model is designed for efficient conversations. It means that it understands the questions or prompts and responds to them like a human. Next, to generate a response, first create a language model and transform the model into a chat model. This converts the chat model into a conversational LLM to engage in dialogues. 

Chat models handle various chat messages to make the model effective in the dynamic chat environment. For example,
-  **Human message** helps user inputs, 
-  **AI message** is generated by the model, 
-  **System message** helps instruct the model, 
-  **Function message** helps the function to call outcomes with a name parameter,
-  **Tool message** helps in tool interaction to achieve results. 
 
Each chat message consists of two key properties. The **role** (human, system etc.) means who is speaking and the **content** means what is being said. Look at the example of a system-generated message, in which the model has given instructions to be an AI bot to respond to the question, What to eat in one short sentence. To respond to this question, the chat model creates a list of messages. First, configure the model as a fitness activity bot using a system message. Then simulate the past conversation using human message and AI message. 

```python
msg = mixtral_llm.invoke(
    [
        SystemMessage(content="You are a supportive AI bot that suggests fitness activitis to a user in one short sentence"),
        HumanMessage(content="I like high-intensity workouts, what should I do?"),
        AIMessage(content="You should try a CrossFit class"),
        HumanMessage(content="How often should I attend?")
    ]
)

print(msg)
```

Using these settings, the model next generates responses based on the previous dialogue. You can operate the chat model using only a human message as input and allow the model to generate responses without system message or AI message queues. It means that the chat bot responds directly to human inputs. 

##### Prompt templates
The next component of LangChain is prompt templates. The prompt templates in LangChain translate the user's questions or messages into clear instructions. The language model uses these instructions to generate appropriate and coherent responses. 

The types of prompt templates are 
- `StringPromptTemplate` is useful for single-string formatting. 
- Chat prompt templates is useful for message lists and specific templates such as Message prompt template including `AIMessagePromptTemplate`, `SystemMessagePromptTemplate`, `HumanMessagePromptTemplate`, and `ChatMessagePrompt Template` allows flexible role assignment. 
- The Messages placeholder provides full control over message rendering. 
- `FewShotPromptTemplate` provides specific examples or shots for LLMs. 

##### Output Parsers
The Output Parsers transform the output of an LLM into a more suitable format for generating structured data. The LangChain provides a library of Output Parsers for various data formats including JSON, XML, CSV, and Panda DataFrames. Output Parsers allow you to tailor the model's output to meet specific data handling needs. 

For example, let's use `CommaSeparatedListOutputParser` to convert LLM's response into CSV format. This Output Parser effectively structures the output and simplifies it to handle and analyze in spreadsheet applications. 

```python
from langchain.output_parsers import CammaSeparatedListOutputParser

format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    template="Answer the user query. {format_instructions}\nList five {subject}.",
    input_variables=["subject"],
    partial_variables={"format_instructions": format_instructions},
)
chain = prompt | mixtral_llm | output_parser

chain.invoke({"subject": "ice cream flavors"})
```
<br>

```o
['Cholocate', 'Vanilla', 'Strawberry', 'Mint Chocolate Chip', 'Butter Pecan']
```


##### Advanced methods of prompt engineering

###### Zero-shot prompt
This type of prompt instructs an LLM to perform a task without any prior specific training or examples. 


```
Classify the following statement as true or false:
The Eiffel Tower is located in Berlin. 
```

Then the model gives the response.  This task requires the LLM to understand the context and information without any previous tuning for the specific query. 

###### One-shot prompt
A one-shot prompt gives the LLM a single example to help
it perform a similar task. For example, 

```
Translating a sentence from English to French
English: "How is the weather today?" 
Franch: "comment est le temps aujourd' hui?"

Now, translate the following sentence from English to French:
English: "Where is the nearest supermarket?"
```
The LLM shows how to translate a sentence from English to French. This serves as a template. Then it's given a new sentence (where is the nearest supermarket) and is expected to translate it into French using the learned format.  The AI uses the initial example to correctly perform the new translation. 

###### Few-shot prompting
The AI learns from a small set of examples before tackling a similar task. This helps the AI generalize from a few instances to new data.  For example, the LLM is shown three statements, each labeled with an emotion. 
```
Here are few examples classifying emotions in statements:
Statement: "I just own my forst marathon!"
Emotion: Joy
Statement: "I can't beleive I lost my keys again."
Emotion: Frustration
Statement: "My best friend is moving to another country."
Emotion: Sadness

Now, classify the emotion in the following staement:
Statement: "That ovie was so scary I had to cover my eyes."
Emotion:
```
The LLM will output the emotion. 


###### Chain-of-thought 

Chain-of-thought or COT prompting is a technique used to guide LLM through complex reasoning step-by-step. This method is highly effective for problems requiring multiple intermediate steps or reasoning that mimics human thought processes. 

LangChain employs a "chain of thought" processing model that breaks down complex queries or tasks into smaller, manageable steps and enhances the model's understanding of context and its ability to make accurate inferences, resulting in more relevant and precise responses. With this ability, LangChain mimics human problem-solving processes, making the interactions with AI more natural and intuitive.

###### Self-consistency
It is a technique for enhancing the reliability and accuracy of outputs by generating multiple independent answers to the same question and then evaluating these to determine the
most consistent result.  In this example, the query is:
```
When I was six, my sister was half my age. Now I am 70. What age is my sister? 

Provide three independent calculations and explanations, then determine the most consistent result.
```

You can view the model outputs with three different ways of calculation and determine a consistent answer. This approach demonstrates how self-consistency can verify the reliability of the responses from LLMs by cross-verifying multiple paths to the same answer. 


#### Tools and applications 
Certain tools can facilitate interactions with LLMs, such as OpenAI's Playground, LangChain, Hugging Face's Model Hub, and IBM's AI Classroom. They allow you to develop, experiment with, evaluate, and deploy prompt. They enable real-time tweaking and testing of prompt to see immediate effect on outputs. Moreover, they provide access to various pre-trained models suitable for different tasks and languages. They also facilitate the sharing and collaborative editing of prompts among teams or communities. Finally, they offer tools to track changes, analyze results, and optimize prompt based on performance metrics. 

##### LangChain for prompt engineering

LangChain 
- uses prompt templates, predefined recipes for generating effective prompt for LLMs. 
- templates might include 
  - instructions for the language model, 
  - a few-shot examples
  - specific question directed at the language model. 
  
Here is a code snippet to apply a prompt template from LangChain. 

```python
from langchain_core.prompts import PromptTemplate
prompt_template = PromptTemplate.from_template("Tell me a {adjective} joke about {content}")

prompt_template.format(adjective="funny", content="chickens")
```

This generates the prompt: "Tell me a funny joke about chickens". This approach simplifies prompt creation, making prompt consistent and adaptable to different contexts. 


In prompt applications, an **agent** is a crucial concept. Powered by LLMs and integrated tools like LangChain, agents perform complex tasks across various domains using different prompts. Transformative applications include Q&A agents with sources, content agents for creation and summarization, analytic agents for data analysis and business intelligence, and multilingual agents for seamless context aware translation and communication. LangChain finds its use in various aspects:

- **Content summarization**: LangChain can automatically summarize articles, reports, and documents, highlighting key information for quick consumption that helps users stay informed about developments in their field without dedicating hours to reading.
- **Data extraction**: The LangChain framework's ability to retrieve specific information from unstructured texts for data analysis and management. It can extract key financial figures from reports or identify relevant case law in legal documents, simplifying the process of turning text into actionable insights.
- **Question answering systems**: Building sophisticated QA systems with LangChain can transform customer support and information retrieval services. By understanding and responding to queries with contextually relevant answers, these systems can provide a higher level of service and efficiency.
- **Automated content generation**: LangChain's capabilities extend to content creation, enabling the automatic generation of written materials. The framework opens new possibilities for automating routine writing tasks, from drafting emails to generating creative writing or technical documentation.

#### LangChain Chains and Agents for Building Applications

LangChain is a platform embedded with APIs to develop applications, empowering them to infuse language processing capabilities. LangChain uses certain tools, including documents, chains, and agents for building applications. 

In LangChain, chains are a sequence of calls. A sequential chain consists of basic steps where each step takes one input to generate one output to create one seamless flow of information. The output from Step 1 becomes the input for Step 2. 

Let's look at the creation of a sequential chain of three individual chains. The aim of this chain is to identify the recipe and the estimated cooking time for the famous dish available in the inserted location. The users leverage Chain 1 for selecting the geographic region to get the famous dish in that location, Chain 2 for providing the recipe, and Chain 3 for estimating the cooking time. Chain 1 in the sequence uses a user's prompt as input for a specific dish based on the user specified location.

To do this, we create a prompt template object using the defined template, specifying the input variable as location. Then create an LLM chain object named location chain, using LLM-based language model, such as Mixtral LLM. Therefore, the output will be stored under the key meal. 

```python
from langchain.chains import LLMChain, SequentialChain

template = ''' Your job is to come up with a classic dish from the area that users suggest.
{location}
YOUR RESPONSE:
'''
prompt_template = PromptTemplate(template=template,
    input_variables = ['location']
)

# Chain 1
location_chain = LLMChain(llm=mixtral_llm, prompt=prompt_template, output_key='meal')
```

Let's look at Chain 2. In the second chain of our sequential setup, use the output from the first chain, that is, the name of the dish is the input. The output from this chain will be the recipe itself. 

```python
template = '''Given a meal {meal}, give a short and simple recipe on how to make that dish at home.

YOUR RESPONSE:
'''
prompt_template = PromptTemplate(template=template,
    input_variables = ['meal']
)

# Chain 2
dish_chain = LLMChain(llm=mixtral_llm, prompt=prompt_template, output_key='recipe')
```

Next is Chain 3. Take the recipe obtained from the second chain as the input. This chain is designed to estimate the cooking time for the meal based on the recipe. Like Chain 2, define a template to estimate the cooking time for a given recipe. Next, create a prompt template with recipe as the input variable. Lastly, create an LLM chain named recipe_chain. 

```python
template = '''Given the recipe {recipe}, estimate how much time I need to cook it.

YOUR RESPONSE:
'''
prompt_template = PromptTemplate(template=template,
    input_variables = ['recipe']
)

# Chain 3
dish_chain = LLMChain(llm=mixtral_llm, prompt=prompt_template, output_key='time')
```

Now using three chains, the overall setup is a sequential chain that wraps all the individual chains together, creating a unified process. By invoking the query through this combined chain, you can trace the flow of information from start to end. You can set the verbose option to true to view the overall output. This provides a clear and detailed view of how each input is transformed through chain into the final output. 

```python
overall_chain = SequentialChain(
    chain=[location_chain, dish_chain, recipe_chain],
    input_variables = ['location'],
    output_variables = ['meal', 'recipe', 'time'], 
    verbose =True
    )

overall_chain.invoke(input={'location': 'China'})
```

And you will get an answer.

#### Memory

Do you know how memory is stored in the LLM applications? In LangChain, memory storage is important for reading and writing historical data. Each chain relies on specific input, such as user and memory. Chain reads from memory to enhance user inputs before executing its core logic and writes the current runs inputs and outputs back to the memory after execution. This ensures continuity and context preservation across interactions. 

The *chat message history class* in LangChain is designed to manage and store conversation histories effectively, including human messages and AI messages. This allows adding messages from the AI and users to the history. In this example, call a `ChatMessageHistory` class and add AI message `Hi` into the memory. The memory will append this AI message as input. 

```python
from langchain.memory import ChatMessageHistory

history = ChatMessageHistory()
history.add_ai_message("Hi")
history.add_user_message("What is the capital of France?")
```
After running this code, we have the following objects in the memory:

```sh
# Memory
[AIMessage(contet="Hi"), HumanMessage(content="What is the capital of France?")]
```

Now, add the user's message, "what is the capital of France?", and the memory will append this as human message input. You would receive responses based on the stored memory. 

#### Agents
Agents in LangChain are dynamic systems where a language model determines and sequences actions such as pre-defined chains. The model generates text outputs to guide actions, but does not execute them directly. However, agents integrate with tools such as search engines, databases, and websites to fulfill user requests. For example, if a user asks for the population of Italy, the agent uses the language model to find options, query a database for details, and return a curated list. This shows the agent's ability to autonomously leverage LLM reasoning with external tools. 

In this example, let's create a Pandas DataFrame agent using LangChain. This agent allows users to query and visualize data with natural language. To set it up, instantiate the `create_pandas_dataframe_agent` class. 

```python
df = pd.read_csv("example.csv")
agent = create_pandas_dataframe_agent(
    mixtral_llm,
    df,
    verbose=True,
    return_intermediate_steps=True
)

agent.invoke("How many rows in the dataframe?")
```

In this example, the LLM transforms queries into Python code, executed in the background, enabling precise answers to the number of rows of the data frame. 



### Choose the right model for your use case

If you want your AI garden to grow, you need to ensure that you're using a variety of models (multi-model approach). This means you can pick and choose from different models to find the right one for the right use case, which gives you the opportunity to look at how each of those models is designed as you find the right fit. You need to ask specific, important questions. 
  - Who built it? 
  - What data was it trained on? 
  - What guardrails are in place for it? 
  - What risks and regulations do you need to consider and account for? 

Identifying the best use case to fit your business needs. That begins with a **prompt**.  
- Find an optimal model that satisfies your prompt. What a good prompt does is clearly articulate your use case and the problem you're solving with AI. 
- Research the available models, considering model size, performance, costs, risks, and deployment methods. Pass the prompt to various models to experiment and see which works best.
- Start with a large model and work with it until you satisfy your original prompt. Then, try to duplicate the result using smaller models.  That enables you to choose the best model for the use case.
- Continually evaluate and govern that model with ongoing testing so you assess how it's working based on performance and cost benchmarks. 
- Continually update the data and the prompt where needed to keep it relevant and also test new models as they become available. 
- In addition to the three elements of performance, accuracy, reliability, and speed, also consider size, deployment method, transparency, and any potential risks. 
- Implement these using a team that not only crosses disciplines, but also crosses lines of business. Don't think of it as proprietary to any one department.
- Keep on continuous testing, governance, and optimization, all of which are essential to keep that model up to date and running optimally. 

### Comparing AI system Designs
When designing AI systems, the correct approach depends on the task's complexity, adaptability needs, and operational requirements. Let's compare three paradigms: 
- Single LLM features 
- Structured workflows
- Autonomous AI agents


At the most basic level, you can use LLMs for simple, single-turn tasks with no memory or context across calls.

##### Single LLM features

Single LLM features have the following key characteristics:

- Stateless processing: No retention of information or context across interactions.
- Direct input-output flow: Straightforward request-response mechanism.
- Predefined tasks: Suitable only for clearly defined, single-step actions.

###### Best uses
Simple, well-defined tasks that require no memory or multi-step logic such as:
  - Text summarization
  - Sentiment classification
  - Information extraction
  - Translation

###### Advantages
- Speed and simplicity: Fastest to build and run
- Deterministic output: Same input, same output
- Low cost: Minimal compute and orchestration overhead
Limitations

###### Limitations:

- No adaptability: Cannot handle context or dynamic decision-making
- No memory: Each input is processed independently

##### Structured workflows: Multi-step, predictable processes

Structured workflows orchestrate LLM and tool calls through explicit, deterministic code paths. Consider processing insurance claims, where each document is scanned, information is extracted, validated, and stored. Each step must follow a precise, predictable order, making structured workflows ideal.

Structured workflows have the following key characteristics:

- Deterministic execution: Inputs produce consistent outputs.
- Explicit control flow: All steps and decisions are predefined.
- Predefined tool chains: Tool use is fixed and transparent.

###### Best uses

- Repetitive, multi-step tasks with clear logic and minimal ambiguity
- Regulatory or compliance-driven applications
- Scenarios requiring consistency, traceability, and auditability

You'll find structured workflows work well for the following scenarios:

- Document and data pipelines (Optical Character Recognition (OCR) → extraction → validation → storage)
- Batch report generation
- Financial and healthcare transaction processing

###### Advantages
- Predictable and reliable: Easy to monitor, debug, and audit
- Cost-efficient: No unnecessary exploration
- Compliance-ready: Supports versioning, error handling, and audit trails

###### Limitations

- Rigidity: Difficulty adapting to new or ambiguous scenarios
- Development overhead: The necessity to code each exception or variant

##### Autonomous agents: Flexible, context-aware reasoning

Autonomous agents allow LLMs to plan sequence actions and adapt as conditions change. Agents choose which tools to use and how to achieve their goals based on real-time context and feedback. Imagine an AI-driven virtual assistant helping a user plan a vacation. It dynamically gathers user preferences, researches destinations, suggests accommodations, and adapts recommendations based on feedback. This requires an autonomous agent capable of planning, context-awareness, and iterative improvement.

###### Agents Capabilities

Autonomous agents have the following core capabilities:

- Dynamic planning: Decomposes goals and adjusts steps as needed
- Contextual awareness: Remembers past steps and adapts to user and environment feedback
- Tool orchestration: Selects tools and changes strategies dynamically

###### Best uses

- Complex, open-ended tasks with unclear solution paths
- Scenarios requiring real-time adaptation and reasoning
- Environments with high variability or need for personalization

Examples

- Research agents synthesizing new information
- Adaptive customer support and troubleshooting
- Automation that iteratively refines results based on feedback

###### Advantages
- Highly adaptable: Handles unforeseen situations
- Dynamic decision-making: Iterates and improves over time
- Reduces human intervention: Manages complexity autonomously

###### Limitations

- Unpredictable outcomes: Requires robust monitoring and safeguards
- Higher complexity and cost: More difficult to debug and guarantee compliance


| AI System type	| Process | 	Use Case	| Pros | 	Cons
| ------------- | --------- | --------- | ----------- | --------- |  
| Single LLM	| Input → LLM → Output	| Summarization, classification | 	Simple, fast, low cost |	Not adaptable, lacks context |
| Workflow	| Parallel LLMs → Aggregation → Output	| Structured multi-step tasks |	Predictable, easy to audit |	Rigid, not dynamic |
| Agent	| Plan  → Act → Observe → (repeat agent loop) | 	Complex, adaptive automation |	Flexible, learns from feedback |	Unpredictable, complex, pricier |

In practice, hybrid architectures are common. They combine workflow reliability with agent flexibility to achieve the best results.

Recent standards, including **Model Context Protocol (MCP)**, from Anthropic, and Agent Communication Protocol, or ACP, from IBM, ease integration, monitoring, and governing both approaches at scale.

##### Key takeaways

When selecting an AI agent, reflect on the following considerations:

- Start simple: Use the most straightforward solution that fulfills your needs. For example, you can use single LLM features for atomic needs.
- Leverage workflows: When predictability, compliance, and efficiency matter
- Deploy agents selectively: Only when adaptability, complex reasoning, or open-ended problem solving are required


### When (and When Not) to Use AI Agents

AI agents offer powerful capabilities, but they aren't always the best solution. So we need to evaluate when agents make sense—and when simpler tools are more effective. This is about the balance between innovation and practicality.

Not all AI systems operate at the same level of complexity. The following table describes the spectrum of differences. Before you build an AI agent, ask yourself the following questions:

1. Is the task ambiguous or predictable?

    Use agents when the task is ambiguous:

    - The decision path is unclear or cannot be mapped in advance
    - Tasks involve exploration, troubleshooting, or creativity

    Use workflows when the task is predictable:
    - You can define all rules and outcomes
    - The process follows a clear, repeatable structure

2. Is the value of the task worth the cost?

    AI agents are more expensive to operate due to exploration overhead. They can consume 10 to 100× more tokens than a workflow. 

    | Scenario |	Recommendation |
    | -------- | ----------- |
    | Strategic planning with high ROI	| Use an agent
    | Basic customer support task	| Use a workflow instead
3. Does the agent meet minimum capabilities?

    Before launch, test the agent on three to five key skills.

    Here are some examples:
    - A research agent must identify, filter, and summarize credible sources
    - A coding agent must write, fix, and validate code snippets
    - A customer support agent must classify issues, resolve common queries, and escalate complex cases appropriately
    - A data analysis agent must clean datasets, detect anomalies, and summarize key trends
    
    If the agent fails these tests, scale back or redesign the agent.

4. What happens if the agent makes a mistake?

    Evaluate the answers to these questions

    - Can you catch and correct errors quickly? If so, then using an agent might be appropriate.

    - What's the risk if something is missed? Does the consequence of missing the answer affect the customer’s or organization’s well-being or safety?

    - Does the agent include built-in correction or validation tools?

    Use agents when risk is manageable or reversible.

#### Current AI agent challenges

Even powerful agents have challenges.

Challenge	Why It Matters
- Reasoning inconsistency:	Agents may succeed once but fail on similar tasks
- Unpredictable costs:	Resource use can spike depending on complexity
- Tool integration issues:	Agents need well-integrated tools and stable APIs

### When not to use agents

There are some situations when agents are not a good solution. Avoid agents for:

- High-volume, low-margin tasks, such as basic chat support
- Real-time applications, such as instant fraud detection
- Zero-error systems, including medical or security decisions
- Heavily regulated industries need deterministic outcomes
- Guidelines for managing risk when deploying agents

### Tools, Agents, and Function Calling in LangChain

Tools are essentially functions made available to LLMs. For example, a weather tool could be a Python or a JavaScript function with parameters and a description that fetches the current weather of a location.

A tool in LangChain has several important components that form its schema:

- **Name**: A unique identifier for the tool.
- **Description**: A brief explanation of the tool's purpose.
- **Parameters**: The inputs the tool expects to function correctly.

This schema enables the LLM to understand when and how to use the tool effectively.

##### Tool Calling

Contrary to the term, in tool calling, LLMs do not execute tools or functions directly. Instead, they generate a structured representation indicating which tool to use and with what parameters.

When you pose a question to the LLM that requires external information or computation, the model evaluates the available tools based on *their names and descriptions*. If it identifies a *relevant tool*, the model generates a structured output (typically formatted as a JSON object) that specifies the tool's name and appropriate parameter values. This is still text generation, just in a structured format intended for tool input. An *external system* then interprets this structured output, executes the actual function or API call, and retrieves the result. This result is subsequently fed back to the LLM, which uses it to generate a comprehensive response. Here's the workflow example in simple terms:

Define a weather tool and ask a question like: "What's the weather like in NY?"
- The model halts regular text generation and outputs a structured tool call with parameter values (e.g., "location": "NY").
- Extract the tool input, execute the actual weather-checking function, and obtain the weather details.
- Pass the output back to the model so it can generate a complete final answer using the real-time data.

###### Function calling vs. tool calling

Function calling and tool calling refer to essentially the same concept. Both describe the same capability: enabling an LLM to request specific external functions to be executed with structured parameters The concepts, workflows, and implementations are functionally identical - the difference is primarily in naming convention rather than technical distinction.

###### Tools in LangChain

Tools are utilities designed to be called by a model: their inputs are structured in a way that models can generate, and their outputs are intended to be passed back to the model. These tools perform specific actions such as searching the web, querying databases, or executing code. A toolkit is a collection of related tools designed to work together for a common purpose or integration.

###### Ways to initialize and use tools

LangChain provides several methods to initialize and use tools:

- Using Built-in Tools: LangChain offers a variety of built-in tools for common tasks. For example, the `WikipediaQueryRun` tool allows fetching data from Wikipedia.
- Loading Tools with `load_tools`: LangChain provides a `load_tools` function to load multiple tools conveniently. This function can be used to load tools like wikipedia, `serpapi`, `llm-math`, etc.
- Creating Custom Tools: You can define your own tools using the Tool class or the `@tool` decorator. This allows you to wrap any function as a tool that the LLM can invoke.
- Tools as OpenAI Functions: LangChain tools can be converted to OpenAI functions using the `convert_to_openai_function` utility. This allows you to use LangChain tools with OpenAI's function calling API. Additionally, you can use bind_functions or bind_tools methods to automatically convert and bind tools to your OpenAI chat model.

LangChain supports a wide range of external tools bundled into toolkits. These allow LLMs to interact with real-world data sources, APIs, and services. Some popular categories include:

- Wikipedia: Fetch summaries and information from Wikipedia articles.
- Search engines: Integrate with search engines like Bing, Google, and DuckDuckGo to fetch real-time search results.
- APIs: Access various APIs for tasks like weather data retrieval, financial data analysis, etc. and many more!

##### Agents

Agents are decision-making systems powered by LLMs that can reason, use tools, access memory, and take actions to complete tasks. Unlike a standalone LLM that only generates text, an agent figures out how to respond, which might involve searching for information, querying a database, or executing code.

An agent is a high-level orchestration system rather than a single function (like tool); it encapsulates the LLM itself along with supporting components like tools, memory, and an execution framework. In essence, tool calling allows agents to leverage tools to interact with the real world, making them capable of dynamic, context-aware decision-making and real-world problem solving beyond text generation alone.

### Architecture of an AI agent in LangChain


Let's discuss all the components in the architecture of an AI agent in LangChain.

###### AI Agent
This is the main intelligence unit. It includes LLM, tools, memory, and logic to take actions. The agent is responsible for processing input queries and figuring out how to respond, not just with text, but by taking actions when needed.

###### Large Language Model (LLMs)
At the core of the agent, the LLM interprets the input and determines what to do next. It may decide to:

- Recall something from memory (via RAG)
- Use a tool
- Directly generate a response

######  Tool(s)

As explained earlier, they are functions or external capabilities that an agent can use. The LLM generates structured requests specifying which tool to use and with what parameters. The tool is then executed outside the model, and the result is returned.

###### Memory
Memory allows the agent to store and retrieve useful information, enabling long-term context and personalized interaction. It could include:

- Short-term in-RAM data
- Structured storage like SQL
- Semantic memory in VectorDBs (e.g., for document search)

###### Action
When the LLM decides a tool needs to be used, it outputs a structured "action," which is then executed by the system. The outcome of this action is either shared with the user or used in further reasoning steps.

###### Connection to the external world
The World represents everything outside the AI agent — the OS, internet, APIs, physical devices, etc. The agent can interact with these systems through tools, bridging the gap between the LLM and real-world applications.

###### Example

Let's see an example of the flow:

User asks:
"What is the weather in New York?"

The input is received by the AI agent. This is a natural language query requiring real-time information that the LLM alone may not have. 
- LLM processes the query and identifies the need for a tool:
  - The LLM understands that it doesn't have real-time weather data and determines that a weather tool (e.g., an API) should be used. It prepares a structured tool call, specifying the tool name and input parameters (e.g., `location = "New York"`).
- Tool is invoked as an action:
  - The agent passes this structured request to the external weather tool. The tool is executed, reaching out to a weather API to fetch the latest conditions for New York. This is the "action" step, where the LLM's plan is turned into a real-world operation.
- Memory may be referenced or updated:
  - If the user has asked about weather before, the agent might reference this prior context from memory to personalize or compare results. Memory can also be updated with the new interaction for future reference.
- LLM receives tool output and generates a response:
  - Once the tool returns the result (e.g., "It's currently 68°F and sunny"), the LLM takes this output and uses it to generate a complete, user-friendly response.

- Agent responds and completes the task using real-world data:
  - The user receives the final output:
  "It's currently 68 degrees and sunny in New York City."

The task is now completed using both the LLM's processing capability and external world data accessed via the tool.

##### Agents in LangChain

LangChain offers several methods to create and utilize agents:

###### Using built-in agent types
LangChain offers predefined agent types such as zero-shot-react-description and chat-zero-shot-react-description. These are useful for simple reasoning tasks where the model decides which tool to use based on the description alone.

###### Creating agents with OpenAI functions
LangChain supports agents that can leverage OpenAI's function-calling capabilities. You can create such agents using create_openai_functions_agent, which makes it easy to interact with structured tools in a safe and predictable way.

###### Building agents with LangGraph
LangGraph is a low-level orchestration library that provides more flexibility and control. Using the high-level create_react_agent method, you can create agents capable of complex reasoning, action chaining, memory integration, and real-time streaming.

###### Executing agents with AgentExecutor
Agents are executed using AgentExecutor, which manages the loop of calling the LLM, processing tool outputs, and determining when the final response is ready. This ensures the agent can dynamically react to intermediate results during its reasoning process.

###### Adding memory
By using components like MemorySaver, agents can maintain context across multiple interactions. This allows for personalized and context-aware responses, where an agent can remember things like user preferences, past questions, or conversation history.

### LangChain Tool Creation Methods

All tools built in LangChain follow a basic blueprint called `BaseTool`. It establishes the core functionality and structure that tools need to implement. Both `Tool` and `StructuredTool` classes inherit from `BaseTool`, making it the foundation of the tool hierarchy.

- Tool Creation by Subclassing `BaseTool` Class:

    Subclassing from BaseTool is the most flexible method, providing the largest degree of control at the expense of more effort and code. This approach allows you to define custom instance variables, implement both synchronous and asynchronous methods, and have complete control over tool behavior.
    ```python
    from langchain.tools import BaseTool
    from typing import Optional
    from langchain.callbacks.manager import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun

    class CustomCalculatorTool(BaseTool):
      name = "Calculator"
      description = "Useful for mathematical calculations"

      def _run(self, query: str, run_manager:Optional[CallbackManagerForToolRun] = None) -> str:
        ''' Use the tool synchronously'''
        # Custom logic here
        return "Calculation result"
      async def _arun(self, query:str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None)  -> str:
        ''' Use the tool asynchronously'''
        # Custom async logic here
        return "Async calculation result"

    ```

- Tool Creation using the `Tool` Class:
    LangChain provides a `Tool` class (which inherits from `BaseTool`) to encapsulate a function along with its metadata, such as name, description, and argument schema. This allows for more control over the tool's behavior and integration. Note: This is the traditional approach that primarily handles single string inputs, though it shares the same base class as `StructuredTool` for backward compatibility.

    ```python
    from langchain.agents import Tool

    def add_numbers(a: int, b: int) -> int:
      '''Add two numbers.'''
      return a + b

    add_tool = Tool(
      name="AddTool",
      func=add_numbers,
      description="Adds a list of numbers and returns the result."
    )
    ```

- Tool Creation Using `@tool` Decorator
   
    The `@tool` decorator is a convenient way to create a tool from a Python function. It automatically infers the tool's name, description, and argument schema from the function's signature and docstring. Tools created using `@tool` decorator create a `StructuredTool` (which inherits from `BaseTool`), which allows LLMs to handle more complex inputs, including named arguments and dictionaries, improving flexibility and integration with function calling models.

    ```python
    from langchain_core.tools import tool

    @tool
    def divide_numbers(a: int, b:int) -> float:
      ''''Divide a by b'''

      return a / b
    ```

- Structured Tool Creation

    `StructuredTool` (which inherits from `BaseTool`) provides the most flexibility for function-based tools and is the modern approach for creating tools that can operate on any number of inputs with arbitrary types. It supports complex argument schemas and both synchronous and asynchronous implementations.

    ```python
    from langchain_core.tools import StructuredTool

    def multiply_numbers(a: int, b: int) -> int:
      '''Multiply two numbers'''
       return a + b

    async def amultiply_numders(a: int, b: int) -> int:
      '''Multiply two numbers asynchronously '''

      return a + b

    calculator = StructuredTool.from_function(
      func=multiply_numbers,
      coroutine=amultiply_numbers,
      name="Calculator",
      description="multiply numbers",
      return_direct=True
    )
    ```
    Note: The `@tool` decorator and `StructuredTool` are the recommended modern approaches for function-based tools, while `BaseTool` subclassing provides the ultimate flexibility for complex custom behaviors. The Tool class remains for compatibility with existing codebases.

##### Checking & Using Your Tools

  Once you've defined your tools, here's how you can inspect and use them:

   - Tool Schema
    You can look at a tool's name, description, and what inputs it expects using the following:
      ```python
      print("Name: \n", divide_numbers.name)
      print("Description: \n", divide_numbers.description)
      print("Args: \n", divide_numbers.args)
      ```

   - Direct Invocation of Tools
    Tools can be invoked directly using the `invoke()` method by passing a dictionary of arguments. This is especially useful for testing outside of an agent or LLM context.

      ```python
      result = divide.invoke({"a": 10, "b": 2})   # output: 5.0
      ```

  - Tool Binding to Models
    Before a model can use tools, you need to tell it which ones are available and what kind of information they need.

      ```python
      tools = [add_tool, divide_numbers]
      llm_with_tools = llm.bind_tools(tools)
      ```
  - Tool Invocation via Model
    
      Once tools are bound to a model, the model can decide to invoke them based on the input prompt. The model's response will include the tool call details, which the application can then execute.

      ```python
      response = llm_with_tools.invoke("What is 10 divided by 2?")
      ```

##### LangChain Built-in Tools

  LangChain offers a collection of tools that are ready to use. Always check their official guides, as some might have costs or specific requirements.

  | Use case | Tool |  Purpose |  
  | ---- | ----- | -----
  Search | SerpAPI, Wikipedia, Tavily | Web and knowledge search 
  Math & Code |  LLMMathChain, Python REPL, Pandas | Computation, data analysis 
  Web/API |  Requests Toolkit, PlayWright |  Web requests, scraping |
  Productivity |  Gmail, Google Calendar, Slack, GitHub  |Messaging, scheduling, repo management |  
  Files/Docs |  FileSystem, Google Drive, VectorStoreQA | Document access, file ops 
  Finance |  Stripe, Yahoo Finance, Polygon IO | Payments, market data 
  ML Tools |  DALL-E, HuggingFace Hub  |  Image/gen model interaction

##### Agents in LangChain

Agents are intelligent systems powered by LLMs that use tools, memory, and reasoning logic to perform complex tasks.

Components of an Agent:

- LLM: Core reasoning unit.

 - Tools: External functions the LLM can call.

 - Memory: Stores conversational or task context.

 - Executor: Loops over reasoning steps until a final answer is ready.

###### Agent Types:
- `zero-shot-react-description`
- `chat-zero-shot-react-description`
- `create_openai_functions_agent`
- `LangGraph-based agents`


#### LangChain LCEL Chaining Method

LangChain Expression Language (or LCEL) is a pattern for building LangChain applications that utilizes the pipe `|` operator to connect components. This approach ensures a clean, readable flow of data from input to output. 

To create a typical LCEL pattern, you need to 
- Define a template with variables and curly braces 
- Create a prompt template instance 
- Build a chain using the pipe operator to connect components. 
- Invoke the chain with input values. 
  
  
Let's see this in action with a concrete example. In LangChain, **runnables** serve as an 
- interface and building blocks 
- connect different components like LLMs, retrievers, and tools into a pipeline. 

There are two main runnable composition primitives. 



**RunnableParallel** runs multiple components
concurrently while using the same input for each. 

```python
from langchain_core.runnables import RunnableParallel

chain = RunnableParallel({
  "key1": runnable1,
  "key2": runnable2
})
```

**Runnable sequence** chains components sequentially,
passing the output from one component as input to the next. 

However, LCEL provides elegant syntax shortcuts. For example, instead of using runnable sequence, the same sequential chain can be created by simply connecting runnable 1 and runnable 2 with a pipe, 
```sh
chain = Runnable 1 | Runnable 2 
```
making the structure more readable and intuitive. 

LCEL also converts regular code into runnable components. 
  - When you use a dictionary, it becomes a runnable
parallel, which runs multiple tasks simultaneously. 
  - When you use a function, it becomes a RunnableLambda, which transforms inputs. 

For more complex workflows, consider using LangGraph while still leveraging LCEL within individual nodes. As you develop your own applications, take advantage of LCEL's strengths, including parallel execution, async support, simplified streaming, and automatic tracing. These capabilities enhance both the power and maintainability of your applications. 


#### Build Interactive LLM Agents

Now that you've built a foundation in setting up LLMs with basic tool integrations, it's time to take things a step further. Begin by setting up the actual question you want the model to answer. "What is 3 plus 2?" First, insert the user's question into the chat history so the model can see it as a part of the conversation context. Convert the plain text question into a HumanMessage, which is a symbol wrapper that tells LangChain, This text came from the user." 

```python
from langchain_core.messages import HumanMessage

query = "What is 3 + 2?"
chat_history = [HumanMessage(content = query)]
```
It carries the user's input and marks it appropriately for the model during the conversation.  Chat history list will hold all messages exchanged during the conversation, including user inputs, tool outputs, and model responses, ensuring th

It's time to run the tool enabled model. First, pass the complete chat history, which contains the user's question, into LLM with tools. At this point, the model reviews the conversation context, identifies the available tools, select the appropriate one, such as add function, and extract the parameters. 

```python
response1 = llm_with_tools.invoke(chat_history)
print(response1)
```

```o
AIMessage(
  tool_calls = [{
    "name": "add",
    "args": {
      a: 3,
      b: 2
    },
    "id": "call_add123",
    "type": "tool_call"
  }]
)
```

Notice the model's output. Instead of a plain text response, an AI Message object is returned containing a tool calls array. Focus only on the part of the message related to tool usage. This is the model's way of saying, "I want to call this function with these arguments." Next, extract the details of the tool call and manually execute the addition ourselves. Finally, append the AI Message, including the tool call instruction, into chat history so the next model invocation has access to both the user's original query and the model's previous output. 

```python
chat_history.append(response1)
```

Let's take a quick look at its key details. Here's the tool call parameters. `Add` is the name of the tool the model wants to call. The function arguments are a JSON string that specifies the inputs to pass into the tool. For example, "A equals 3 and B equals 2." Here is a unique identifier for this tool call. It's used to link the response back to the request. This is if more than one tool is called. The type field specifies that this is a tool call, not a text or other output type. Now parse the tool parameters, execute the tool manually, and then send the result back to the LLM as part of the conversation. First, parse the add tool with arguments "A equals 3 and B equals 2" and feed them into the function. Then extract the model's tool-call instructions into a variable. 

```python
tool_calls_1 = response_1.tool_calls
```

You will get a list showing exactly which tools the LLM selected and with what arguments, ready to be inspected and manually invoked. Next you will extract the tool's name from the first entry by running this command. You will then pull out "AddTool," telling exactly which function to call next. 

```python
tool_1_name = tool_call_1[0]['name']
tool_1_args = tool_call_1[0]['args']
```

Extract the parameters to pass the function using the args key, for example "A3 and B2." Finally extract the tool call "ID." This ID links the tool's result back to the model's original request, so the LLM knows which tool call the response belongs to. This is especially important when multiple tools are called at once. 

Use the tool map created earlier to call the function you need. The key is the tool name determined by the LLM stored in tool_1_name. In addition, use the tool parameters provided by the LLM stored in tool_1_args. 

```python
tool_map = {"tool_1_name": AddTool, "tool_2_name": DivideTool}

tool_response = tool_map[tool_1_name].invoke(tool_1_args)
```

The tool map returns the output generated via the tool, in this case "2 plus 3 equals 5." Finally, wrap the tool's output in a Tool Message by creating a Tool Message object. 

```python
from langchain_core.messages import ToolMessage

tool_message = ToolMessage(content=tool_response, tool_call_id=tool_call_1_id)
```

Finally, update the chat history by appending the new tool message to the chat history list. The history now contains the original HumanMessage, the AIMessage from the model, and the new ToolMessage with the tool's result. Begin by passing the updated chat history to the LLM with tools using the invoke method. 

```python
chat_history.append(tool_message)

answer = llm_with_tools.invoke(chat_history)
```

The LLM will generate a final response using the tool's output, but format the answer naturally as a part of the conversation. You can create an agent class that encapsulates everything you did above. 

```python
class ToolCallingAgent:
  def __init__(self, llm):
    self.llm_with_tools = llm.bind_tools(tools)
    self.tool_map = tool_map
  
  def run(self, query: str) -> str:
    chat_history = [HumanMessage(content=query)]

    response = self.llm_with_tools.invoke(chat_history)
    if not response.tool_calls:
      return response.content
    .
    .
    .
```


###  AI-powered SQL agents
In this section, we describe benefits of AI-powered SQL agents, explain some AI-powered SQL agent capabilities, identify AI-powered SQL agent limitations and considerations, and explain how AI-powered SQL agents retrieve information. 

First, let's explore the benefits of AI-powered SQL agents. AI-powered SQL agents bridge the gap between natural language and SQL, enhancing data accessibility. SQL is a powerful tool, but using SQL requires specialized knowledge. Enabling a natural language interface provides a broader range of users with the ability to access and interpret data without the need for deep technical skills. AI-powered SQL agents provide the following capabilities. 
- Can read and understand the database schemas, which enables them to answer questions about specific tables
- Retrieve schemas only from relevant tables
- Query management. 
  - Support multi-step querying when one query isn't enough to answer the question fully. 
  - If a query fails, the agent 
      - captures the error
      - analyzes the traceback
      - automatically retries the request using a corrected version of the query. 

AI interpretations of queries can be inaccurate and complex queries might require manual adjustments. So continuous testing and validation are essential for reliability. 


Let's explore how AI-powered SQL agents powered by large language models process a query. 
1. The user asks a question using natural language
2. The AI-powered SQL agent receives the question
3. The LLM interprets the question and generates an SQL query 
4. A database connector sends the SQL query to the database
5. The database processes the SQL query. 
6. The database sends the raw data back to the database connector. 
7. The database connector passes the data back to the LLM. 
8. The LLM parses, processes, and formats the raw data into a clear, readable response. 
9. Finally, the AI-powered SQL agent displays the user's answer in clear, natural language, completing the flow from the initial question to the final response.








-----------------

#### Natural Language Interfaces for Data Systems

- Explain how natural language interfaces convert user queries into data insights

- Differentiate between rule-based, machine learning, and hybrid NLI approaches

Introduction

Data is the lifeblood of modern organizations, but its value can only be realized when people effectively access and analyze it. Traditionally, accessing data has required specialized technical skills such as SQL programming or familiarity with business intelligence tools. This technical barrier has created a divide between those who can query data directly and those who need insights but lack the technical expertise.

 Natural language interfaces (NLIs) for data systems bridge this gap by allowing users to interact with databases and analytics platforms using everyday language. Instead of writing complex SQL queries, users can simply ask questions such as “What were the sales in the Northeast region last quarter?” or “Show me customers who purchased more than $1000 last month.”

This reading explores how NLIs work, their evolution, key technologies, and design approaches. By the end, you'll be able to explain, compare, and evaluate different NLI systems and their applications in real-world data environments.

##### The evolution of data access interfaces

The journey from traditional query methods to natural language interfaces has evolved through several stages:

- Command-line interfaces - Required precise syntax and technical expertise
- Graphical query builders - Provided visual tools, but still required understanding of data structure
- Dashboard interfaces - Offered pre-built visualizations with limited flexibility
- Natural language interfaces - Enable intuitive, conversational access to data

This evolution represents a fundamental shift in how we think about human-data interaction, moving from requiring users to learn the language of computers to enabling computers to understand human language.

##### How natural language interfaces work

Natural language interfaces for data systems transform human questions into structured queries that databases can execute. This process involves several sophisticated components working together:

1. User input query
The process begins with the user submitting a natural language question. Unlike traditional database queries, these questions:
   - Use everyday vocabulary rather than technical terms
   - May be ambiguous or incomplete
   - Contain implicit assumptions about what data is important

2. AI-driven query formulation
This critical component interprets the natural language and transforms it into a structured format:
   - Identifies key entities and metrics mentioned in the query
   - Maps natural language terms to database schema elements
   - Determines the analytical intent (comparison, trend analysis, distribution, etc.)
   - Formulates the appropriate technical query (SQL, API calls, etc.)

    The AI leverages its understanding of both language semantics and data structures to bridge the communication gap between humans and machines

3. Database data extraction
Once the AI has formulated a structured query:
   - The system connects to the relevant data sources
   - Executes the query against databases or data warehouses
   - Retrieves the necessary raw data
   - Handles authentication, optimization, and error management

    This step involves translating the AI's understanding into actual data retrieval operations

4. Data analysis process
With the raw data retrieved, the system:
   - Cleans and preprocesses the data
   - Applies appropriate statistical methods
   - Performs calculations and aggregations
   - Identifies patterns, trends, or anomalies
   - Prepares the data for visualization or presentation

    This step transforms raw data into meaningful analytical results

5. Insight synthesis
The system goes beyond just processing numbers to:
   - Interpret the analytical results in context
   - Identify key findings and significant patterns
   - Prioritize information based on relevance
   - Generate natural language explanations of the findings
   - Select appropriate visualization methods

This is where AI adds value—not just calculating results but understanding their significance

6. Presentation insight
Finally, the system delivers insights back to the user:
   - Presents visualizations (charts, graphs, dashboards)
   - Provides natural language summaries of key findings
   - Offers contextual explanations and interpretations
   - Suggests potential follow-up questions or analyses

    The output combines visual and textual elements to communicate findings effectively, regardless of the user's technical background

#### Types of natural language interfaces for data

There are two primary types of natural language interfaces for data systems:

1. One-shot query systems
These systems handle individual, standalone queries without maintaining context between interactions:
  Strengths:
     - Simpler to implement
     - Good for direct, specific queries
     - Easier to optimize for performance

    Limitations:
    - Cannot handle follow-up questions
    - No memory of previous interactions
    - Limited ability to refine or clarify questions

2. Conversational interfaces
These systems maintain context across multiple interactions, enabling a dialogue between the user and the system:

    Strengths:
   - Support follow-up questions and clarifications
   - Enable iterative data exploration
   - More natural interaction pattern
   - Can disambiguate vague queries through dialogue

   Limitations:
     - More complex to implement
     - Require dialogue state tracking and management
     - May have higher latency due to context processing
Conversational interfaces for data are rapidly gaining popularity because of their unique ability to enable exploration of data and derivation of insights in small incremental steps as the conversation with the data progresses. They can understand, respond, and clarify ambiguity through interactions with the user in natural language, while persisting the context of the conversation across multiple turns.

#### Key technologies powering natural language interfaces

Several advanced technologies work together to make natural language interfaces possible:

1. Foundation Language Models
Large language models such as GPT, BERT, and others provide the backbone for understanding natural language queries:
   - Interpret user intent from natural language
   - Handle various phrasings of the same question
   - Understand domain-specific terminology
   - Generate human-like explanations and summaries

1. Semantic parsing and named entity recognition
These techniques identify the key components of a query:

   - Extract entities (products, regions, metrics) from text
   - Understand relationships between entities
   - Map natural language terms to database schema elements
   - Identify query operations (filtering, sorting, aggregating, etc.)

    Semantic parsing is particularly critical for natural language interfaces as it enables the extraction of a structured semantic representation of the user query. This involves parsing natural language queries for detecting intents and entities, which are then mapped to database schema elements.

1. SQL generation
 Converting natural language to database queries requires:

   - Building syntactically correct SQL statements
   - Handling complex queries with joins, nested conditions
   - Managing different database dialects
   - Optimizing queries for performance
 
    The complexity of the structured queries generated such as SQL and SPARQL makes the query translation from natural language very challenging. The system needs to infer appropriate entity mappings from natural language to schema elements and derive correct query structures from linguistic patterns embedded in a query.

4. Dialogue management
For conversational interfaces, dialogue management systems:
   - Track the state of the conversation
   - Maintain context across multiple queries
   - Identify ambiguities that need clarification
   - Manage the flow of the interaction

    Dialogue management includes several key components:

    - State tracking: Keeping track of the current state of data exploration given the prior set of queries issued by the user in a data exploration session.
    - Decision making: Choosing an appropriate external knowledge source and generating structured queries to retrieve data for a given user query.
    - Natural language response generation: Providing a natural language response conditioned on the identified intents, extracted entities, the current context of the conversation, and the results obtained from external knowledge sources.

#### Approaches to building natural language interfaces

There are two main approaches to building natural language interfaces for data systems:

###### Rule-based approaches
Rule-based systems use semantic indices, ontologies, and knowledge graphs to identify entities in queries and understand their relationships:
   - They map parts of a natural language query to concepts and relationships in the underlying data model
   - They use grammar-based techniques for query interpretation and SQL generation
   - They're strong in semantic understanding and domain adaptation

  However, these systems can be brittle when handling linguistic variations in natural language queries.

###### Machine learning/deep learning approaches
These systems (often called text-to-SQL systems) use deep learning to translate natural language to SQL:

   - They encode user input as features using techniques such as word embeddings or pre-trained language models
   - They train models to generate SQL queries without explicit entity mapping
   - They're more robust to paraphrasing and linguistic variations
    
  However, they typically require large amounts of training data and may struggle with complex queries or new domains.

###### Hybrid approaches
Emerging hybrid approaches combine the strengths of both rule-based and machine learning systems:

   - They use deep learning for entity tagging or natural language understanding
   - They incorporate domain knowledge through ontologies or knowledge graphs
   - They combine statistical models with rule-based techniques for different parts of the pipeline
   - They aim to balance accuracy, robustness, and domain adaptability

#### Applications and use cases

Natural language interfaces to data systems are transforming how organizations interact with their data across many domains:

###### Business intelligence

- Executives can ask direct questions about business performance
- Sales teams can query CRM data without technical assistance
- Operations staff can access metrics through simple questions
- Finance teams can explore financial data through conversation

Conversational business intelligence systems are particularly valuable as they allow business users and analytics teams to quickly analyze data and understand reasons and key drivers for business behaviors through natural dialogue.

###### Data science and analytics

- Simplifies exploratory data analysis
- Enables quick hypothesis testing through natural questions
- Democratizes access to analytics capabilities
- Accelerates the data-to-insight pipeline

###### Enterprise information systems

- Provides unified access to siloed data sources
- Enables cross-departmental data exploration
- Reduces dependency on IT for data access
- Accelerates decision-making with timely insights

###### Challenges and limitations

Despite their potential, natural language interfaces for data systems face several challenges:

###### Ambiguity and context

Natural language is inherently ambiguous. When a user asks, “How are sales this year?”, this could refer to:

- Total sales vs. last year
- Sales by product category
- Sales by region
- Monthly sales trends

The inherent characteristics of natural language queries, such as ambiguity in terms of intent and entities, implied query context, linguistic variations, and incomplete queries, make query understanding and interpretation difficult.

###### Schema understanding

The system must map natural language terms to the correct database entities:

- Different databases use different naming conventions
- The same term may have different meanings in different contexts
- Not all database structures are intuitive to non-technical users

Different domains, such as finance and healthcare, have their own unique characteristics and vocabulary. An effective NLI solution should not only understand the semantics of a particular domain but also be adaptable across different domains.

###### Query complexity

While simple queries are handled well, more complex analytical needs present challenges:

- Nested conditions and multi-table joins
- Window functions and advanced analytics
- Temporal and geospatial operations
- Complex aggregations

Detecting whether a natural language query requires translation to a nested structured query is non-trivial due to non-obvious linguistic patterns and inherent ambiguities. Building a nested query requires identifying proper sub-queries and figuring out the correct conditions to join or combine them.

###### Data security and governance

Natural language interfaces must respect security boundaries:

- User access permissions
- Data privacy regulations
- Sensitive data handling
- Audit and compliance requirements

###### Recent advances and benchmarks

Several benchmarks have been developed to evaluate natural language interfaces to data:

- WikiSQL: Contains pairs of natural language questions and SQL queries distributed across Wikipedia tables
- Spider: A cross-domain dataset with complex SQL queries involving joins and nested queries
- SParC: A context-dependent, multi-turn version allowing follow-up questions
- CoSQL: A dialogue version that simulates real database querying scenarios

These benchmarks have driven progress in the field, with recent systems achieving increasingly higher accuracy on complex queries across multiple domains.

#### The future of natural language interfaces for data

As this technology continues to evolve, several trends are emerging:

##### Multimodal interactions

Future systems will integrate:

- Natural language with visual interfaces
- Voice and text input methods
- Gesture-based data exploration
- Collaborative data analysis environments

##### Autonomous data exploration

Advanced systems will proactively:

- Suggest relevant analyses
- Identify anomalies and patterns
- Alert users to significant changes
- Generate insights without explicit queries

##### Explainable AI integration

As queries become more complex, systems will:

- Explain how they interpreted the question
- Show the reasoning behind their analysis
- Provide transparency in data transformations
- Build trust through clear explanations



## Generative vs. Agentic

Generative AI are really fundamentally reactive systems. They wait for you to prompt them to generate text, image, code, audio etc. They're using patterns they learned during training. Generative AI are essentially sophisticated pattern matching machines. They've learned the statistical relationships between words and between pixels and between sound waves. So when you provide a prompt, gen AI predicts what should come next based on its training. But its work does end at generation. It doesn't take further steps without your input. 

Agentic AI systems, by contrast, are not reactive. They are proactive systems. Like generative AI, they often start with a user prompt. But that prompt is then used to pursue goals through a **series of actions**. LLMs serve as the backbone for chatbots. LLMs also provide the reasoning engine that powers agentic systems. 

So consider a personal shopping agent. Given a product to purchase as input, it actively hunts for availability across platforms. It might monitor price fluctuations, it might handle checkout processes, and it might even coordinate delivery. Largely by itself, seeking input only from you, only when it's needed. But how does it do that? Well it turns out that the LLMs can also be used to provide reasoning capabilities to AI agents. It's called **chain of thought** reasoning. And this is what LLMs are so very good at. It's a process where the agent basically breaks down a complex task into smaller logical steps.  Looking ahead, the most powerful AI systems are going to be intelligent collaborators. They'll understand when to explore options through generation and when to commit to causes of action through agentic action. 


### Introduction: The Shifting Landscape of AI

At first, large language models (LLMs) could only generate text based on prompts. When these models gained the ability to use tools, call functions, and access memory, they started to act more like agents. AI agents are like digital assistants designed to carry out specific tasks—such as answering questions, organizing data, or summarizing content. As demands grew more complex, a new approach emerged: Agentic AI.

Agentic AI refers to systems made up of multiple AI agents that work together. Instead of just reacting to one command, they can:

- Break down big goals into smaller tasks.
- Adapt to new inputs or situations.
- Communicate and coordinate with one another.

In short, Agentic AI moves beyond one-task agents and enables teams of AI working collaboratively to achieve more complex goals.

##### What are AI Agents?

AI Agents can be characterized as autonomous software entities designed for goal-directed task execution within specific digital environments. Their operation typically involves three capabilities:

- **Autonomy**: The ability to function with minimal human intervention after initial deployment, capable of perceiving environmental inputs, reasoning over contextual data, and executing actions in real-time.
- **Task-Specificity**: Each agent is optimized for narrow, well-defined tasks such as email filtering, database querying, and so on.
- **Reactivity**: Agents respond to inputs from users, APIs, or other software environments in real time.


##### What Is Agentic AI?

Agentic AI takes things a step further. Instead of just one agent doing one job, Agentic AI brings multiple agents together into a team. These agents coordinate tasks, exchange information, adapt roles dynamically, and share memory. Key features include:

- Task Decomposition: Goals are split into subtasks automatically.
- Inter-Agent Communication: Agents share updates and results via messaging or shared memory.
- Memory and Reflection: Agents remember past steps and learn from outcomes.
- Orchestration: A lead agent or system coordinates the team.

Example: Planning a vacation—one agent books the flight, another finds hotels and a third checks visa requirements. A coordinator agent makes sure everything matches your preferences.

The following illustration contrasts a standalone AI Agent with a collaborative Agentic AI system in a smart home scenario.

##### Smart home AI comparison

<p align="center">
  <img src="./assets/agentic_ai/agentic1.png" alt="drawing" width="600" height="400" style="center" />
</p>

[AI Agents vs. Agentic AI: A Conceptual Taxonomy, Applications and Challenges](https://arxiv.org/pdf/2505.10468)

##### Key Architectural Differences between an AI Agent and Agentic AI

A structured taxonomy helps clarify the differences:

- AI Agent: Single entity, handles one task, uses tools, operates in narrow contexts
- Agentic AI: Multi-agent systems, goal-driven orchestration, adapts across time and context, supports parallel task execution

Feature	| AI Agent	| Agentic AI
------| ----------| ------------
Design	| One agent, one task	| Multiple agents with distinct roles
Communication	| No coordination with others	|Constant communication and coordination
Memory	| Stateless or minimal history |	Persistent memory of tasks, outcomes, and strategies
Reasoning	| Linear logic (do step A → B)	|Iterative planning and re-planning with advanced reasoning
Scalability	| Limited to task size	| Can scale to handle multi-agent, multi-stage problems
Typical Applications	| Chatbots, virtual assistants, workflow helpers	| Supply chain coordination, enterprise optimization, virtual team leaders

The architectural diagram below illustrates the transition from traditional AI Agent design (perception → reasoning → action) to Agentic AI systems built around collaboration, planning, and memory.

<p align="center">
  <img src="./assets/agentic_ai/agentic2.png" alt="drawing" width="600" height="400" style="center" />
</p>

[AI Agents vs. Agentic AI: A Conceptual Taxonomy, Applications and Challenges](https://arxiv.org/pdf/2505.10468)

The transition from AI Agents to Agentic AI involves several fundamental architectural enhancements:

###### From Single to Multiple Agents:

- Rather than operating as single units, Agentic AI systems consist of multiple agents, each assigned specialized functions or tasks (such as summarization, retrieval, or planning).
- These agents interact via communication channels like message queues, blackboards, or shared memory.

###### Advanced Reasoning Capabilities

Agentic systems integrate iterative reasoning capabilities using frameworks such as "ReAct (Reasoning and Acting)", "Chain-of-Thought" prompting, and "Tree of Thoughts." These mechanisms allow agents to break down complex tasks into multiple reasoning stages, evaluate intermediate results, and re-plan actions dynamically.

###### Persistent Memory Systems

- Unlike traditional agents, Agentic AI incorporates memory subsystems to preserve and persist knowledge across task cycles or agent sessions.
- Memory types include **episodic memory** (task-specific history), **semantic memory** (long-term facts or structured data), and **vector-based memory** for retrieval-augmented generation.

###### Real-World Applications

AI Agents excel in domains such as:

- Customer support (e.g., chatbots)
- Internal enterprise search
- Email filtering and prioritization

Agentic AI enables:

- Multi-agent research assistants
- Robotic coordination (e.g., drones)
- Collaborative medical decision systems
- Adaptive workflow automation

##### Current Challenges

###### Limitations of AI Agents

AI Agents face significant challenges, including lack of causal understanding, inherited limitations from LLMs such as hallucinations and prompt sensitivity, incomplete agentic properties, and failures in long-horizon planning and recovery.

###### Agentic AI Complexities

Agentic AI systems introduce amplified challenges, including **inter-agent error cascades**, **coordination breakdowns**, **emergent instability**, **scalability limits**, and **explainability issues** stemming from the complexity of orchestrating multiple agents across distributed tasks.

#### The Path Forward: Emerging Solutions

The field is actively developing solutions to address these limitations:

##### Retrieval-Augmented Generation (RAG)

For AI Agents, RAG mitigates hallucinations and expands static LLM knowledge by grounding outputs in real-time data. In Agentic AI systems, RAG serves as a shared grounding mechanism across agents, allowing distributed agents to operate on a unified semantic layer.

##### Tool-Augmented Reasoning

AI Agents benefit from function/tool calling, which extends their ability to interact with real-world systems. For Agentic AI, function calling is instrumental in enhancing both autonomy and structured coordination among multiple agents through orchestrated pipelines.

##### Memory Architectures

Advanced memory systems address limitations in long-horizon planning by persisting information across tasks. Episodic memory allows agents to recall prior actions and feedback, semantic memory encodes structured domain knowledge, and vector memory enables similarity-based retrieval.

#### Looking Ahead: The Future of AI Agents and Agentic AI

###### AI Agents Evolution

AI agents are becoming smarter, so they will soon:

- Proactively recommend actions (not just react).
- Learn from interactions over time.
- Reason about more complex logic and decisions.

###### Agentic AI Advancement

It is expected that in the near future there will be:

- Teams of agents managing real-world workflows.
- Better simulations for testing multi-agent plans.
- Governance systems to ensure ethical coordination.

The diagram below outlines the anticipated evolution of AI Agents and Agentic AI systems across multiple dimensions.

<p align="center">
  <img src="./assets/agentic_ai/agentic3.png" alt="drawing" width="600" height="400" style="center" />
</p>

[AI Agents vs. Agentic AI: A Conceptual Taxonomy, Applications and Challenges](https://arxiv.org/pdf/2505.10468)

#### Building Agentic AI in Practice: Tools and Frameworks

While the concept of Agentic AI is still evolving, developers and researchers are already prototyping these systems using the following emerging AI orchestration frameworks:

- LangChain
A Python framework for building applications around LLMs. It supports tool usage, memory, chains of reasoning, and agent interfaces. LangChain provides the building blocks to combine language models with external data and APIs.

- LangGraph
A framework for building multi-agent workflows using a graph-based execution model. It allows you to define agents as nodes and their interactions as edges, ideal for orchestrating collaborative agents in Agentic AI.

- IBM Bee, CrewAI, AutoGen, and others
These open-source tools simplify the design of multi-agent teams, role assignment, and structured task planning. They allow developers to simulate or deploy collaborative agent environments using memory, messaging, and dynamic delegation.

These frameworks allow developers to begin experimenting with Agentic AI by leveraging reusable modules for planning, reflection, and inter-agent communication.

The distinction between AI Agents and Agentic AI is not static. These definitions continue to evolve as architectures, tools, and expectations shift. The future likely involves hybrid frameworks—blending the simplicity of task-specific agents with the flexibility and intelligence of multi-agent orchestration.

Agentic AI represents the next step in scalable, intelligent systems capable of real-world impact across domains such as robotics, healthcare, logistics, and beyond.

### LangGraph Architecture: Designing Effective Workflows

Now that you've learned the basics of LangGraph—nodes, edges, and persistent state—this reading explores architectural principles for building clear and effective workflows.

#### Why Use Graph Architecture?

Traditional loops and conditional statements quickly become limiting when building complex AI workflows. LangGraph provides:

- **Dynamic Decision-Making**: Workflow paths can branch based on runtime conditions.
- **Clear Visualization**: Easy-to-understand diagrams (such as Mermaid diagrams) that simplify debugging.
- **Reusable Components**: Modular nodes that perform specific tasks and can be independently developed and tested.

Imagine creating a customer support agent:

- Traditional loops handle only simple repetitive checks.
- LangGraph allows branching, loops, and pausing for human interaction, all while maintaining context.

##### State Design Best Practices

State holds the workflow's context and shared data.

Key design principles include:
- Clear Naming: Use descriptive names like user_query or agent_response.
- Flat Structures: Avoid deeply nested states for easier manipulation.

##### Node Design Principles

Each node should perform a single, clear task:

- Processing Nodes: Perform data transformation or computation.
- Validation Nodes: Check conditions or data integrity.
- Integration Nodes: Interface with external systems (APIs, databases).
- Decision Nodes: Direct workflow paths based on conditions.

Nodes communicate through state:
- Read necessary inputs from state.
- Perform the task.
- Update the state accordingly.

##### Edge and Workflow Patterns

Edges control execution flow between nodes. Common patterns include:

Simple Conditional Logic:
```python
def route_decision(state):
    if state["retry_count"] > 2:
        return "human_review"
    elif state["issue_type"] == "resolved":
        return "end_interaction"
    else:
        return "continue_processing"
```

##### Error Handling Strategies

Always plan for errors:
- Include error-specific state fields.
- Create dedicated error-handling nodes.
- Implement graceful fallbacks.

Common strategies:

- Retry Nodes: Attempt an action again.
- Error Nodes: Route to human intervention or logging systems after repeated failures.

##### Testing and Debugging

Maintain testable and debuggable workflows:

- Node Isolation: Test nodes individually.
- Predictable States: Same inputs should produce the same outputs.
- Incremental Development: Add and verify nodes step-by-step.

##### Performance Considerations

Ensure efficient workflow execution:

- Minimize state complexity.
- Isolate costly computations in specific nodes.
- Use caching for repeated expensive operations.

##### Integration Tips

Connecting external systems:

- Separate integration logic clearly.
- Anticipate failures with timeouts and fallback mechanisms.

Incorporating humans effectively:

- Pause workflows for approvals or reviews clearly.
- Provide straightforward paths for human decisions.

##### Common Mistakes to Avoid

Avoid:

- Oversized nodes handling multiple tasks.
- Deeply nested or unclear states.
- Ignoring error conditions.

Instead:

- Use modular nodes with clear responsibilities.
- Explicitly define state schemas.
- Plan clearly for error handling early in design.

##### Example Workflow

Document processing scenario:

- Validate uploaded document.
- Extract text.
- Analyze content.
- Generate summary.
- State Schema example:

```python
from typing import TypedDict
class DocumentProcessingState(TypedDict):
    file_path: str
    text_content: str
    summary: str
    analysis_results: dict
```

Conclusion

Effective LangGraph architecture focuses on simplicity, clarity, and modularity:

- Begin simply and incrementally add complexity.
- Keep state structures explicit and manageable.
- Design independent, clearly defined nodes.
- Proactively handle potential errors.

Adopting these principles will help you create robust and maintainable LangGraph workflows tailored to your specific AI needs.

### LangChain vs LangGraph: Pros, Cons, and Practical Considerations

Recent developments in AI have produced powerful frameworks for building applications around large language models (LLMs). LangChain (released 2022) and LangGraph (released 2023) are two such frameworks from LangChain Inc.

They take different approaches:

- LangChain uses a linear "chain" of components (prompts, models, tools).
- LangGraph uses a graph-based orchestration of stateful, multi-agent workflows.

In practice, LangChain is ideal for straightforward, sequential tasks (for example, a simple QA or RAG pipeline), whereas LangGraph is designed for complex, adaptive systems (for example, coordinating multiple AI agents, maintaining long-term context, or human-in-the-loop approval).

##### What is LangChain?

LangChain is an open-source framework for developing LLM-driven applications. It provides tools and APIs (in Python and JavaScript) to simplify building chatbots, virtual assistants, Retrieval-Augmented Generation (RAG) pipelines, and other LLM-based workflows. At its core, LangChain uses a "chain" or directed graph structure: you define a sequence of steps (prompts → model calls → outputs) that execute in order.

For example, a RAG workflow might chain: (1) retrieve relevant documents, (2) summarize, (3) generate an answer. The LangChain ecosystem includes prebuilt components for prompts, memory buffers, tools (such as search or calculator), and agents that can pick actions. It also integrates with dozens of LLM providers. LangChain's flexible, modular design has made it popular for quickly prototyping AI apps with minimal coding.

The LangChain framework is layered: a core of abstractions (chat models, vectors, tools), surrounded by higher-level chains and agents that implement workflows.

LangChain's architecture is modular. The base langchain-core defines interfaces for models, prompts, memory and tools. The main langchain package adds chains and agents that form the "cognitive architecture". Popular LLM providers (OpenAI, Google, etc.) have their own integration packages, making it easy to switch models. There is also a langchain-community package for third-party extensions. Overall, LangChain serves as a high-level orchestration layer for LLMs and data. It handles inputs/outputs and connects components, but it remains mostly stateless by default (conversation histories can be passed, but long-term memory must be explicitly managed).

##### What is LangGraph?

LangGraph is a newer framework (built by the same team) focused on stateful multi-agent orchestration. LangGraph is an extension of LangChain aimed at building robust and stateful multi-actor applications with LLMs by modeling steps as edges and nodes. In simple terms, while LangChain chains operations, LangGraph lets you build a graph of agents and tools. Each node in the graph can be an LLM call or an agent (with its own prompt, model, and tools), and edges define how data and control flow between them. This architecture natively supports loops, branches, and complex control flows.

LangGraph is explicitly designed for long-running, complex workflows. Its core benefits include durable execution (agents can pause and resume after failures), explicit human-in-the-loop controls, and persistent memory across sessions. For instance, LangGraph provides "comprehensive memory" that records both short-term reasoning steps and long-term facts. It even offers built-in inspection and rollback features so developers can "time-travel" through an agent's state to debug or adjust its course. In practice, LangGraph is used to build applications where many agents work together – for example, one agent might retrieve documents, another summarizes them, a third plans next steps, etc. These agents communicate using shared "state" (a kind of global memory) and can be arranged hierarchically or in parallel. Major companies (LinkedIn, Uber, Klarna, and so on) have begun using LangGraph to build sophisticated agentic applications in production.

##### Key Architectural Differences

Feature	| LangChain	| LangGraph
 -------- | ------------- | -------------
Type	| LLM orchestration framework based on chains and agents. |	AI agent orchestration framework based on stateful graphs.
Workflow Structure	| Linear/DAG workflows (sequence of steps with no cycles). Good for "prompt → model → output” flows	|Graph-based workflows (nodes and edges allow loops, branches, and dynamic transitions). Suited for complex flows.
State Management	| Implicit/pass-through data. Chains carry inputs forward, but long-term state is limited by default	|Explicit global state ("memory bank") that all agents access. State is persistently stored and updated at each step.
Task Complexity	| Best for simple to medium tasks: chatbots, RAG pipelines, sequential reasoning. |	Designed for complex, multi-step tasks and workflows that evolve over time (for example, multi-agent assistants).
Agents and Collaboration	| Typically single-agent or linear chain; agents operate independently without inter-communication.	|Multi-agent. Agents (nodes) can call each other using the graph, share memory, or be arranged hierarchically.

The above table summarizes the LangChain vs. LangGraph trade-offs.

##### Pros and Cons

Framework	| Pros	| Cons
 --------- | :---------- | ---------- 
 LangChain |  • Easy and quick to set up for common LLM tasks <br> • Extensive community and prebuilt components (for example, QA chains, map-reduce, memory buffers) <br> • Excellent for RAG workflows and chatbots <br> • Implicit chaining model requires minimal boilerplate code	| • Not well-suited for long-running or highly interactive processes <br> • Lacks built-in persistent memory and multi-agent orchestration <br> • Workflows cannot natively loop or branch dynamically <br> • Debugging is harder due to opaque state passing between steps 
LangGraph	| • Built for complexity and scale <br> • Agents can run concurrently or sequentially with shared context <br> • Supports durable execution (resume from point of failure) <br> • Deep visibility into internal state and execution path (using LangSmith) <br> • Human-in-the-loop support is first class <br> • Ideal for orchestrating multi-step business processes	| • More complex to learn and set up <br> • Requires explicit definition of states, nodes, and edges <br> • Slower to develop simple use cases compared to LangChain <br> • Ecosystem is newer with fewer templates and extensions <br> • Overhead may be unnecessary for simple tasks

##### When to use which framework?

Use Case |	Use LangChain When…	| Use LangGraph When…
---------- | ------------ | ------------
Workflow Complexity	| You have a clearly-defined, linear workflow.	| You need complex workflows with branching logic or conditional steps.
Development Speed	| You want to build something quickly—ideal for prototyping and MVPs.	| You're building a production-grade system where reliability, traceability, and durability are essential.
Memory Requirements	| Stateless or light memory needs (for example, current conversation only). |	Long-term memory is needed across interactions or agents (for example, remembering context across sessions).
Interaction Style	| Simple LLM tool use (for example, retrieval, transformation, response).	| Multi-turn or human-in-the-loop interactions requiring persistent state and coordination.
System Design	| Linear pipelines such as document Q&A, summarization, or format conversion.	| Multi-agent architectures, process automation, or workflows with retries, dependencies, or approvals.
Team Collaboration |	Individual developer exploring LLM capabilities quickly.	| Teams designing modular, orchestrated systems with accountability and version control.

##### Conclusion

The LangChain and LangGraph frameworks represent two evolving approaches to building with LLMs. LangChain offers a simple, powerful abstraction for chaining prompts and tools in sequence, while LangGraph offers a flexible, stateful architecture for orchestrating complex agent workflows. Developers should choose between them based on the complexity and requirements of their project: use LangChain for straightforward pipelines and experimentation, and adopt LangGraph when you need durable, multi-agent orchestration and fine-grained control. As both frameworks grow, they will likely continue to influence each other. LangChain is already integrating more stateful features (for example, LangGraph memory), and LangGraph can leverage LangChain's components. The right choice depends on the use case at hand, and understanding the trade-offs above will help you select the best tool for your LLM application.


##### Note
LangChain is deprecating its legacy agent framework in favor of LangGraph, which offers enhanced flexibility and control for building intelligent agents. In LangGraph, the graph manages the agent's iterative cycles and tracks the scratchpad as messages within its state. The LangChain "agent" corresponds to the prompt and LLM setup you provide. Refer to LangChain and LangGraph's latest documentation for latest updates.


### Introduction to LangGraph

Getting started with LangGraph

LangGraph |  |
----- | -----
Overview	| LangGraph is an open-source (MIT-licensed) framework for building stateful, graph-based AI agents.
Extension of LangChain |	It builds on LangChain by enabling workflows as graphs of nodes, with explicit control flow and state management.
State management	| A central state object (typically a TypedDict or Pydantic model) is passed between nodes, each of which updates and processes that state.
Workflow capabilities |	Supports branching, looping, memory retention, and conditional logic—beyond what a simple, linear LangChain chain can offer.
Advanced behaviors	| Enables complex agent behaviors such as iterative reasoning, conditional paths, and human-in-the-loop interactions.
Execution features |	Workflows can run over time (durable execution), support human inspection of state, and leverage both short- and long-term memory for decisions.
Ecosystem integration |	Interoperable with the full LangChain ecosystem, including tools, chains, memory components, and LangSmith for observability and debugging.

##### Why graph-based agents?

Traditional LangChain chains are Directed Acyclic Graphs (DAGs). They define a fixed, linear sequence of LLM calls and tool invocations. These chains are suitable for simple, one-pass tasks but lack support for branching or looping.

LangGraph agents operate as **state machines**. They allow the system to revisit steps, make decisions conditionally, and model complex flows like loops, retries, and branching paths.

In a traditional chain, retrieval runs once—if the result is poor, the system is stuck. With LangGraph, the LLM can loop: it can revise the query, retrieve it again, and continue, enabling adaptive behavior.

##### When to use LangGraph

LangGraph is ideal for complex agent workflows that need explicit state and flexible control flow. Use it when your task involves:

Concept	| Explanation
------ | ----
Loops or iteration	| Tasks where the agent might try an action, check results, and repeat until a goal is achieved. (for example, iterative refinement of a query or planning steps.)
Conditional branching	| Workflows with if/else logic. For instance, a support bot that asks follow-up questions based on user replies.
Long-running processes	| Scenarios where the agent must persist state and resume after delays or failures (LangGraph supports durable execution and checkpointing).
Complex state management	| When many variables or data points must be carried through the workflow, LangGraph’s shared state object is more explicit than passing context through nested chains.
Multi-agent or multi-step coordination	| You can design graphs where different nodes represent different agents or tools working together, with the central state tracking their interactions.

#### Core concepts of LangGraph

##### State	
State is the shared, central piece of data that flows through your LangGraph workflow. Think of it as a dictionary (or, more formally, a `TypedDict` or `Pydantic` model) that carries all relevant information from one node to the next. Each node in the graph reads from and updates this state object. 

```python
from typing import TypedDict

class WorkflowState(TypedDict):
    user_query: str
    summary: str
    step_count: int
```

Initialize a state field with an initial value (e.g., `{"user_query": "Hello", "summary": "", "step_count": 0}`) when invoking the graph.


##### StateGraph
StateGraph is the controller or blueprint of the workflow. This class lets you define: 
- What nodes exist 
- How they connect (edges) 
- Where the workflow starts and ends 
-  When to loop or branch conditionally

In other words, `State` is the data that flows through the system (changes during execution) but a `StateGraph` is the structure that defines how that data moves and gets transformed (fixed once compiled). You create a `StateGraph` by passing the state schema type: 
```python
from langgraph.graph import StateGraph
graph = StateGraph(WorkflowState)
```

##### Nodes
Each node is a Python function (or LangChain Runnable) that takes the state dict and returns an updated state. Nodes perform actions such as calling an LLM, running a tool, computing something, etc.  

```python
def summarize(state: WorkflowState) -> WorkflowState:
    text = state["user_query"]
    state["summary"] = llm_summarize(text)  # some LLM call
    return state
```

You can add nodes to the graph using `graph.add_node()`. Each node should update the state and return it. LangGraph can also use LangChain chains or agents as nodes (they must conform to the same state signature).

##### Edges
Edges define how the workflow moves from one node to the next. 

- Linear (normal) edges: Use `graph.add_edge(from_node, to_node)` to always flow from one node to the next. You must specify an entry point and exit using the special `START` and `END` tokens from `langgraph.graph`. 
  ```python
  from langgraph.graph import START, END

  graph.add_edge(START, "summarize")
  graph.add_edge("summarize", "finalize")
  graph.add_edge("finalize", END)
  ```
  Here, we add the edges from `START` to `summarize`, indicating that the graph workflow will begin from `summarize`. After that, we have two other edges, one from `summarize` to `finalize` and another from `finalize` to `END` indicating the end of the workflow. 

- Conditional edges: Use `graph.add_conditional_edges(from_node, decision_func, mapping)` to branch. The `decision_func(state)` should return a string key; then, the workflow moves to whichever node name that key maps to. 
  ```python
  def decide(state: WorkflowState) -> str:
      return "repeat" if state["step_count"] < 2 else "done"

  graph.add_conditional_edges(
      "summarize",
      decide,
      {"repeat": "summarize", "done": END}
  )
  ```  
  In this case, after `summarize` is executed, `decide()` function checks `step_count`. If it returns "repeat", the graph loops back to the `summarize` node again; if "done", it goes to the special END and stops. Conditional edges let LLM-driven or logic-driven functions choose the next step dynamically.

- Compile and run: Once all nodes and edges are added, call `runnable = graph.compile()`. This produces a Runnable object (just like a LangChain Runnable) that you can run with `.invoke(initial_state)` or `.stream(initial_state)`. 
  ```python
  runnable = graph.compile()
  final_state = runnable.invoke({"user_query": "Hello", "summary": "", "step_count": 0})
  ```
  
  This executes the graph: it starts at START, follows edges (running each node's function on the state), and stops at END. The final state dict contains all updates. The compiled graph supports all usual LangChain methods (`.stream()`, `async variants`, `batching`, etc.)

- Visualizing your graph with a Mermaid diagram:	 LangGraph supports generating Mermaid diagrams, a lightweight syntax for rendering flowcharts and state diagrams. This helps you visually understand how your agent moves from one node to another, especially when there are loops or conditional branches. Once your graph is built, you can render a Mermaid diagram using: `print(app.get_graph().draw_mermaid())`


#### A LangGraph Example

In this example, let's build an increment counter using LangGraph.

###### Define the state schema	
We start with defining the State Schema with a `TypedDict` (or `Pydantic` model) listing all fields your workflow needs.  
```python
class GraphState(TypedDict):
    count: int
    message: str
```

This says our state has an integer `count` and a string `message`.

###### Initialize the StateGraph	
 ```python
 from langgraph.graph import StateGraph
 graph = StateGraph(GraphState)
 ```

###### Add nodes 
For each step, write a function that takes and returns the state. Then register it with `add_node()`. 
```python
def increment(state: GraphState) -> GraphState:
    state["count"] += 1
    state["message"] = f"Count is now {state['count']}"
    return state

graph.add_node("increment", increment)
```

You can add as many nodes as needed, possibly using the same function multiple times under different names.

###### Connect edges
Define the flow of execution. At minimum, set a start edge from `START`, and usually end at `END`. For linear flow:

```python
from langgraph.graph import START, END

graph.add_edge(START, "increment")
graph.add_edge("increment", END)
```

```python
# Add nodes
def increment(state: GraphState) -> GraphState:
    state["count"] += 1
    state["message"] = f"Count is now {state['count']}"
    return state
graph.add_node("increment", increment)
```

###### Conditional branching (optional)	
To loop or branch, use `add_conditional_edge()`. For example, to repeat the “increment” node until the count reaches 3:

```python
def decide_next(state: GraphState) -> str:
    return "again" if state["count"] < 3 else "finish"

graph.add_conditional_edges("increment", decide_next, {"again": "increment", "finish": END})
```                                      

Now, after each "increment", the graph checks the returned key: if "again", it loops back to "increment" (making a cycle); if "finish", it goes to END. This simple loop will run the increment node three times.

###### Compile and invoke	
Finally, compile the graph and run it:
```python
app = graph.compile()
result = app.invoke({"count": 0, "message": ""})
```

Here, the initial state has count=0. After invoking, the result contains the updated state (e.g., count = 3 if we looped three times).


### Structuring LLM Tool Calls with Pydantic and JSON Serialization

We know that LLMs can output text, and when we bind tools to our LLM, we can extract parameters and call functions to process the output or perform specific tasks. A schema or data model is a code-level extension of this idea—it allows you to enforce that the LLM outputs data in a specific format, such as a Python class, dictionary, or JSON. This ensures that if you need to feed the LLM output into an API, database, or another function, the output is **structured**, **predictable**, and **integrates seamlessly with other systems**.

Consider a weather API example: it might expect data such as weather conditions ("sunny", "rainy", "cloudy"), temperature as an integer, and the temperature unit ("celsius" or "fahrenheit").

For this, you could create a Pydantic class like this:

```python
from pydantic import BaseModel, Field

class WeatherSchema(BaseModel):
    condition: str = Field(description="Weather condition such as sunny, rainy, cloudy")
    temperature: int = Field(description="Temperature value")
    unit: str = Field(description="Temperature unit such as fahrenheit or celsius")
```

You can then bind this schema as a tool to your LLM and call the LLM as follows:

```python
from langchain_openai import ChatOpenAI
# Create an LLM instance
llm = ChatOpenAI(model="gpt-4.1-nano")  # or your preferred model
weather_llm = llm.bind_tools(tools=[WeatherSchema])
response = weather_llm.invoke("It's sunny and 75 degrees")
# Returns: {"condition": "sunny", "temperature": 75, "unit": "fahrenheit"}
```

In the output, the response of one of the attributes will be a dictionary of key-value pairs the weather API will use. The schema is one of many attributes in the `AIMessage`. Let's better understand it by extracting it from a LLM response.

##### Real Example: Addition Tool with Pydantic and LangChain

In a real-world example, we might ask a language model to book a flight, where the schema could include fields like destination, starting point, time, and date—structured data that would then be passed to an API. In this simplified example, we define a Pydantic schema `Add` to describe the expected input structure (two integers) for an addition task and use it as a tool for a language model (ChatOpenAI) to extract structured data from a natural language query. When a user says something like "add 1 and 10", the LLM interprets the request using the `Add` schema, extracts the numbers, and the code performs the actual addition and prints the result.

```python
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
# Define BaseModel class for addition
class Add(BaseModel):
    """Add two numbers together"""
    a: int = Field(description="First number")
    b: int = Field(description="Second number")
# Setup LLM and bind the Add tool
llm = ChatOpenAI(model="gpt-4.1-nano")
initial_chain = llm.bind_tools(tools=[Add])
# Ask LLM to add numbers
question = "add 1 and 10"
response = initial_chain.invoke([HumanMessage(content=question)])
# Extract and calculate from the LLM response
def extract_and_add(response):
    tool_call = response.tool_calls[0]
    a = tool_call["args"]['a']
    b = tool_call["args"]['b']
    return a + b
# Execute and print results
result = extract_and_add(response)
print(f"LLM extracted: a={response.tool_calls[0]['args']['a']}, b={response.tool_calls[0]['args']['b']}")
print(f"Result: {result}")
```

##### Why Use Pydantic Models for LLM Tool Calls?

In applications where LLMs call external tools or functions—such as in agentic systems or tool-augmented reasoning—it is critical that both inputs and outputs are:

- Structured
- Validated
- Easily serialized to and from JSON

Pydantic lets you define Python classes with strict type validation and built-in JSON serialization, which helps ensure reliable data exchange between your LLM and other systems.

###### Example: Defining Reusable Math Tool Schemas

```python
from pydantic import BaseModel
from typing import Literal
class TwoOperands(BaseModel):
    a: float
    b: float
class AddInput(TwoOperands):
    operation: Literal['add']
class SubtractInput(TwoOperands):
    operation: Literal['subtract']
class MathOutput(BaseModel):
    result: float
```

Tool Functions Using Pydantic Models

```python
def add_tool(data: AddInput) -> MathOutput:
    return MathOutput(result=data.a + data.b)
def subtract_tool(data: SubtractInput) -> MathOutput:
    return MathOutput(result=data.a - data.b)
```

###### Dispatching Tool Calls from JSON Input

```python
incoming_json = '{"a": 7, "b": 3, "operation": "subtract"}'
def dispatch_tool(json_payload: str) -> str:
    base = SubtractInput.parse_raw(json_payload)
    if base.operation == "add":
        output = add_tool(AddInput.parse_raw(json_payload))
    elif base.operation == "subtract":
        output = subtract_tool(SubtractInput.parse_raw(json_payload))
    else:
        raise ValueError("Unsupported operation")
    return output.json()
result_json = dispatch_tool(incoming_json)
print(result_json)  # {"result": 4.0}
```

##### What Does Literal Do?

The Literal type from Python's typing module restricts a variable to one or more specific constant values. In the examples above, it ensures that the operation field only accepts `add` or `subtract`. This helps validate that the input operation matches a known tool and prevents invalid operations from reaching your system.

```python
from typing import Literal
# Define a schema with Literal to restrict operation types
class CalculatorSchema(BaseModel):
    operation: Literal['add', 'subtract', 'multiply', 'divide'] = Field(
        description="The mathematical operation to perform"
    )
    a: float = Field(description="First number")
    b: float = Field(description="Second number")
    
calculator_llm = llm.bind_tools(tools=[CalculatorSchema])
# Test with valid operations
response1 = calculator_llm.invoke("Add 15 and 23")
print(response1.tool_calls[0]['args'])
# Output: {"operation": "add", "a": 15.0, "b": 23.0}
response2 = calculator_llm.invoke("Multiply 7 by 8")
print(response2.tool_calls[0]['args'])
# Output: {"operation": "multiply", "a": 7.0, "b": 8.0}
```

##### Why JSON-Serializable Pydantic Models Are Powerful

Feature |	Benefit
-----| -------
Type Validation	| Ensures inputs and outputs conform to expected schema
Reusability	| Use base classes such as `TwoOperands` across multiple tools
JSON Serialization	| `.json()` and `.parse_raw()` simplify tool chaining and I/O
Extensibility	| Easily add more tools (e.g., multiply, divide)

##### Final Thoughts and Alternatives

Using Pydantic models to define JSON-serializable tool inputs and outputs makes your LLM applications:

- Robust and error-proof.
- Compatible with orchestration frameworks such as LangChain, CrewAI, Watsonx, and so on.
- Easy to test, maintain, and extend.

This forms the foundation of building reliable, multi-tool LLM agents.

###### Optional Note: Pydantic vs. Python Dataclasses

You don't need to use Pydantic exclusively. Since Python 3.7, dataclasses offer a lightweight alternative for defining data models with built-in parsing and serialization. However, Pydantic provides more advanced features such as:

- Validators for complex constraints
- Extra configuration options
- Better integration with libraries like LangChain and FastAPI

Pydantic is generally recommended due to its maturity and feature set, and it's the default choice in popular frameworks.







### Building Self-Improvement Agents with LangGraph

Modern agent architectures enable AI systems to critique and refine their own output for higher quality. These "self-improving" agents use loops where the agent reviews its work and acts on feedback. LangGraph—a graph-based framework for stateful LLM applications—makes it easy to implement these patterns.

At a high level, these can be categorized as three approaches: 
- Reflection agents, reflexion agents, and ReAct agents. Each uses a different strategy for self-improvement:

Agent	| Description
------ | ------
Reflection agents	| Prompts the model to review its own answer (like a teacher grading its work).
Reflexion agents	| Adds external feedback (search or tools) to guide corrections.
ReAct agents	| Alternate reasoning and actions, thinking and doing in a loop (tool calls, chain-of-thought).

LangGraph represents agents as graphs of states and nodes. The state (often a message history) flows through nodes (functions or LLM calls) linked by edges with conditional logic. Below, we explain each agent style, show sample LangGraph code, and give guidance on use cases.

##### Reflection agents

Reflection agents use internal critique to refine outputs. Conceptually, the agent first generates an initial answer, then a second step reflects on that answer. The reflector (often role-played as a teacher or critic) points out flaws or suggests improvements. The agent may loop this generate-then-reflect cycle a few times to polish the answer.

Concept	| Description
------ | --------
Mechanics	| Typically, one node calls the LLM to produce a response, and another node calls the LLM to critique or improve it. A simple LangGraph MessageGraph can model this two-step loop.
Example Code	| Below, generate_answer and critique_answer are two nodes. We loop between them until a max step count is reached. See the pseudocode here:

```python
from langgraph.graph import MessageGraph, END
from langchain_core.messages import HumanMessage, AIMessage
# Node that generates an initial response
def generate_answer(state):
    # (In practice, call an LLM here)
    answer = "This is my first attempt."
    return {"messages": state["messages"] + [AIMessage(content=answer)]}
# Node that critiques and refines the previous answer
def critique_answer(state):
    # (In practice, call LLM to critique)
    critique = "The answer is incomplete; add more detail."
    return {"messages": state["messages"] + [AIMessage(content=critique)]}
builder = MessageGraph()
builder.add_node("generate", generate_answer)
builder.add_node("reflect", critique_answer)
builder.set_entry_point("generate")
# Loop control: alternate until max iterations
MAX_STEPS = 3
def should_continue(state):
    return "reflect" if len(state["messages"]) < 2*MAX_STEPS else END
builder.add_conditional_edges("generate", should_continue)
builder.add_edge("reflect", "generate")
graph = builder.compile()
# Run the reflection agent
initial_message = HumanMessage(content="Explain photosynthesis.")
result = graph.invoke({"messages": [initial_message]})
print(result["messages"][-1])  # Final answer or critique
```

This makes the agent self-critique its answer. In practice, the reflector node is prompted to evaluate the generator's output and return suggestions. The loop continues until no more revisions are needed or a limit is reached.

- When to use: Reflection is useful for creative or open-ended tasks (e.g., drafting text, answering complex questions) where iterative refinement helps. It adds overhead (extra LLM calls) but often yields clearer, more thorough answers. However, since it only relies on the model's own reasoning (no outside data), the final answer may not improve much unless the reflector catches errors. Use Reflection when you want basic iterative self-improvement without adding external searches or tools.

##### Reflexion agents

Reflexion agents formalize the idea of reflection with external grounding. Here the agent not only critiques its output, but also uses external information or citations to do so. Each cycle typically involves three steps:

Step	| Description
------ | -----
Draft (initial response)	| The agent generates an answer and may propose search queries (or tool calls) to gather facts.
Execute tools	| These queries are run (for example, web search) and results are added to the context.
Revise	| A “revisor” node has the agent analyze the draft answer plus fetched info, and explicitly list missing or incorrect parts.

Reflexion forces the agent to cite sources and enumerate what's missing, making corrections more effective. In LangGraph, we chain three nodes in a loop (Draft → Execute Tools → Revise) until no further revisions are needed or a maximum iteration.


Concept	| Description
------- | ---------
Mechanics |	Each iteration adds more grounding. For example, after the draft answer, the agent might search Wikipedia, then the revise step reads the search results and updates the answer. The revised answer goes back into the loop if needed.
Workflow | code	Below is a pseudocode of a Reflexion-style loop. (`tool_search` is a stand-in for any external lookup.)


```python
from langgraph.graph import MessageGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
def draft_answer(state):
    # (LLM draft; could also generate search query)
    response = "The capital of France is Paris."
    return {"messages": state["messages"] + [AIMessage(content=response)]}
def execute_tools(state):
    # (Simulate external info; e.g., search results)
    info = "París (France) - capital: Paris (en.wikipedia.org)"
    return {"messages": state["messages"] + [SystemMessage(content=info)]}
def revise_answer(state):
    # (LLM re-evaluates answer using info)
    revision = "Yes, France’s capital is Paris. I've verified this."
    return {"messages": state["messages"] + [AIMessage(content=revision)]}
builder = MessageGraph()
builder.add_node("draft", draft_answer)
builder.add_node("execute_tools", execute_tools)
builder.add_node("revise", revise_answer)
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")
# Loop control: stop after N iterations
MAX_LOOPS = 2
def continue_reflexion(state):
    # Count assistant messages to determine iteration
    iteration = sum(1 for m in state["messages"] if isinstance(m, AIMessage))
    return "execute_tools" if iteration <= MAX_LOOPS else END
builder.add_conditional_edges("revise", continue_reflexion)
builder.set_entry_point("draft")
graph = builder.compile()
initial_message = HumanMessage(content="What is the capital of France?")
result = graph.invoke({"messages": [initial_message]}) # Final revised answer
```
This agent uses a **built-in search or tool** (`execute_tools`) to ground its critique. The revise node then updates the answer explicitly (e.g., adding evidence or fixing errors). The process stops when the agent judges the answer is good or after a set number of loops.
- **When to use**: Reflexion is ideal when accuracy or factual grounding matters. Because it enforces evidence (citations) and points out missing info, it shines on fact-checking, research, or coding tasks where correctness is critical. It is more complex and slower (requires search/tool calls), but yields highly vetted answers. Use Reflexion Agents for tasks like data lookup, code generation with static analysis, or any QA requiring references.


##### ReAct agents

ReAct (Reason + Act) agents interleave thinking and action. Rather than a separate “reflector” step, a ReAct agent alternates between internal reasoning (chain-of-thought) and taking actions (tool calls, function calls) in one workflow. Each cycle, the agent decides what to do, does it, then reasons again on the updated state.

Workflow of a ReAct agent

Concept	| Description
----------- | ------------
Mechanics	| The agent first uses the LLM to reason or plan (e.g., “I will search for the capital”). This might result in either a final answer or a tool request. If a tool call is needed, the agent calls it (e.g., a search API), adds the observation, and then thinks again with the new info. This continues until the agent outputs a final answer. The architecture is often: LLM node → Tool node → back to LLM, conditional on whether more tools are needed.
Example code	| Below is a simplified version for a weather agent (no actual API calls) showcasing ReAct (pseudocode). We define a StateGraph where the state includes a message history and logic flow:

```python
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
# Simple state with messages and a step counter
def call_model(state):
    # (LLM reasons; may request an action or give an answer)
    last = state["messages"][-1]
    if "weather" in last:
        # chain-of-thought leading to an action
        thought = AIMessage(content="Let me find the weather for you.")
        return {"messages": state["messages"] + [thought]}
    else:
        # final answer
        answer = AIMessage(content="It's sunny in NYC today.")
        return {"messages": state["messages"] + [answer]}
def call_tool(state):
    # (Simulate a weather API/tool result)
    tool_result = AIMessage(content="Weather(temperature=75F, condition=sunny)")
    return {"messages": state["messages"] + [tool_result]}
# Decide whether to act or finish based on last message
def next_step(state):
    last = state["messages"][-1]
    if "find the weather" in last:
        return "tools"
    return "end"
graph = StateGraph(dict)  # using a plain dict state
graph.add_node("think", call_model)
graph.add_node("act", call_tool)
graph.set_entry_point("think")
# If the model’s message triggers an action, go to 'act'; else end.
graph.add_conditional_edges("think", next_step, {"tools": "act", "end": END})
graph.add_edge("act", "think")
compiled = graph.compile()
result = compiled.invoke({"messages": [HumanMessage(content="What is the weather in NYC?")]})
print(result["messages"][-1])  # Final assistant answer
```

Here the agent thinks (calls the model) and acts (calls a tool) alternately. The next_step function checks the content of the last assistant message to decide. In practice, a ReAct agent's prompt would instruct the model to output either an action or the final answer, and LangGraph routes accordingly.

- When to use: ReAct is best for tasks that require tool use or complex planning, like interacting with APIs, databases, or multi-step reasoning. Because it weaves in actions dynamically, it can adapt to tasks (e.g., “Call calculator tool then interpret output”). It is simpler than Reflexion but more powerful than a basic chain-of-thought. Use ReAct agents when you need the model to reason and perform external actions in sequence. For quick setups, LangGraph even offers create_react_agent to instantiate a standard ReAct pattern with one call.

##### Comparison of agent styles

Aspect	| Reflection agent	| Reflexion agent	| ReAct agent
----- | ------ | ------- | ------
Core idea	| Model critiques its own answer	| Model critiques with external feedback and citations	| Model reasons and acts (calls tools) in loop
Structure |	Generator → Reflector → (loop)	| Draft  → (Search/Tool) → Revisor → (loop)	| LLM → (conditional Tool call) → LLM → …
Graph components	| 2 nodes (generate, reflect)	| 3+ nodes (draft, execute tools, revise)	| 2 nodes (think, act) with conditional branching
Feedback source |	Internal (LLM self-review) |	External (tool or search results + LLM review)	| External (tool calls informed by model reasoning)
Benefits	| Simple setup; improves coherence & detail	| High accuracy; enforces evidence and completeness	| Flexible tool use; handles complex tasks
Drawbacks	| May plateau (no new info); extra compute	| More complex and slow (searches/tools each loop)	| Requires designing tools; complexity in prompts
Use cases |	Refining essays, content drafts	| Fact-checking, coding, QA with citations	| Question answering with APIs, step-by-step tasks

##### Conclusion

Each architecture adds complexity (and cost in tokens/time) but also power. Reflection is simplest, ReAct adds structure, and Reflexion adds grounding. In practice, LangGraph makes it easy to experiment: you can even start with the built-in `create_react_agent` for a ReAct baseline, then customize as needed.

By understanding these patterns, you can build agents that evaluate and refine their own outputs. Whether through introspection or by leveraging tools and external data, self-improving agents aim for higher-quality, more reliable AI behavior.

### Multi-Agent LLM Systems Fundamentals

Large Language Models (LLMs) have revolutionized AI by handling a wide range of language tasks. However, relying on a single LLM agent to manage complex, multi-faceted workflows often leads to limitations such as *context overload*, *role confusion*, and *difficulty in debugging*.

Multi-agent LLM systems overcome these challenges by distributing the workload across multiple specialized LLM agents that collaborate through well-defined communication and coordination patterns. This design mirrors effective human teamwork, where clear roles and focused expertise lead to better outcomes.

#### Why Use Multiple LLM Agents?

##### Challenges of a Single LLM Agent

- Context overload: A single agent juggling data retrieval, analysis, writing, and critique within one conversation can lose track of details or degrade performance.
- Role confusion: Switching between distinct cognitive modes (creative writing vs. critical review) often causes inconsistent output quality.
- Debugging difficulty: Identifying which reasoning step caused an error is hard when all logic runs in one model.
- Quality dilution: The agent may be "good enough" at many tasks but not excel in any.

##### How Multi-Agent LLM Systems Help

By assigning each specialized agent a focused role, multi-agent LLM systems:

- Maintain clear responsibilities for each subtask.
- Enable targeted prompt engineering per agent.
- Facilitate modular debugging and quality control.
- Support scalable architectures by adding or updating agents independently.

##### Tangible Examples of Multi-Agent LLM Systems

###### Example 1: Automated Market Research Report

Workflow:

- Research Agent: Collects data on market trends, competitors, and recent news from databases and APIs.
- Data Analysis Agent: Interprets numerical trends, detects growth patterns, and flags anomalies.
- Writing Agent: Crafts a structured, engaging report using the research and analysis inputs.
- Critique Agent: Reviews the draft for logical consistency, completeness, and clarity.
- Editor Agent: Polishes grammar and style, ensuring the final output meets publishing standards.

Benefit: Each agent is optimized for a distinct cognitive task, leading to a faster, more accurate, and well-rounded report than a single LLM attempting all steps.

###### Example 2: Customer Support Automation

- Intent Detection Agent: Classifies the user's request (billing, technical support, general inquiry).
- Knowledge Retrieval Agent: Fetches relevant FAQ answers or ticket histories.
- Response Generation Agent: Creates a personalized, context-aware reply.
- Escalation Agent: Detects unresolved issues and hands off to a human agent with a summary.

Benefit: Specialized agents enable dynamic and accurate handling of diverse customer requests while ensuring smooth handoffs.

###### Example 3: Legal Contract Review

- Clause Extraction Agent: Identifies and extracts key clauses from lengthy contracts.
- Compliance Agent: Checks clauses against regulatory requirements.
- Risk Analysis Agent: Flags ambiguous or risky terms.
- Summary Agent: Produces an executive summary highlighting concerns.
- Report Generator Agent: Compiles findings into a formatted legal memo.

Benefit: Dividing the review into subtasks helps ensure thoroughness, legal accuracy, and actionable summaries for clients.

##### Communication and Collaboration Patterns

###### Sequential (Pipeline)

Agents work in sequence, passing outputs downstream.

Example: Research → Analysis → Writing → Review

###### Parallel with Aggregation

Multiple agents perform tasks simultaneously, then a compiler agent integrates results.

Example: Technical Writing, SEO Analysis, and Fact-Checking tasks all run concurrently for a blog post.

###### Interactive Dialogue

Agents exchange messages to clarify and refine.

Example: A requirements agent queries a data agent, which asks the filter agent for more details before finalizing recommendations.

###### Communication Protocols

Effective multi-agent coordination relies on standardized communication protocols, including:

- Model Context Protocol (MCP): An open standard designed to enable LLMs to interact seamlessly with external tools, databases, and services via a structured, JSON-RPC based interface. MCP facilitates real-time context sharing and modular integration across diverse AI components.

- IBM Agent Communication Protocol (ACP): A protocol aimed at standardizing message exchanges among autonomous AI agents. ACP supports modular, secure, and scalable communication, underpinning frameworks such as BeeAI for enterprise-grade multi-agent collaboration.

##### Frameworks Supporting Multi-Agent LLM Systems

Several emerging frameworks simplify building, orchestrating, and managing multi-agent LLM systems:

- LangGraph: Enables graph-based orchestration where agents read/write shared state, supports conditional routing, and manages complex workflows visually.

- AutoGen: Allows agents to self-organize, negotiate task ownership through multi-turn conversations, and improve collaboration adaptively over time.

- CrewAI: Focuses on structured multi-agent workflows with strict interface contracts between agents. It enables high-fidelity data passing using typed data models (e.g., Pydantic), enforcing clear input/output definitions to reduce errors.

- BeeAI: Designed for enterprise AI workflows, BeeAI supports modular multi agent orchestration. It emphasizes reliability, scalability, and easy integration into existing AI pipelines and uses IBM's ACP for agent communication.

##### Implementation Challenges and Design Considerations

- Context Management: How to share relevant information without overwhelming agents.
- Granularity: Finding the right balance between too few (generalist) and too many (overhead) agents.
- Communication Costs: Balancing thorough information exchange with latency and compute efficiency.
- Error Handling: Defining fallback or retry mechanisms when agents fail.

##### Summary: Why Multi-Agent LLM Systems?

By leveraging specialized LLM agents that collaborate efficiently, multi-agent LLM systems produce higher-quality, more reliable, and maintainable AI workflows. They excel in complex, multi-step applications where diverse cognitive skills and flexible coordination are essential.


### Building Multi-Agent Systems with LangGraph



This guide demonstrates how to implement multi-agent workflows using LangGraph, a graph-based framework for orchestrating AI agents working together to complete complex tasks.

##### What is LangGraph?

LangGraph structures AI workflows as directed graphs where each node represents an agent or processing step, and edges control the flow of data and execution between them. The shared state enables collaboration among all agents.

###### Key Benefits

- Modular design with independently testable agents
- Dynamic routing based on runtime conditions
- Shared memory accessible by all nodes
- Visual, clear workflow representation

##### State Management

Before implementing, you need to define a shared state type that all agents will read from and update.

```python
from typing import TypedDict, List, Optional
class SalesReportState(TypedDict):
    request: str
    raw_data: Optional[dict]
    processed_data: Optional[dict]
    chart_config: Optional[dict]
    report: Optional[str]
    errors: List[str]
    next_action: str
```

##### Agent Nodes

Agents are functions that receive the shared state, perform their task, and return the updated state. Below are simplified placeholders to illustrate this pattern.

###### Agent Function Placeholders

```python
def data_collector_agent(state: SalesReportState) -> SalesReportState:
    # Placeholder: collect raw data based on request
    # Update state with raw_data and set next_action
    return state
def data_processor_agent(state: SalesReportState) -> SalesReportState:
    # Placeholder: process raw_data and update processed_data
    # Set next_action to next step
    return state
def chart_generator_agent(state: SalesReportState) -> SalesReportState:
    # Placeholder: create chart configuration from processed_data
    # Update chart_config and set next_action
    return state
def report_generator_agent(state: SalesReportState) -> SalesReportState:
    # Placeholder: generate textual report using processed_data
    # Update report and set next_action to complete
    return state
def error_handler_agent(state: SalesReportState) -> SalesReportState:
    # Placeholder: handle errors, prepare error messages in report
    # Set next_action to complete
    return state
```

##### Routing Logic

The workflow requires a router to decide which agent to run next based on the current state.

Routing Function Example

```python
def route_next_step(state: SalesReportState) -> str:
    routing = {
        "collect": "data_collector",
        "process": "data_processor",
        "visualize": "chart_generator",
        "report": "report_generator",
        "error": "error_handler",
        "complete": "END"
    }
    return routing.get(state.get("next_action", "collect"), "END")
```

##### Building and Compiling the Workflow Graph

Using LangGraph's `StateGraph`, you add nodes for each agent, define conditional edges based on the routing function, and set the entry point.

###### Workflow Construction Example

```python
from langgraph.graph import StateGraph, END
def create_sales_report_workflow():
    workflow = StateGraph(SalesReportState)
    workflow.add_node("data_collector", data_collector_agent)
    workflow.add_node("data_processor", data_processor_agent)
    workflow.add_node("chart_generator", chart_generator_agent)
    workflow.add_node("report_generator", report_generator_agent)
    workflow.add_node("error_handler", error_handler_agent)
    workflow.add_conditional_edges("data_collector", route_next_step, {
        "data_processor": "data_processor", "error_handler": "error_handler", END: END
    })
    workflow.add_conditional_edges("data_processor", route_next_step, {
        "chart_generator": "chart_generator", "error_handler": "error_handler", END: END
    })
    workflow.add_conditional_edges("chart_generator", route_next_step, {
        "report_generator": "report_generator", "error_handler": "error_handler", END: END
    })
    workflow.add_conditional_edges("report_generator", route_next_step, {
        "error_handler": "error_handler", END: END
    })
    workflow.add_conditional_edges("error_handler", route_next_step, {END: END})
    workflow.set_entry_point("data_collector")
    return workflow.compile()
```

##### Running the Workflow

Once compiled, the workflow can be invoked with an initial state. This runs the agents in order, respecting the routing logic.

###### Running Example

```python
def run_sales_report_workflow():
    app = create_sales_report_workflow()
    initial_state = SalesReportState(
        request="Q1-Q2 2024 Sales Analysis",
        raw_data=None,
        processed_data=None,
        chart_config=None,
        report=None,
        errors=[],
        next_action="collect"
    )
    print("Starting workflow...\n")
    final_state = app.invoke(initial_state)
    print("\nWorkflow Complete\n")
    if final_state["errors"]:
        print("Errors:")
        for err in final_state["errors"]:
            print(f"- {err}")
    print("\nFinal Report:\n", final_state["report"])
    return final_state
if __name__ == "__main__":
    run_sales_report_workflow()
```

#### Multi-Agent Systems and Agentic RAG with LangGraph

##### Why multi-agent systems?

Challenge faced by single LLM agents	| Multi-agent system solution
----------- | ----------------
Context overload	| Splits tasks among agents to reduce burden
Role confusion	| Agents specialize in distinct cognitive roles
Debugging difficulty	| Modular agents ease error tracing
Quality dilution	| Each agent excels at a focused subtask

##### Typical multi-agent communication patterns

Pattern	| Description	| Example
------- | --------- | -----------
Sequential (Pipeline)	| Agents work one after another, passing results	| Research → Analysis → Writing → Review
Parallel with aggregation	| Multiple agents run concurrently, results combined	| SEO analysis, fact-checking, writing run in parallel
Interactive dialogue	| Agents exchange messages to clarify or refine	| Requirements agent queries data agent before finalizing

##### Real-world multi-agent use cases

Use case	| Agents & workflow	| Benefit
---------- | ------------- | ------------
Automated market report	| Research → Data analysis → Writing → Critique → Editing	| Faster, accurate, well-rounded reports
Customer support |	Intent detection → Knowledge retrieval → Response → Escalation	Dynamic, personalized, scalable responses
Legal contract review	| Clause extraction → Compliance check → Risk analysis → Summary	|Thorough, accurate, actionable legal reviews

##### Communication protocols

- Model Context Protocol (MCP): JSON-RPC-based interface for LLMs to interact with external tools/services, enabling modular, real-time collaboration.
- IBM Agent Communication Protocol (ACP): Standardizes message exchange among autonomous agents for secure, scalable enterprise workflows.


##### Frameworks supporting multi-agent LLM systems

Framework	| Focus/Features
------------- | --------------
LangGraph	| Graph-based orchestration, shared state, dynamic routing
AutoGen	| Agent self-organization, negotiation, adaptive collaboration
CrewAI	| Structured workflows, strict typed interfaces (Pydantic), high-fidelity data passing
BeeAI	| Enterprise-grade modular orchestration, uses IBM ACP

##### LangGraph multi-agent workflow essentials

- Directed graph nodes: represent agents/tasks
- Edges: control flow between agents
- Shared state: a TypedDict or similar, passed and updated by all agents
- Routing logic: dynamically determines the next agent based on the state

###### Example of shared state definition

```python
from typing import TypedDict, Optional, List
class SalesReportState(TypedDict):
    request: str
    raw_data: Optional[dict]
    processed_data: Optional[dict]
    chart_config: Optional[dict]
    report: Optional[str]
    errors: List[str]
    next_action: str
```

###### Example of agent node skeleton
```python
def data_collector_agent(state: SalesReportState) -> SalesReportState:
    # Collect raw data based on state['request']
    state['raw_data'] = {...}
    state['next_action'] = 'process'
    return state
```

Repeat for other agents: `data_processor_agent`, `chart_generator_agent`, `report_generator_agent`, `error_handler_agent`.

###### Example of routing function

```python
def route_next_step(state: SalesReportState) -> str:
    routing = {
        "collect": "data_collector",
        "process": "data_processor",
        "visualize": "chart_generator",
        "report": "report_generator",
        "error": "error_handler",
        "complete": "END"
    }
    return routing.get(state.get("next_action", "collect"), "END")
```

##### Building the workflow graph

```python
from langgraph.graph import StateGraph, END
def create_workflow():
    workflow = StateGraph(SalesReportState)
    workflow.add_node("data_collector", data_collector_agent)
    workflow.add_node("data_processor", data_processor_agent)
    workflow.add_node("chart_generator", chart_generator_agent)
    workflow.add_node("report_generator", report_generator_agent)
    workflow.add_node("error_handler", error_handler_agent)
    # Define conditional edges based on routing decisions
    workflow.add_conditional_edges("data_collector", route_next_step, {...})
    # Repeat for other nodes...
    workflow.set_entry_point("data_collector")
    return workflow.compile()
```

##### Running the workflow

```python
def run_workflow():
    app = create_workflow()
    initial_state = SalesReportState(
        request="Q1-Q2 Sales Report",
        raw_data=None,
        processed_data=None,
        chart_config=None,
        report=None,
        errors=[],
        next_action="collect"
    )
    final_state = app.invoke(initial_state)
    return final_state
```

##### Agentic RAG systems

- Combine Retrieval, Reasoning, and Verification using specialized agents
- Retrieval agent fetches relevant knowledge/data
- Reasoning agent performs inference and decision-making
- Verification agent checks results for accuracy and consistency
- Multi-agent design improves reliability and trustworthiness

##### Best practices & challenges

Challenge	| Strategy
------- | -------
Context management	| Share only relevant info, avoid overload
Granularity	| Balance agent count — not too few or too many
Communication cost	| Optimize message size and frequency
Error handling	| Implement fallback, retries, and error agents

### Agentic AI Protocols


#### What are AI agent protocols?

AI agent protocols establish standards of communication among AI agents and between AI agents and other systems. These protocols specify the syntax, structure and sequence of messages, along with communication conventions such as the roles agents take in conversations and when and how they respond to messages.

Agent-based AI systems often run in silos. They're built by different providers using diverse AI agent frameworks and employing distinct agentic architectures. Real-world integration becomes a challenge, and coupling these fragmented systems requires tailored connectors for all possible types of agent interaction.

This is where protocols come in. They turn disparate multi-agent systems into an interlinked ecosystem where AI-powered agents share a way of discovering, understanding and collaborating with each other.

Although agentic protocols are part of AI agent orchestration, they don't act as orchestrators. They standardize communication but don't manage agentic workflow coordination, execution and optimization.

#### Examples of AI agent protocols

Many protocols are still in their early stages, so they have yet to be widely used or applied on a large scale. This lack of maturity means that organizations must be prepared to act as early adopters, adjusting to breaking changes and evolving specifications.

As agentic technology evolves, new protocols might emerge. Here are some current common AI agent protocols:

- Agent Communication Protocol (ACP)
- Agent Network Protocol (ANP)
- Agent-User Interaction (AG-UI) Protocol
- Agent2Agent (A2A) Protocol
- Model Context Protocol (MCP)
- Agent Payments Protocol (AP2)

###### Agent Communication Protocol (ACP)

ACP is an open standard for agent-to-agent communication. With this protocol, we can transform our current landscape of siloed agents into interoperable agentic systems with easier integration and collaboration. With ACP, originally introduced by IBM's BeeAI, AI agents can collaborate freely across teams, frameworks, technologies and organizations. It's a universal protocol that transforms the fragmented landscape of today's AI agents into interconnected teammates and this open standard unlocks new levels of interoperability, reuse and scale. 


###### Agent Network Protocol (ANP)

The Agent Network Protocol (ANP) is an open-source communication framework created specifically for intelligent agents. It functions much like HTTP but is tailored for a future where agents are the primary entities operating online. ANP allows these agents to locate, connect with, and interact across the internet, fostering an open and secure environment for collaboration between agents.

ANP addresses a core challenge in the AI ecosystem: the lack of a standardized, secure, and efficient method for agent-to-agent communication. By offering a unified protocol, ANP lays the groundwork for seamless interaction among agents in the evolving landscape of the AI-driven internet.

###### Agent-User Interaction (AG-UI) Protocol

AG-UI is a streamlined, open protocol based on events, designed to standardize the connection between AI agents and user-facing applications.

It emphasizes ease of use and adaptability, providing a consistent structure for the exchange of agent state, user interface intents, and user interactions between your agent and front-end applications. This enables developers to quickly build and deploy agent-driven features that are stable, easy to debug, and user-friendly—without needing to rely on complex, custom integrations, allowing them to focus on core application functionality.


###### Agent2Agent (A2A) Protocol

The A2A protocol is an open standard for AI agent communication initially launched by Google and now managed under the Linux Foundation. It follows a client-server model setup with a three-step workflow:

Discovery occurs when an entity (a human user or another AI agent) initiates a task request to a client agent, which then looks up remote agents to determine the best fit.
Once the client agent identifies a remote agent capable of fulfilling the task, it then goes through authentication. The remote agent is responsible for authorization and granting access control permissions.
Communication proceeds with the client agent sending the task and the remote agent processing it. Agent-to-agent communication happens over HTTPS for secure transport, with JSON-RPC (Remote Procedure Call) 2.0 as the format for data exchange.

###### Model Context Protocol (MCP)

Introduced by Anthropic, the Model Context Protocol (or MCP) provides a standardized way for AI models to get the context they need to carry out tasks. In the agentic realm, MCP acts as a tier for AI agents to connect and communicate with external services and tools, such as APIs, databases, files, web searches and other data sources. MCP encompasses these three key architectural elements:

- The MCP host contains orchestration logic and can connect each MCP client to an MCP server. It can host multiple clients.
- An MCP client converts user requests into a structured format that the protocol can process. Each client has a one-to-one relationship with an MCP server. Clients manage sessions, parse and verify responses and handle errors.
- The MCP server converts user requests into server actions. Servers are typically GitHub repositories available in various programming languages and provide access to tools. They can also be used to connect LLM inferencing to the MCP SDK through AI platform providers such as IBM and OpenAI.

In the transport layer between clients and servers, messages are transmitted in JSON-RPC 2.0 format using either standard input/output (stdio) for lightweight, synchronous messaging or HTTP for remote requests.

###### Agent Payments Protocol (AP2)

In September 2025, Google unveiled a new standard called Agent Payments Protocol (or AP2). It is built in collaboration with leading players in the payments and technology sectors. AP2 is designed to enable secure, cross-platform transactions, initiated by AI agents. It can also function as an extension to existing protocols such as Agent2Agent (A2A) and the Model Context Protocol (MCP).

Unlike traditional payment systems that rely on a person clicking a 'Buy Now' button, AP2 is designed for scenarios where autonomous agents make purchases on behalf of users.


##### How do the A2A, MCP, and AP2 protocols work together for agentic commercial transactions?

The A2A, MCP, and AP2 protocols work together to form the framework of agentic commercial transactions. For example:

- User request: "Order me a new pair of wireless headphones with noise cancellation, under $250."
- A2A at work: The shopping agent communicates with a retailer's product agent and a payment service agent to manage the process.
- MCP at work: The agent gathers product details, user preferences, and past purchase history via the Model Context Protocol.
- AP2 at work: After selecting a suitable product, the agent creates a Cart Mandate. Once the user reviews and approves it, the payment agent securely processes the transaction.

#### Criteria for choosing an AI agent protocol

With the lack of benchmarks for standardized evaluation, enterprises must conduct their own appraisal of the protocol that best fits their business needs. They might need to start with a small and controlled use case combined with thorough and rigorous testing.

Here are a few aspects to keep in mind when assessing agent protocols:

- Efficiency: protocols are designed for swift data transfer and rapid response times. Although some communication overhead is expected, but it must be kept to a minimum.
- Reliability: AI agent protocols must be able to handle changing network conditions throughout agentic workflows, with mechanisms in place to manage failures or disruptions. For instance, ACP is designed with asynchronous communication as the default, which suits complex or long-running tasks. Meanwhile, A2A supports real-time streaming using SSE for large or long outputs or continuous status updates.
- Scalability: Protocols must be robust enough to cater to growing agent ecosystems without compromising performance. Evaluating scalability can include increasing the number of agents or links to external tools over a period of time, either gradually or suddenly, to observe how a protocol operates in those conditions.
- Security: Maintaining security is paramount, and agent protocols are increasingly incorporating safety guardrails. These include authentication, encryption and access control.



### What is MCP?

Model Context Protocol (MCP) is a new open-source standard to connect your agents to data sources such as databases or APIs. MCP consists of multiple components. The most important ones are the host, the client, and the server. 

Your MCP host will include one or more MCP clients. The MCP host could be an application chat app, code assistant in your IDE etc. The MCP host will connect to one or more MCP servers. The MCP host and servers will connect over each other through the MCP protocol which is a transport layer in the middle. Whenever your MCP host or client needs a tool, it's going to connect to the MCP server. The MCP server will then connect to, for example, a database (SQL or NoSQL) or APIs or data sources such as a local file type or maybe code. This is especially useful when you're building a code assistant in your IDE. 

Let's look at an example of how to use MCP in practice. Suppose we have our MCP host and client,  a large language model and our MCP servers. Let's assume our MCP client and host is a chat app. You ask a question such as what is the weather like in a certain location or how many customers do I have? The MCP host will need to retrieve tools from the MCP server. The MCP server will then conclude and tell which tools are available. From the MCP host, you would then have to connect to the large language model and send over your question plus the available tools. If all is well, the LLM will reply and tell you which tools to use. Once the MCP host and client knows which tools to use, it knows which MCP servers to call. When it calls the MCP server in order to get a tool result, the MCP server will be responsible for executing something that goes to a database, to an API, or a local piece of code. For this, there could be subsequent calls to MCP servers. The MCP server will reply with a response which you can send back to the LLM. And finally, you should be able to get your final answer based on the question that you asked in the chat application. 

The MCP Protocol is a new standard which will help you to connect your data sources via MCP server to any agent. Even though you might not be building agents, your client might be.

### Why MCP?

Model Context Protocol (MCP) is an open protocol that allows developers to build an MCP server using any language, including JavaScript, C Sharp, Python, Java, and others. It standardizes the connections between AI applications, large language models, and any available external data and service. 

There are two main needs for an AI agent. The first is to provide context in the form of contextual data, such as documents, database entries, or articles. The second is to provide tools and tool capabilities for AI agents to perform actions or execute tools, such as performing a web search, calling an external service to book a reservation, or performing a complex calculation. 

MCP behaves like a universal layer to enable developers to build tools that interact with any resource in a uniform and standardized manner. Why is this standardization so important to developers? 

Benefit | How it helps
------ | ---------
Extendability | Easily add functionality or tools in the future without affecting the existing environment
Interoperability | Build applications across platforms and vendors
Consistency | Use tools that behave the same regardless of the model
Reusability | Build Once and reuse everywhere 
Rapid development | No need for custom integrations 

###### Key benefits of MCP 

Feature | Description | Benefit
-------- | -------- | ---------
Standardized Integration | Common protocol to make connections between models and external data | <ul><li>Simplifies and speeds up the development process</li><li>Eliminates custom connections </li></ul> 
Simple Architecture | Simple client-serving plug-and-play model | <ul><li>Makes it easy to establish connections</li><li>Enables fast and scalable set up and deployment</li></ul>
Interoperability | Supports diverse and mixed development environments | <ul><li>Allows developers to work across different platforms, models, and frameworks</li></ul>
Security | Built-in security with OAuth2.0 authorization and token-based authentication | <ul><li>Enhances security and acess</li><li>Can be further enhanced by adding end-to-end encryption for data and communications traffic via SSL/TLS</li></ul>
Minimized AI Hallucinations | unlike LLMs, MCP has its own external sources of up-to-date training data | <ul><li>Reduces occurences of AI hallucination</li><li>Decreases the likelihood of incorrect and outdated information in responses</li></ul>
Agentic workflow support | AI agents can talk to other AI agents to reach the best possible result | <ul><li>Enables more complex multi-step stacks</li><li>Boosts automation opportunites</li></ul>
Data relevance | Unlike LLMs, MCP servers can fetch the ost recently available data from their connections to external data sources | <ul><li>Increases the relvance of the information returned</li></ul>


#### MCP Applications

In terms of enterprise organizations, some common tasks including 
- Connecting to internal systems, such as databases, customer relationship management or CRM systems, and ticketing platforms and services
- Workflow automation and report generation
- Provide access to live information such as stocks and share prices, current weather reporting, and the latest up-to-the-minute news. 
  
In terms of agentic AI, MCP can be used to 
- enable models to use autonomous reasoning to determine what tool to select based on the goals of the user. And agents powered by MCP can use multiple data sources to help deliver more enhanced context to make quicker and more informed decisions. 
- For DevOps, these include CICD pipeline automation, code management for repos, such as GitHub, infrastructure automation, and automated incident response solutions. 
- For NetOps, uses might include network management and performance monitoring, router and firewall configuration, and anomaly detection and issue remediation. 
- For SecOps, MCP could be beneficial in proactive threat mitigation and response systems, real-time incident orchestration, and automated identification and management of security vulnerabilities. 



Imagine you're an AI engineer building a Retrieval Augmented Generation, or RAG, system for a bank that has over 100,000 documents. Your goal is to retrieve only a few relevant pieces of text from this massive collection and feed them into your Large Language Model, or LLM, to answer questions or generate insights. To implement this, you'd typically need to store all 100,000 documents along with their vector embeddings. You could use a vector database for this, but managing and scaling such an infrastructure would be quite complex and time-consuming. So, instead, imagine having a dedicated server that does everything you need. When your LLM receives a query, it sends a prompt to the MCP server. The server then performs the retrieval, RAG step, i.e., determining which documents are most relevant, and then returns only those relevant chunks of text. That's exactly what an MCP server can do for you. 

Bear in mind that MCP is a relatively new technology, and therefore, we are only just discovering all the possible ways we can use it to our advantage. 

### MCP Architecture 

###### MCP Host
  - AI application responsible for coordinating and managing MCP clients. This is where your AI operates and interact with people
    - Might be a chatbot, an IDE-based code assistant (Claude Desktop, VS Code)
    - Specialized AI-powered IDE tool (Cursor, Windsurf) 
    - A desktop application. 
  - Acts as the bridge between the user and the underlying AI model
    - Captures user input (text, commands or files)
    - Send the input to AI for processing
    - Presents AI response in a clear, usable way
  - Handles supoorting functions
    - Managing conversations
    - Maintaining context
    - Applying interface controls
    - Integrates with other tools
  - Provides structure, interface and user experience

######  MCP Client
  - Maintains a connection and communication between MCP host and MCP server 
  - Runs within the MCP host application
    - Translates user requests into structured standardized format
  - Can be viewed as a unified gateway for communication
    - MCP host sends structured requests to MCP client using JSON-RPC protocol
    - All interactions routed through single consistent interface
- Interpret server responses
- Perform error handling
- Ensure responses are contextually relevant, accurate and appropriate
- Plays critical role in managing session lifecycle
  - Interruptions
  - Timeouts
  - Reconnections
  - Session termination
- Each MCP client maintains a 1-1 relationship with specific MCP server
- Some well-known MCP client examples:
  - BeeAI (IBM)
  - Copilot Studio (Microsoft)
  - Clause Desktop
  - Claude.ai
  - Cursor AI
  - Windsurf Editor

###### MCP Server
  -  Is an external service that supplies context to LLMs by 
     - Translating user requests into concrete server-side actions
   - Examples of MCP integrations include Slack, GitHub, Git, D ocker, or Websearch. 
   - These servers are typically GitHub repositories 
     - Available in various programming languages, including C Sharp, Java, TypeScript, Python, and others. 
     - Provide access to MCP tools. MCP servers
  - Act as a bridge between LLM interface and the MCP SDK. 
    - They enable the creation of reusable MCP services 
  -  Highly flexible, as they can integrate with both internal systems and external resources or tools. 
 
MCP primitives are the core concept of MCP, defining what servers and clients can provide for one another. They determine the kinds of contextual information that can be shared with AI applications and the actions that can be carried out. In MCP, servers can expose three types of core primitives as follows: 
- Tools offer functions that AI apps can invoke to perform 
    -  calculations, sending messages, file operations, database calls, or data retrieval via API calls. 
 -  Resources, which provide AI apps with access to contextual information from internal or external data sources. 
    -  returns data only, such as file contents, database records, or API responses; no actions 
 - Prompts, which are reusable templates and workflows that structure and streamline communication and interactions between the LLM and the server. 

###### MCP Architecture Layers 
- Data Layer
  - Defines a JSON-RPC-based client-server protocol
  - Lifecycle management
  - Core primitives
- Transprot Layer
  - Two-way communication channels and authentication between clients and servers. 
  - Connection establishment, message framing, and secure communication between MCP nodes. 
  
In the client-to-server stream, MCP protocol messages are converted into JSON-RPC format, allowing for the transport of different data structures and their processing rules. In the reverse, server-to-client stream, the returned responses in JSON-RPC format are converted back into MCP protocol messages. The three JSON-RPC message types include 
- Requests 
- Responses
- Notifications
  
Requests require a response from the server, whereas notifications do not.  

Between clients and servers, the MCP transport layer defines two primary transport mechanisms, both transporting messages in JSON-RPC 2.0 format. The first is Standard Input-Output, or STDIO. This mechanism is particularly effective for integrating local resources, as it provides simple, lightweight, synchronous message exchange with systems, such as local file systems, databases, and local APIs. 

<p align="center">
  <img src="./assets/agentic_ai/mcp1.png" alt="drawing" width="600" height="250" style="center" />
</p>

The second is Streamable HTTP. This mechanism facilitates client-to-server communication over HTTP post, with optional support for server-sent events to enable streaming. It is designed for remote server interactions and accommodates standard HTTP authentication methods such as bearer tokens, API keys, and custom headers. The MCP standard advises using OAuth authorization framework to obtain authentication tokens. 


### MCP in Action

In 2026, the way we build applications has fundamentally shifted, thanks to Model Context Protocol (MCP). MCP is the protocol that finally lets us retire our collection of duct tape, baling wire, and hand-rolled JSON glue code. Instead of writing yet another custom integration every time an AI model needs to talk to an API, MCP standardizes. So it's basically a USB-C for AI agents. One connector to rule them all. 

Plug your AI models into corepositories, communication platforms, mapping services, or any tool you need and watch the magic happen. No more adapters. No more SDK bingo. The MCP host is where the main app runs. This is everything running right in the center. It includes the MCP client and connects to all the tools and data the AI needs to do its job. The MCP client sits inside the host and talks to one or more MCP servers. Servers can find the right function to call and handle the back and forth to get the answers the AI needs. The MCP server is where all the tools live. The server connects to external systems and offers up three things. 
- Tools which are functions that the AI can call.
- Resources where all the data comes from. 
- Prompts which are predefined, preset instructions that help guide the AI's behavior. 
 
##### Some real-world applications of MCP
 
 ###### GitHub
  Inside the GitHub MCP server, you connect your AI agent directly to the GitHub's API. This means the AI can automatically handle tasks like managing repositories, issues, pull requests, branches, and releases, all while taking care of authentication and error handling for you. 
  
The AI can do the following:
- It can review pull requests automatically and flag potential problems. It helps spot bugs earlier by analyzing code changes. 
- It helps enforce consistent coding standards across your team. 
-  It can sort and prioritize incoming issues so your team knows exactly what to work on and what's really most important. 
-  It keeps your dependencies up to date without you having to lift a finger. 
-  It scans for security vulnerabilities and alerts you early, so no more nasty surprises later. 

If you're managing multiple repositories or even just one really busy one, MCP takes care of a lot of the routine work that normally eats up developer time. Instead of spending hours on maintenance, your team gets to really focus on what's most important. 

###### SaaS 
You run a company that provides online software. Your customers often need help such as password resets, Billing questions, Bug reports, Technical troubleshooting of all sorts. The problem is that normally you'd need a big support team answering emails, looking up customer data, checking logs, and sometimes escalating the issues to the engineer. With MCP, you connect your agent to all the tools your support team normally uses:
- The customer database to find user info
- The billing system to check payments
- The server logs to analyze issues
-  The knowledge base to find help articles
-   The ticketing system to create or update support tickets. 

Because MCP defines a standard way for the AI to talk to all these tools, you don't have to build a custom connection every single time to connect them. It's much easier. The AI can see all the data it needs, call the right functions, and handle most support cases automatically. For example, a customer writes, "Hi, I can't log in. It says my subscription expired, but I just paid." The AI, using MCP, can then look up the customer's account, check billing records, verify payment status, update the subscription if needed, and then reply, "Thanks for reaching out. I've confirmed your payment and reactivated your account. You should be able to log in now." The benefit is that you have faster support for your customers, less work for your human support team, fewer mistakes because AI always checks all systems the same way, and it's much easier to scale as your company grows. So in simple terms, MCP lets your AI talk to your company's systems like a real support agent, but faster, 24-7, and without needing custom code for every system. 

These two real-life examples of MCP have shown us that MCP really is a game changer. Those teams who build their applications using MCP will take their applications to a new level. 


### Run existing MCP Server

Fast MCP servers behave like remote function libraries. Instead of importing locally, you connect to services anywhere. Here we have a simple MCP server to perform mathematical calculations. The client is your code that connects to an MCP server and is represented by a Python object. 

```python
response = client.call_tool("add", {"a": "1", "b": "2"})
print(response)
```
```o
3
```

Each server has tools, like `add` and `multiply`. The server can run anywhere. Your code talks to it over the network. We call tools using the tool name and parameters. The tool name is `add`, and we pass parameters `a` and `b` with values 1 and 2. The parameters are set, the add tool is selected, and the result 3 is transmitted back via the transport. 

The MCP server communicates with the client via a transport mechanism. 

- Standard Input Output (STDIO): This is used to communicate with a server running on your own system. 
  - STDIN: The client uses STDIN to send messages to the MCP server. 
  - STDOUT: The server uses STDOUT to communicate back to the client. 
  - STDIO is ideal for testing in applications that don't need network connectivity. 

- Hypertext Transfer Protocol (HTTP): used to communicate with a server over a network connection, rather than just your own system.  
  - The client sends HTTP requests to the MCP server, and the server responds with HTTP responses. 
  - Used for remote setups, where the client and server may run on different machines or across the internet. 
  - Enables MCP servers to integrate with web tools and cloud services. 
 
###### Example: Context7
Context 7 provides up-to-date code documentation and API references formatted specifically for LLMs. You input library names, search queries, or requests, and the output it returns can be up-to-date code snippets and documentation formatted in Markdown for LLMs, library trust scores and compatibility details, or API references and function descriptions. When we access the MCP server, we input the tool we want, in this case, resolve library ID and parameters with the library name. The tool then selects the appropriate documentation, in this case, a list of library names with a lot of other info, in this case, scikit-learn. 

The second tool is getLibraryDocs. The parameter is the output from before. The MCP server will select the tool and send the appropriate docs. First, let's see how to use the STDIO transport. We need to import some libraries. 

```python
from fastmcp import Client
from fastmcp.client.transports import StudioTransport

stdio_transport = StdioTransport(
  command="npx",
  args=["-y", "@upstash/context7-mcp"]
)

stdio_client = Client(stdio_transport)
```

First, we import client from fastmcp and stdio-transport from the `fastmcp.client.transports` module. The client library handles our connection to the MCP server, and the stdio-transport library enables communication with servers running locally on your system using standard input-output. We need to create an stdio-transport object to establish the connection with the client. It connects your Python code to the MCP server via the stdin and stdout transports. The arguments include the `npx` command that downloads and runs the `context 7` MCP server – think about it like a PIP install for JavaScript. It also includes the args command with the `- y` flag to auto-confirm installation prompts. At `upstash` or `context-7` mcp is the name of the server package we discussed earlier. We pass the stdio-transport object into the client constructor to create the stdio-client object. We use the stdio-client object to communicate with the MCP server. 

```python
async with stdio_client as client:
    # List avialable tools
    tools = await client.list_tools()
```

Since communication between stdio-client and the MCP server can take some time, Python uses `async` methods to prevent freezing, which allows other tasks to run concurrently. The `async with stdio-client` as client command creates a context manager that automatically handles connection setup and cleanup after the code in the indent has run. The `client.list-tools` command requests available tools from the server. The `await` keyword pauses this task until the server responds, while allowing other tasks to continue. The tools variable stores the list of available functions we discovered that we can call. The MCP server will send a list of tool objects as shown in the image. 



```o
[Tool(name='resolve-library-id', title='Reso...', Tool(name='get-library-docs', title='Get Library Docs', ... meta=None)]
```
We show it as a list, but what is actually transmitted depends on the protocol used. Let's examine the tools list that contains the two tool objects returned by the MCP server. The number of tools corresponds to the list length. To access info for the first tool, we use the first element of the list and examine the tool attribute .name. We get resolve-library-id, which searches for library identifiers. Similarly, for the second element, we use the name attribute to get get-library-docs, which retrieves actual documentation. 

```python
print(f"Number of tools: {len(tools)}")

print(f"First tool: {tools[0].name}")
```
```o
"Number of tools: 2"
"First tool: resolve-library-id"
```

Now, let's see how to use the HTTP transport instead. The code for using HTTP transport is almost identical to the one we saw previously for STDIO. So again, we start by importing the streamable HTTP transport library from the fastmcp.client.transports module. We create a transport layer using the streamable HTTP transport object, HTTP underscore transport. The parameter URL connects to the MCP server's HTTP endpoint at the specified URL. This enables bidirectional communication over standard web protocols anywhere. 

```python
from fastmcp.client.transports import StreamableHttpTransport

http_transport = StreambaleHttpTransport(
  url="http://mcp.context7.com/mcp"
)

http_client = Client(http_transport)
```


The HTTP transport object is used as input for the client constructor, creating an HTTP  client. 

```python
async with http_client as client:
    tools = await client.list_tools()
    response = await client.call_tool("resolve-library-id", {"libraryName": "fastmcp"})
    docs = await client.call_tools("get-library-docs", {
      "context7CompatibleLibraryID": "/punkpeye/fastmcp", "tokens": 5000
    })


```


We use async with HTTP underscore client as a client pattern. The client.list underscore tools method returns identical tool objects. We use client.call underscore tool, resolve library ID, to find library IDs and search for FastMCP options. And we use client.call underscore tool, get library docs, to retrieve code snippets using the library ID. Only the transport differs, the rest is identical. 

###### Other transport types
Here's a concise summary of the other MCP transports that can be used. 
- Server Sent Events (SSE): 
  - Legacy transport for HTTP communication, now replaced by the streamable HTTP transport
  -  Uses asymmetrical channel communication
- In-memory transport: 
  -  Direct connection within the same Python process
  - No network overhead
  - Direct function calls between the different components
  -  Primarily used for development and testing purposes

 But the bottom line is, you shouldn't need to use these as STDIO and HTTP cover most real-world MCP use cases. 
 
 ##### Tools and Clients
 - When clients receive tools, they are structured by MCP
    -  Clients require some sort of adaptation to MCP to understand and use the tools. 
    -  LangChain has an MCP adapters library to convert MCP tools into LangChain-structured tools compatible with LangChain and LangGraph. 
-  MCP is a standard that requires clients to conform to its specifications to both access and invoke its capabilities. 
-  The term "tool" is used across different agentic frameworks and protocols
-  Different clients adapt tools to different scenarios. 


### Build an MCP Application with Python
The next step is to build your first MCP application.  Building an MCP application starts by integrating your MCP server with an agent, allowing the agent to use your server tools, data, or capabilities within a larger workflow.

- Transport is how MCP servers and clients communicate
- Integrate your MCP server with an agent
  - Allows the agnet to use your server's tools, data or capabilities within a larger workflow
 
 Let's say we want the MCP server to process X and give us Y. The agent receives a prompt that requires an MCP tool. The agent selects which MCP tool to use and extracts the parameter X and sends it to the MCP server via the client and transport. The MCP server processes X and produces Y. The MCP server sends the result back to the LLM on the agent via the transport and client. The agent processes the response and provides the final output. 
 
 You should be able to recall that there are two main MCP server subtypes, and each uses its own transport layer 
 - HTTP for remote MCP servers 
 - STDIO for local MCP servers. 

In the lab, we'll use Multi-Server MCP Client, which can connect to and update multiple MCP servers simultaneously. It supports both HTTP and STDIO MCP servers at the same time. There's no documented hard limit on how many MCP servers you can connect to with the Multi-Server MCP Client. All you provide is a list, or dictionary, of the servers. In practice, the limit is based on your resources and design, not an enforced cap. The big win for you is that it handles the low-level transport details on your behalf so you can focus on application logic. 

Using Multi-Server MCP Client, our application will connect to two real-world MCP servers. The `MetMuseum-MCP` server provides access to the Metropolitan Museum of Art's collection database – over 400,000 artworks with metadata, images, and historical information. The `Context 7` server offers LLM-optimized documentation search across major frameworks and libraries, returning documentation that AI can understand and use directly. The agent selects the best MCP tool based on the prompt. For example, if you ask, what's the best place to see art in NYC? The agent will select a tool from the MetMuseum server to search their collection. But if you ask, what is scikit-learn? The agent will select a tool from the Context 7 server to fetch documentation about the library. 

Next, we'll look at how to code the application. First, we import Multi-Server MCP Client. Then we create the Multi-Server MCP Client client object. The `Context 7` key contains the values required for connecting to the `Context 7` HTTP server. And the `MetMuseum` key contains the information needed for the STDIO server using `npx`. 

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient(
  {
    "context7": {
      "url": "http://mcp.context7/mcp",
      "transport": "streamable_http",
    },
    "met-museum": {
      "command": "npx",
      "args": ["-y", "metmuseum-mcp"],
      "transport": "studio"
    }
  }
)
```

Next, we'll import the components for our agent system. `Create_react_agent` builds a ReAct agent and a language model for powering the agent's reasoning. In-Memory Saver provides conversation memory, allowing the agent to remember context across multiple exchanges. Asyncio enables asynchronous execution for non-blocking MCP server communication. 

```python
from langchain.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

import asyncio
```

The agent needs to know what functions it can call. So, we fetch tools from all connected servers using the get_tool method. The output is a list which we can print. The first element is the output of the library documentation for Resolve library ID. The agent will use this to call the ideal tool. The second element is the output for the library documentation for the MetMuseum server. Let's create the agent and everything we need. 

```python
openai_model = ChatOpenAI(model="gpt-5-nano")

checkpointer = InMemorySaver()

config = {"configurable": {"thread_id": "conversation_id"}}

agent = create_react_agent(
  model=openai_model,
  tools=tools,
  checkpointer=checkpointer
)

response = await agent.invoke(
  {
    "message": [{"role": "system", "content": "You are smart, useful agent with tools to access code library documentation and the Met Museum"},
    {"role": "user", "content": "Give a brief introduction of what you do and the tools you can access."}
    ]
  },
  config=config
)

print(response['message'][-1].content)
```


First, we'll initialize the LLM object. Before we create the agent, the agent needs to remember previous messages in the conversation. Therefore, we'll create a Memory Saver object. We then set the configuration with a thread_id. This is useful if you have more than one conversation happening simultaneously. It acts like a session identifier. Finally, we create the agent. To create the chatbot, we start by prompting the agent. This includes setting the agent's role, asking the agent to provide an introduction of its capabilities and available tools, and then printing out the response. The agent responds with a text introduction explaining its capabilities and available tools. This includes describing how it can help users access software documentation by resolving library IDs and retrieving documents, and how it can help explore the MetMuseum collection by listing departments, searching for museum objects, and getting detailed object data and images. We then create a basic chatbot to prompt the user, starting with a while loop. We ask the user to provide 1 to ask a question or 2 to exit. If the choice is 1, we prompt the user for their question, and we feed the user prompt into the agent using a invoke. 

```python
while True:
    choice = input('''Menu: 1. Ask the agent a question 2. Quit Enter your choice (1 or 2): ''')
    if choice == "1":
        print("You questions")
        query = input("> ")
        response = await agent.ainvoke({"messages": query}, config=config)
        print(response['message'][-1].content)
    else:
        print("Goodbye!")
        break
```


We then print the agent's response. If the choice is 2, we print goodbye and break out of the loop. The chatbot displays the menu, and the user selects option 1 to ask a question. The user enters what is the Met, and the agent responds with detailed information about the Metropolitan Museum of Art, including its location, founding date, and collections. Depending on whether we use a .py file or notebook, we wrap the code in an async main function and use async.io in parentheses main with this check to run the asynchronous code. In this video, you learned that there are two main MCP server subtypes, and each uses its own transport layer. HTTP transport is used for remote MCP servers. STDIO transport is used for local MCP servers. Multi-server MCP client can connect to and update multiple MCP servers simultaneously. Multi-server MCP client supports both HTTP and STDIO MCP servers at the same time. There is no documented technical hard limit on how many MCP servers you can connect to with multi-server MCP client. The InMemorySaver object provides conversation memory, allowing the agent to remember context across multiple exchanges. Asyncio enables asynchronous execution for non-blocking MCP server communication. And when using a .py file, we wrap the code in an async main function and use asyncio.run in parentheses main to run the asynchronous code.



### Hello World of MCP Servers

After watching this video, you'll be able to Create MCP servers in STDIO and HTTP transports using FastMCP Register custom tools, resources, and prompts to an MCP server Test MCP servers with client connections and manual tool calling And create a multi-server client and ReAct agent FastMCP servers work like remote function libraries. 

Previously, you saw how to build a client and create a simple MCP application using an agent. In this video, we'll build a calculator MCP server and explore three concepts – tools, resources, and prompts. The transport is how MCP servers communicate. We'll integrate your MCP server with an agent and introduce a more complex client and transport layer for working with these agents. We'll create a FastMCP server object. Name will identify the MCP server. 

```python
from fastmcp import FastMCP
mcp = FastMCP(name="CalculatorMCPServer",
    instructions='''This server provides data analysis tools. Call get_average() to analyze numerical data.'''
)
```

And instructions will define the natural language documentation LLMs use to determine when the tools apply. 

###### Tools
Creating an MCP tool is exactly like creating a tool in LangChain. Just like the `@tool` decorator in LangChain, here we use the `@mcp.tool` decorator. 

```python
@mcp.tool()
def add(a: int, b:int) -> int:
    '''Add two integers together... '''
    return a + b

@mcp.tool()
def subtract(a: int, b:int) -> int:
    '''Subtract one integer from another... '''
    return a - b
```

Also like a LangChain tool, we have a function. In this case, a function to add two integers together. And we have a doc string. Remember, the agent needs this to determine what the function does. It's how the LLM decides when to use this tool. We use type hints for the inputs `a` and `b` plus the return value. In addition, we create a subtraction tool following the same pattern. 

###### Resources
Resources are like filing cabinets that AI systems can access. Here we define a resource with `@mcp.resource` and a URI template of file path. 

```python
@mcp.resource("file:///endpoint/{name}")
def return_template_documtn(name: str) -> str:
    ```Read a document by name```
    return f"Document contents of {name}"
```

The name is a path parameter and is also an input to the function. When a client requests this resource, the name value is extracted from the URI, passed into the function as the name parameter, and the function returns the result. For example, when we call the resource with the parameter `dog`, the client sends a request to `file:///endpoint/dog`. The string `dog` gets passed into the function, and it returns document contents of `dog`. And this result is sent back to the client. The resource returns a file. 

The name parameter is used in the URI pattern extracted from the URI. That same parameter is then passed to the function, which uses it to construct the file path, and load the text from disk in the server assuming the file exists. 

```python
@mcp.resource("file://endpoint1/{name}")
def read_document(name: str) -> str:
    '''Read a document by name from the path directory'''
    try:
        with open(f"path/{name}", "r") as f:
            return f.read()
    except FileNotFoundError:
        return f"Document {name} not found in path directory"
    except Exception as e:
        return f"Error reading document: {str(e)}"
```

###### Prompts
Prompts are reusable templates for common tasks. Instead of writing the same instructions over and over, you can save them once and the AI can call them by name. 

```python
@mcp.prompt(title="Code Review")
def review_code(code: str) -> str:
    return f"Please review this code:\n\n{code}"
```

We pass in the parameter value, for example, `code = 'n = 1'`. The output of the server is, please review this code, `'n = 1'`. 

###### Client: In-memory transport
In-memory transport is a fast way to test your MCP server. We import client from `fastmcp` and instantiate it by passing in our MCP server object. 

```python
from fastmcp import Client
client = Client(mcp)
```

This creates an in-memory connection, the simplest way to test when your client and server are in the same process. 

###### Test tools and resources
Now let's call our MCP tool to test it. We create an async function that wraps the client connection. The async with client command establishes the connection to the MCP server. The async context manager automatically handles opening and closing the connection. We then call the tool by name, which is `add`, and pass the parameter as dictionary. The await keyword waits for the server's response without blocking. 

```python
async def call_add_tool(a: int, b: int):
    async with client:
        result = await client.call_tool("add", {"a": a, "b", b})

    return result

print(await call_add_tool(4, 5))
```
```o
9
```
When we `call_add_tool` with 4, 5, we get back a response object with multiple ways to access the result of 9. 


Next we'll test resources. We create an async function that calls a resource with `client.read_resource`. Next we call the function, and we get a response. 
```python
async def call_resourcel(name):
    async with client:
        result = await client.read_resource(f"file:///endpoint/{name}")

    return result

print(await call_resource("README.txt"))
```
```o
"Document contents of README.txt"
```

###### Create and test MCP Server
Let's create an HTTP MCP server. We call the `run_http_async` method on our MCP object passing in the port number. 

```python
PORT = 8000
mcp.run_http_async(portt=PORT)
```
This starts the MCP server as an HTTP server running on port 8000 accessible at path `/mcp`. The server can be located anywhere, but as long as you have access to the URL, you can find it. 

Now we need to configure how the client will communicate with our HTTP server. We create a transport object using streamable HTTP transport. This tells the client to use HTTP as a communication protocol. 

```python
from fastmcp.client.transports StreamableHttpTransport

transport_http = StreamableHttpTransport(url=f"http://127.0.0.1:{PORT}/mcp")

http_client = Client(transport_http)
```

The URL parameter specifies the location of the server. Finally, we create the client using the transport underscore HTTP object as input to the constructor. Our application will use this client to communicate with the MCP server. 


Next we'll test our MCP client. Our test function takes three parameters, the client, and two integers, with client as a parameter now, making the function reusable with different clients. Inside async with client opens the connection, and await client dot call underscore tool invokes the add tool. The client sends integers 4 and 5 to the server. 

```python
async def test_client_http(client: Client, a: int, b: int) -> int:
    async with client:
        result = await client.call_tool(f"add", {"a": a, "b": b})
        return result

print(await test_client_http(http_client, 4, 5))
```
```o
9
```

Remember, the server can be located anywhere, on your own machine or on another continent. The two numbers are added on the server and the result is 9. This result is sent back to the client. 

###### MCP HTTP-powered Agent
We'll now import the essential components for creating our MCP HTTP powered agent. First, we import `create_react_agent` and `ChatOpenAI`, and then define chat model as our large language model. Then we import `load_MCP_tools`, which converts MCP tools to lang chain format. Finally, we import the `ClientSession` and `streamablegttp_client` components. Next, we connect our HTTP powered agent to the MCP server using async methods. 

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

llm = "openai:gpt-5-nano"

from langchain_mcp_adapters.tools import laod_mcp_tools
from mcp import ClientSession
from mcp.client.streamable_http import streamblehttp_client

async with streamablehttp_client(f"http://127.0.0.1:{PORT}/mcp") as (read, write, _sid):
    async with ClientSession(read, write) as session:
        await session.initialize()
        tools = await load_mcp_tools(session)
        agent = create_react_agent(model=llm, tools=tools)
        agent_response = await agent.ainvoke({"messages": "Uase the add tool to add 2 and 1 and let me know if you used a tool"})
```

Streamable HTTP client establishes the HTTP connection and returns three parameters: read, write, and_sid. The streams for receiving and sending data, plus a session ID. These streams are passed to client session, which manages the MCP protocol communication. We initialize the session with `await session.initialize()`. This performs the MCP handshake. We load the MCP tool using `load_MCP_tools` passing in the active session. We create a ReAct agent, passing in our LLM and the converted tools. Finally, we invoke the agent with a prompt. The agent can now discover and use our MCP tools to respond. The agent extracts the numbers from the text, in this case, 1 and 2. These integers are sent to the MCP server. The result is calculated and sent back to the agent. The LLM converts it into a natural language response. 

###### STDIO MCP Server
For the STDIO transport, the server runs on your own system. The client launches the server as a child process and communicates via `stdin/stdout` pipes, unlike HTTP, where you connect to an already running server. This is done via the Python file `stdio_server.py`. 
If you're using Jupyter, you'll need to save it as a `.py` file. The code is identical to before, except we add `mcp.run()` at the end to start the server on stdin slash stdout. Next, we'll configure the STDIO transport. To do this, we run the Python file `stdio_server.py`. Using STDIO with an LLM is identical to using HTTP, except for the transport configuration. However, instead of streamable HTTP underscore client, we use stdio underscore client with stdio server parameters to launch the server as a child process. The rest is the same. 


<!-- Go to `https://github.com/joshuazhou744/enhanced-mcp-server` -->


### MCP Client Architecture and Fundamentals

The host process orchestrates everything, managing client instances, enforcing security policies, and aggregating context from multiple servers. MCP Clients are protocol translators. Each client instance maintains a dedicated one-to-one connection with exactly one server. The key principle, if your application needs to connect to three servers, your host creates three separate client instances. Multiple applications can connect to the same server simultaneously. Each gets their own client-server connection. MCP servers expose specialized capabilities through three primitives. 
- Resources for data, 
- tools for action, and 
- prompts for templates. 
  
<p align="center">
  <img src="./assets/agentic_ai/mcp2.png" alt="drawing" width="600" height="250" style="center" />
</p>

###### JSON-RPC foundation
MCP is built on JSON RPC 2.0, providing three message types. 
- Requests expect responses. They have an ID and await a result or an error. 
- Responses contain either a result or an error object. 
- Notifications are fire and forget. They have no ID and no response is expected. 

This foundation enables bidirectional communication. Clients call server methods such as `list_tools` and `call_tool`, while servers can send sampling requests to clients. All messages are JSON objects transported over STDIO or streamable HTTP. 

###### MCP Client Connection: 3 phases
Every MCP connection follows three mandatory phases.  
- Phase 1: initialization:
    - The client sends an initialized request with protocol version and client capabilities. 
    - The server responds with its capabilities and protocol version. 
    - The client must send an initialized notification to complete the handshake. 
  This negotiation ensures both sides agree on supported features for the connection. 
- Phase 2: operation:
    -  Bidirectional communication begins. 
    -  Clients discover capabilities through `list_ tools`, `list_ resources`, and `list_ prompts`.
    -  Clients invoke tools, read resources, and get prompts. 
    -  Servers can request sampling during this phase. 
- Phase 3: shutdown: 
   - The client sends a shutdown request.
   -  The server responds, and then the connection closes with a cleanup process. 

During initialization, clients and servers declare their capabilities. 
- Client capabilities include experimental features, sampling support, and routes for file system access. 
- Server capabilities advertise available features, tools, resources, prompts, and whether they support list underscore changed notifications. 

Both sides examine capabilities to determine what operations are possible. For example, if a server doesn't advertise tools capability, the client knows not to call list underscore tools. This negotiation enables graceful feature degradation and protocol evolution. 

MCP defines three core primitives: 
- Tools are server-exposed functions that clients can invoke, such as `read_file`, `query_database`, or `fetch_API`. 
    - Servers define tool schemas with name, description, and input schema using JSON schema. 
- Resources are data sources with URIs. Servers expose file URIs that clients can read. Resources can be static or dynamically generated. 
- Prompts are reusable templates for LLM interactions. 
    - Servers define prompt templates with arguments and message structures. 
    - Client's call a get underscore prompt function with arguments to receive formatted messages ready for LLM consumption. 
- The STDIO transport is MCP's simplest transport mechanism. It's perfect for local development and command-line tools. The client spawns a server process and communicates through standard input and output. 
    - Messages are newline-delimited JSON. The server writes responses to stdout and error logs to stderr. The STDIO transport is process-based. When the client terminates, the server process ends. 
    - STDIO is secure for local-only scenarios. No network exposure, no authentication needed. It's the default transport for local MCP servers like filesystem, git, or database connectors
- Production clients typically connect to multiple servers simultaneously. The host process creates one client instance per server connection. Each client independently manages its own lifecycle, initialization, capability negotiation, and operation. The host aggregates capabilities across all connected servers, combining tool lists, resource URIs, and available prompts. When the LLM needs to call a tool, the host routes the request to the correct client based on tool name. 
  
- Both of the official MCP Software Development Kits, or SDKs, offer complete feature parity. TypeScript SDK uses Node.js with event-driven I.O., provides native HTTP and SSE support, integrates naturally with web frameworks, and includes the official reference implementation. Python SDK leverages AsyncIO for asynchronous operations, has excellent STDIO transport support, integrates with FastMCP for rapid server development, and works seamlessly with data science tools such as Jupyter Notebooks. Both SDKs are fully cross-compatible. Python clients work with TypeScript servers and vice versa. Therefore, you should choose which SDK to use based on your existing stack and deployment environment. You need to test your MCP client implementation systematically. Test servers are built into the official SDKs. Verify the initialization handshake completes with proper capability negotiation. Test all three phases – initialization, operation, and shutdown. Validate JSON RPC message formatting – all requests must have IDs, notifications must not. Verify transport layer, STDIO process management or HTTP connection handling. Test error handling for malformed responses and timeouts. Test capability detection – ensure your client respects server capabilities. And finally, use the MCP Inspector for debugging – it visualizes message flow and validates protocol compliance.


### Streambale HTTP, Roots and Sampling



MCP supports two transport mechanisms 
- STDIO transport spawns a local server subprocess and communicates through STDIN and STDOUT with newline-delimited JSON RPC messages. The server lifecycle is tied to the client process, perfect for local tools such as file system servers and Git interactions 
- Streamable HTTP transport enables remote servers over networks The client connects to an HTTP endpoint using bidirectional streaming The server runs independently and can handle multiple clients simultaneously, each with its own connection Streamable HTTP replaced the deprecated SSE transport in March 2025, offering full bidirectional communication through a single endpoint The key difference is that STDIO is process-local with low latency, whereas Streamable HTTP is network-remote and production-ready 

Here's a quick comparison of the two transports for reference 

Feature | STDIO | Streamable HTTP
------- | -------- | ---------
Scope | Local Process | Remote Work
Connection | stdin/stdout | Bidirectional HTTP
Lifecycle | Tied to client | Independent server
Use Case | Local tools, IDE | Cloud servers, multi-user
Authentication | None (local) | Required (network)

Streamable HTTP 
- uses a single endpoint with bidirectional communication. 
- The client connects to the server's `/mcp` endpoint, `fastmcp`'s default path. 
- Both client and server can send messages through this connection at any time 
- The client sends JSON RPC requests, and the server responds with results 
- The server can also initiate requests, such as sampling for LLM capabilities, and the client responds 
- Notifications flow in both directions without expecting responses 
- This bidirectional pattern mirrors STDIO's STDIN, STDOUT, but uses modern HTTP protocols with better reliability and resumability features 
- The key benefits of Streamable HTTP are It is a single endpoint for all communication 
- It uses fully bidirectional messaging 
- It provides built-in resumability support 
- It offers simple implementation 

Implementing Streamable HTTP in your MCP client is straightforward with the official SDK. You start by importing the Streamable HTTP underscore client from the MCP SDK Then you connect your server URL with the slash MCP endpoint, which is the default fastMCP convention The client then returns three values, readStream, writeStream, and sessionInfo You use these streams to create a client session with the MCP SDK's session manager The session handles all protocol details Initialization handshake, capability negotiation, message routing, and bidirectional communication You call initialize to negotiate capabilities and send the initialized notification and then you're ready for operation The client automatically handles message framing, error recovery, and connection management This code snippet shows the client implementation pattern 

Sampling is an MCP feature that lets servers request LLM capabilities from the client during tool execution The workflow is as follows 
1. The server executes a tool and determines it needs LLM reasoning, perhaps to analyze code, summarize findings, or generate content 
2. The server sends a sampling slash create message request with messages array, system prompt, model preferences, and generation parameters such as temperature and max tokens 
3. The client receives this request and prompts the user for approval This is critical for security and cost control 
4. Only after approval does the client invoke the LLM 
5. The LLM response flows back through the client to the server which incorporates it into the tool result 

Sampling is optional Clients declare support and capabilities during initialization Servers must handle cases where sampling isn't available.

#### Roots: MCP security boundaries

<p align="center">
  <img src="./assets/agentic_ai/mcp4.png" alt="drawing" width="600" height="400" style="center" />
</p>


- Roots are an MCP primitive for defining file system security boundaries 
- During capability negotiation, clients can advertise roots support 
- When a client supports roots, it declares specific URLs that define allowed file system paths such as these: 
    - `file:///home/user/projects`
    - `file:///var/app/data` 
- The MCP specification defines roots as the allowed directories for server file system access 
- Servers that respect roots will only access files within declared root paths 
- The client enforces these boundaries and before any file operation, it validates that the canonical path falls within a declared root 
    - Canonical path resolution is critical 
    - It can help resolve symlinks, normalize paths, and it checks the actual location to prevent traversal attacks such as accessing the `etc/password` folder directly 
- Roots implement the principle of least privilege meaning that servers only access what they explicitly need. 
    - Sandboxing for dev environments
    - Multitenant isolation
    - Clear security audit trails

#### Multi-transport session management
Production MCP clients often manage multi-server connections across both transports simultaneously. 
- The architecture uses a session manager that maintains registry mapping server identifiers to transport specific connections 
    - Each entry contains transport type, STDIO or streamable HTTP
    - Connection state (connecting / ready / error / closed)
    - Active session instance 
- Request router looks up server
- Retreives the corresposnsing session and dispatches request through appropriate transport
- Enables flexible deployment topologies:
    - All STDIO for local deployment
    - All streambale http for cloud deployment
    - Hybrid architecture (local and remote)
- Anstraction layer makes transport differences trnaparent to application logic 
  

#### MCP Security with Permissions and Elicitation

As AI systems progress from answering questions to executing tools and interacting with real systems, security becomes critical. 
- A single unchecked action can expose sensitive data or trigger unintended outcomes. 
- MCP addresses this by introducing structured controls that ensure that actions are intentional, visible, and accountable. 

MCP security relies on two mechanisms: 
- Permissions:  
  - client-side policies that control tool execution
  - determining which tools run automatically, require approval, or are blocked
  - MCP defines three policies:
      - Allow: executes a tool immediately without user interaction
      - Deny: Blocks the tool for dangerous or restricted operations 
      - Ask: Requires elicit user approval
      - These checks happen on the client before any server communication.
- Elicitation: is server-initiated structured input. 
    - When extra confirmation or data is needed, the server sends a JSON schema defining fields, types, and validation rules. 
    - The client presents this to the user, validates the input, and returns structured data. 

  <p align="center">
  <img src="./assets/agentic_ai/mcp5.png" alt="drawing" width="600" height="400" style="center" />
</p>

[Reference for this image](https://www.coursera.org/learn/build-ai-agents-using-mcp/lecture/mvJDU/mcp-security-with-permissions-and-elicitation)

###### Policies
Policies can be global per tool or argument-specific. For example, a client may allow reading test.txt but require approval for `production.yaml`. Policies are stored in a `permissions.json` file that users can edit, ensuring that security rules travel with the client rather than the server. 

Here is a simple guide for choosing the right permission policy. 
- Use allow for safe, read-only actions with no side effects, such as reading files or listing directories. 
- Use deny for dangerous, disabled, or unauthorized operations, such as deleting databases or exposing secrets. 
- Use ask for actions that modify data, cost money, or have side effects, such as writing files, sending emails, or making API calls. 

###### Permission enforement workflow
Permission enforcement happens entirely on the client before any server interaction. 
- The LLM decides to call a tool and generates the function call with arguments. 
   -  For allow policies, the client immediately calls the server tool and logs the operation.
    - For deny policies, the client returns an error to the LLM without contacting the server and logs the denial. 
    - For ask policies, the client prompts the user with full details, tool name, arguments, and risk level. 
  
###### Elicitation - Server initiated structure input
When a server needs additional information to complete an operation, it doesn't just ask a freeform question. It sends an elicitation request with a formal JSON schema defining exactly what data is needed. Once the user completes the form, the client validates the entire structure against the schema. Only valid data gets sent back to the server. The user can accept with data, decline the request, or cancel the operation entirely. 

Elicitation use cases include multi-step workflows,
destructive operations, missing parameters, compliance documentation, and security acknowledgments.

Schemas clearly define what information is needed and how it is validated, ensuring consistent behavior across environments. 

There are numerous benefits, including structured
validation, audit trail with context, reusability across servers, version control, and compliance readiness. 

Production MCP clients classify operations by risk.
Critical actions involve system-level control, such as executing commands or modifying security
settings. These default to deny unless explicitly overridden with multiple approval gates. High-risk actions are destructive, such as deleting files or dropping databases. These default to ask plus elicitation requiring confirmation with structured input. Medium-risk actions modify data, such as writing files or updating records. These default to ask requiring user approval. Low-risk actions are typically read-only, such as listing or reading files. These default to allow with minimal friction. 

Risk-based controls ensure that MCP clients apply stronger safeguards as operational risk
increases. This block of code highlights risk assessment logic. 

Audit logging is essential for security monitoring and compliance. Each permission decision is logged with a timestamp, tool name, arguments, applied policy, risk level, and outcome. Logs capture whether actions were automatic, required approval, approved, or denied, and whether execution succeeded. 

Elicitation logs include the schema and submitted data. Logs should be stored in append-only structured formats such as JSON lines with metadata such as user ID, session ID, and server identity. 

For compliance, audit logs prove what operations
occurred, who authorized them, and when. This includes incident investigation, security monitoring, compliance reports, and access review. This block of code highlights the audit log entry structure. When an AI assistant attempts a sensitive operation, the client's permission layer intercepts
it and assesses risk. For ask policies, the client prompts the user with full context, including risk level and tool details. The user can approve to proceed, deny to cancel, or request more information. If approved, the client may trigger server elicitation for additional confirmation. For example, deleting a file triggers elicitation requiring the user to type the file name and reason. The client validates this input and sends it to the server. Effective permission management relies on clear security patterns. Apply least privilege by starting with deny all and allowing only required tools. Grant permissions gradually as needed. Use environment-specific policies. Development can allow most tools. Staging should require approval. And production enforces deny by default with audited exceptions. Apply user-based policies so that different users receive appropriate permissions. 

Temporary
permissions can be granted for tasks and revoked automatically. Support inheritance through base profiles and role-based templates such as Reader, Editor, and Admin. Store permissions.json in version control to track changes and enable rollback. 


### Cheat Sheet: MCP Hosts and Clients


This module focused on building and securing MCP Clients. Use this cheat sheet to quickly review the concepts, transport methods, and advanced security patterns you implemented.

#### MCP client architecture

##### Base/derived pattern

The base/derived pattern separates low-level MCP protocol handling from application-specific logic. The base client manages connections, sessions, and server communication, while derived classes add UI or business logic. This architecture promotes code reuse and makes it easy to build multiple client applications (CLI, GUI, web) that share the same protocol implementation.

```python
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
class MCPBaseClient:
    def __init__(self, server_script: str):
        self.session = None
        self.exit_stack = AsyncExitStack()
        self._connected = False
    async def connect(self):
        if self._connected:
            return
        server_params = StdioServerParameters(command="python", args=[self.server_script])
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        read, write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
        self._connected = True
    async def list_tools(self):
        await self.connect()
        return (await self.session.list_tools()).tools
    async def call_tool(self, tool_name: str, arguments: dict):
        await self.connect()
        return await self.session.call_tool(tool_name, arguments)
# Derived application inherits all protocol methods
class MCPGUIApp(MCPBaseClient):
    async def gui_list_tools(self):
        tools = await self.list_tools()
        return "\n".join([f"• {t.name}: {t.description}" for t in tools])
```

Key Benefits: The `AsyncExitStack` ensures proper cleanup of connections, `_connected` flag enables lazy initialization (connect only when needed), and derived classes inherit all protocol methods automatically without reimplementing connection logic.

##### Server-initiated operations

MCP allows servers to initiate requests back to the client, enabling advanced capabilities such as filesystem sandboxing, AI completions, and structured user input.

##### Roots (filesystem security)

Roots define trusted base directories for file operations. By validating all file paths against these roots, you prevent path traversal attacks where malicious code could access files outside the intended workspace using patterns such as ../../etc/passwd.

```python
from pathlib import Path
BASE_DIR = Path(__file__).parent / "workspace"
def is_within_roots(path: Path) -> bool:
    try:
        path.resolve().relative_to(BASE_DIR.resolve())
        return True
    except ValueError:
        return False
@mcp.tool()
def read_file(filepath: str) -> str:
    path = BASE_DIR / filepath
    if not is_within_roots(path):
        return "Error: Access denied"
    return path.read_text()
```

##### Sampling

Sampling allows the MCP server to request LLM completions from the client's AI model. This enables the server to leverage AI capabilities without direct model access.

Security critical: Always require explicit human approval before executing sampling requests, as they can trigger arbitrary prompts.

##### Elicitation

Elicitation requests structured user input from the client using Pydantic schemas. Unlike simple text prompts, elicitation enforces type validation and ensures responses match expected formats. This is useful for operations requiring explicit user confirmation, such as destructive actions or sensitive operations.

```python
from fastmcp import Context
from pydantic import BaseModel
class ApprovalSchema(BaseModel):
    approved: bool
    reason: str
@mcp.tool()
async def delete_file(ctx: Context, filepath: str) -> str:
    response = await ctx.elicit(
        message=f"Delete {filepath}?",
        response_type=ApprovalSchema
    )
    if not response.approved:
        return f"Cancelled: {response.reason}"
    Path(filepath).unlink()
    return "Deleted"
```

##### Transport methods

MCP supports multiple transport mechanisms for client-server communication. Choose yours based on your deployment model: STDIO for local development and subprocess communication, and HTTP for remote servers and production deployments.

##### STDIO (local)

STDIO transport launches the server as a subprocess and communicates via standard input/output streams. This is ideal for local development, testing, and scenarios where the server runs on the same machine as the client.

```python
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters
server_params = StdioServerParameters(command="python", args=["server.py"])
async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
```

##### HTTP (remote)

HTTP/Streamable HTTP transport enables network communication between client and server. Use this for production deployments where the server runs remotely, microservices architectures, or when you need to expose your MCP server as a web service accessible to multiple clients.


```python
# Server
from fastmcp import FastMCP
mcp = FastMCP("HTTP Server")
mcp.run(transport="http", host="127.0.0.1", port=8000)
# Client
from mcp.client.streamable_http import streamablehttp_client
async with streamablehttp_client("http://127.0.0.1:8000/mcp") as (read, write, _):
    async with ClientSession(read, write) as session:
        await session.initialize()
```

##### Security patterns

Building secure MCP clients requires multiple defense layers: permission policies control which operations are allowed, audit logging tracks all actions for compliance, and human-in-the-loop approvals prevent unauthorized operations.

##### Permission policies

Permission policies implement a three-tier authorization model: allow (auto-approve), ask (require confirmation), and deny (block completely). This gives you fine-grained control over which tools can execute automatically versus which need user approval.

```python
class MCPPermissionClient:
    def __init__(self):
        self.permissions = {
            "read_file": "allow",
            "write_file": "ask",
            "delete_file": "deny"
        }
    def check_permission(self, tool_name: str) -> str:
        return self.permissions.get(tool_name, "ask")
    async def call_tool_with_permission(self, tool_name: str, args: dict, approved=False):
        permission = self.check_permission(tool_name)
        if permission == "deny":
            return "Permission denied"
        if permission == "ask" and not approved:
            return "Approval required"
        return await self.session.call_tool(tool_name, args)
```

##### Audit logging

Audit logs create an immutable record of all tool executions and authorization decisions. This is essential for security monitoring, compliance requirements, debugging issues, and understanding system behavior in production environments.

```python
def log_audit(operation: str, decision: str):
    log_entry = f"[{datetime.now().isoformat()}] {operation} - {decision}\n"
    with open("audit.log", "a") as f:
        f.write(log_entry)
```

##### AI host integration

AI host applications act as orchestrators that connect LLMs with MCP servers. The host translates between the LLM's tool-calling format (e.g., OpenAI function calling) and MCP's protocol, enabling the LLM to discover and execute MCP tools dynamically during conversations.

##### LLM tool calling

This pattern converts MCP tools into OpenAI function calling format, enabling the LLM to select and execute tools based on user queries. The host manages the conversation loop: sending messages to the LLM, detecting tool calls, executing them via MCP, and feeding results back to continue the conversation.


```python
from openai import OpenAI
class MCPHostApp(MCPBaseClient):
    def __init__(self, server_script: str):
        super().__init__(server_script)
        self.llm_client = OpenAI()
    async def get_available_tools(self):
        return [{
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.inputSchema
            }
        } for t in await self.list_tools()]
    async def chat(self, user_message: str):
        self.conversation_history.append({"role": "user", "content": user_message})
        response = self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.conversation_history,
            tools=await self.get_available_tools()
        )
        if response.choices[0].message.tool_calls:
            for tool_call in response.choices[0].message.tool_calls:
                result = await self.call_tool(
                    tool_call.function.name,
                    json.loads(tool_call.function.arguments)
                )
                self.conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })
            return self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=self.conversation_history
            ).choices[0].message.content
        return response.choices[0].message.content
```

##### Synthetic tools

Synthetic tools expose MCP primitives (resources, prompts) as callable LLM functions. This allows the LLM to access non-tool MCP features through the same unified tool-calling interface, making resources and prompts first-class operations in the conversation flow.

```python
async def get_available_tools(self):
    tools = [convert_to_openai(t) for t in await self.list_tools()]
    tools.append({
        "type": "function",
        "function": {
            "name": "mcp_read_resource",
            "description": "Read resource by URI",
            "parameters": {"type": "object", "properties": {"uri": {"type": "string"}}}
        }
    })
    return tools
```

##### Best practices

###### Client design:

- Use the base/derived pattern to separate protocol logic from application logic.
- Implement lazy initialization with connection state tracking.
- Always use AsyncExitStack for proper async resource cleanup, preventing connection leaks.

###### Security:

- Always validate file paths against roots to prevent traversal attacks.
- Implement explicit permission policies for all sensitive operations.
- Maintain comprehensive audit logs for security monitoring.
- Use human-in-the-loop approval for high-risk operations like sampling or deletions.

###### Transport:

- Use STDIO for local development, testing, and single-machine deployments.
- Switch to Streamable HTTP for production environments, remote servers, or when you need to serve multiple clients simultaneously.

###### LLM integration:

- Convert MCP tools to OpenAI function schemas for compatibility with popular LLMs.
- Use synthetic tools to expose resources and prompts as callable functions, giving LLMs full access to all MCP capabilities.

### END


# This is h1
## This is h2
### This is h3
#### This is h4
##### This is h5
###### This is h6



Notes from LangChain Official Documentation

LangGraph FAQ

- Do I need to use LangChain to use LangGraph? What’s the difference?
    -  No. LangGraph is an orchestration framework for complex agentic systems and is more low-level and controllable than LangChain agents. On the other hand, LangChain provides a standard interface to interact with models and other components, useful for straight-forward chains and retrieval flows.
How is LangGraph different from other agent frameworks?
Other agentic frameworks can work for simple, generic tasks but fall short for complex tasks bespoke to a company’s needs. LangGraph provides a more expressive framework to handle companies’ unique tasks without restricting users to a single black-box cognitive architecture.
Does LangGraph impact the performance of my app?
LangGraph will not add any overhead to your code and is specifically designed with streaming workflows in mind.
Is LangGraph open source? Is it free?
Yes. LangGraph is an MIT-licensed open-source library and is free to use.
Is LangSmith Deployment (formerly LangGraph Platform/Cloud) open source?
No. LangSmith Deployment is proprietary software that will eventually be a paid service for certain tiers of usage. We will always give ample notice before charging for a service and reward our early adopters with preferential pricing.
How do I enable LangSmith Deployment?
All LangSmith users on Plus and Enterprise plans can access LangSmith Deployment. Check out the docs.
How are LangGraph and LangSmith Deployment different?
LangGraph is a stateful, orchestration framework that brings added control to agent workflows. LangSmith Deployment is a service for deploying and scaling LangGraph applications.
How does LangGraph fit into the LangChain ecosystem? 
Our open source frameworks help you build agents:
LangChain helps you quickly get started building agents, with any model provider of your choice.
LangGraph allows you to control every step of your custom agent with low-level orchestration, memory, and human-in-the-loop support. You can manage long-running tasks with durable execution.
LangSmith is a platform that helps AI teams use live production data for continuous testing and improvement. LangSmith provides:
Observability to see exactly how your agent thinks and acts with detailed tracing and aggregate trend metrics.
Evaluation to test and score agent behavior on production data and offline datasets for continuous improvement.
Deployment to ship your agent in one click, using scalable infrastructure built for long-running tasks.


### Motivation

A solitary LLM is fairly limited
- it doesnt have access to tools, external context, multi-step workflows
- it cant alone perform multi-stpe workflows

So many LLM applications use a control flow with steps before and after LLM calls (tool calls, retreival steps).

This control flow forms a **chain** which are very reliable: the same chain of steps occurs every time you invoke the workflow. But we do want LLM systems to pick their own control flow for certain kinds of problems. You might want a LLM application that can choose its own set of steps depending on the problems they face. This is really what an agent is. An angent is a control flow defined by an LLM. So you have chains that are fixed control flows versus the control flows defined by LLMs.

There are many kind of agents depneding on the amount of control given to LLMs. For example a router can be though of as a low control agent where LLM controls a single step in a flow where it might choose between a narrow set of options. For example, i go from step 1 to step 2 or 3 based on the LM decision. On the other extreme, we have a fully autonomous  agent that can take any sequence of steps through some set of given options or even it can generate its own next move based on some potentially avialbel resources.

 <p align="center">
  <img src="./assets/agentic_ai/agentic4.png" alt="drawing" width="600" height="400" style="center" />
</p>

However there are some practical challenges. As you increase the level of contorl given to LLM, **application reliability** might decrease. LangGraph is aimed to help you increase the reliability here, allowing you to build agents that maintain the reliability even as you push out the level of control to LLMs. In short, LangGraph balances reliability with control. 

In many application we want to combine developer intuituion with LLM control. LangGraph expressses custom control flows as graphs which contain nodes (steps in your application such as tool call, retreival) and edges are connectivity between nodes. 

<p align="center">
  <img src="./assets/agentic_ai/agentic5.png" alt="drawing" width="600" height="400" style="center" />
</p>

There are few specific pillars for LangGraph to help you acheive  this goal that we are talking about: 
- Persistent:
- Streaming:
- Human-in-the-loop: 
- controlability

LangGraph plays very nicely naturally with LangChain. We often use the LangChain components in our LangGraph workflows. For example in a RAG application where you have a retreiver step from a vector store and an LLM step that takes the retreived documents and answers the questions. The retriever here could use a Langchain vector storage. Likewise the LLM node can a LanChain integration but You dont have to use LangChain in any of these cases. The main difference is that LangChain is built for linear, sequential workflows (chains), while LangGraph is designed for cyclical, stateful workflows (graphs). 

LangGraph is an extension of LangChain, not a replacement. You often use LangChain’s components (like model wrappers and tool integrations) inside a LangGraph workflow.

Why we still need both:
- Foundation vs. Orchestration: LangChain provides the essential "building blocks"—like prompt templates, document loaders, and vector store connectors. LangGraph is the "orchestrator" that arranges these blocks into complex, repeating cycles. You actually use LangChain components inside LangGraph nodes.
- Simplicity vs. Control: For a simple task like summarizing a PDF, using LangGraph is "over-engineering." LangChain’s linear chains (LCEL) are much faster to write, easier to read, and have less "boilerplate" code for basic jobs.
- The "Glue Code" Problem: Before LangGraph, developers used LangChain but had to write messy "if/else" loops in raw Python to handle retries or decisions. LangGraph was created to replace that "glue code" with a structured state machine that is easier to debug and visualize.
- Production Reliability: LangGraph adds specialized "production" features that LangChain doesn't focus on, such as Checkpointers (which let you save an agent's progress to a database so it can resume after a crash) and Human-in-the-loop (pausing the AI to wait for a human's approval). 

##### The typical workflow
Most developers don't choose one over the other; they follow a path: 
- Start with LangChain to quickly prototype prompts and data retrieval.
- Move to LangGraph when the app needs to "think," self-correct, or handle multi-turn conversations that aren't just a straight line. 

These guides compare LangChain and LangGraph, focusing on why LangGraph was developed to handle complex control flow, statefulness, and event-driven execution beyond LangChain's linear chains.


#### When to Use LangChain
- Simple Pipelines: Use it when your task follows a clear A → B → C order, such as summarizing a document or a basic RAG (Retrieval-Augmented Generation) pipeline.
- Prototyping: It is excellent for quickly connecting an LLM to a data source using pre-built components like prompt templates and output parsers.
- Stateless Apps: Ideal for apps that don't need to "remember" complex logic or backtrack based on errors. 

##### When to Use LangGraph
- Agentic Workflows: If your AI needs to make a decision, use a tool, check the result, and potentially loop back to try again, LangGraph is the better tool.
- Multi-Agent Systems: Use it to orchestrate multiple specialized agents (e.g., one researcher agent and one writer agent) that need to hand off tasks to each other.
- Human-in-the-loop: It has native support for pausing an agent to wait for human approval or input before continuing a cycle.
- Persistence: It easily maintains state across sessions, allowing you to "save" and "resume" an agent's progress.



LangGraph API creates a API around our graph so we can call it to execute our command related to the graph. For example: client.runs.stream(...)

Its is like a FastAPI wrapper around our graph... is this correct?
Spot on. That is exactly what it is—a specialized FastAPI-style wrapper designed specifically for the unique needs of stateful agents.
While you could write your own FastAPI wrapper, the LangGraph API (via the LangGraph Platform) gives you several "production" features out of the box that are surprisingly hard to code from scratch:
1. Built-in Persistence (The "Checkpointer")
In a standard FastAPI app, if you want to resume a conversation, you have to manually save the state to a database (Postgres/Redis) and load it back up.
LangGraph API: Automatically manages a Postgres database for you. When you call a thread_id, it handles all the "Select/Update" logic behind the scenes.
2. Long-Running Task Management
Standard FastAPI requests often timeout after 30–60 seconds.
LangGraph API: It uses a Background Task Queue. When you start a run, it can take minutes (or hours if waiting for a human). The API handles the "is it done yet?" polling and background execution for you.
3. Double-Texting & Interrupts
What happens if a user sends a second message while your LLM is still generating the first one?
Standard FastAPI: You’d likely have two conflicting processes fighting over the same state.
LangGraph API: It has built-in concurrency control. You can choose to "reject," "enqueue," or "interrupt" the previous run when a new one comes in.
4. Human-in-the-Loop Endpoints
The API provides specific endpoints just for the "resume" logic we struggled with earlier.
Instead of you manually fixing the state in code, the API has a threads.state.update endpoint and a runs.resume command specifically for moving past those breakpoints.
5. Streaming by Default
Streaming token-by-token or node-by-node (the stream_mode you used) is tricky to implement reliably over Server-Sent Events (SSE). The LangGraph API handles the SSE protocol for you so your frontend gets smooth updates.
The Verdict: If you are building a simple "one-and-done" bot, FastAPI is fine. If you are building a multi-step agent that needs to remember where it was, wait for humans, and handle complex state updates, the LangGraph API saves you weeks of infrastructure work.
Do you want to see how the Client SDK (the client.runs.stream part) looks when you actually connect it to a frontend?



 In LangGraph, a "thread" is a logical ID, not a physical Linux/OS thread.
Here is the breakdown of the difference:
1. It’s a "Database Key," not a "CPU Thread"
In a Linux process, a thread is a unit of execution that shares memory and runs on a CPU core.
In LangGraph, a thread is just a unique identifier (like a UUID or a string) used to look up a conversation's history in a database.
Linux Thread: Managing instructions for the processor.
LangGraph Thread: Mapping a user to their specific "Save Game" file.
2. State vs. Execution
State (LangGraph): When you send a thread_id, LangGraph goes to your Postgres/Sqlite database, pulls out the last known state (the "checkpoint"), and hands it to the AI.
Execution (Linux): Your Python code (the LangGraph library itself) runs on standard OS threads to process that data.
3. Why call it a "Thread"?
The name comes from "Conversation Threading" (like in an email chain or a Slack reply). It represents a linear sequence of events.
