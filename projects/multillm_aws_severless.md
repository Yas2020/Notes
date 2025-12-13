# THE MASTER PLAN FOR YOUR MULTI-LLM SERVERLESS CHATBOT PROJECT
You will create two documents:
- Doc A — Deep Notes (4–6 pages)
This is for your own understanding. This includes:
  - Architecture
  - Why each AWS service
  - Tradeoffs
  - Alternatives
  - Scaling scenarios
  - Failure cases
  - SQS patterns
  - Step Functions retries
  - Data pipeline details
  - Evaluation logic
  - Cost optimization
  - Security decisions
  - Multi-LLM orchestration logic

This is where you think, reason, ask questions, explore alternatives, and write raw notes. Nobody sees this. It’s your brain backup.

- Doc B — Interview Version (1 page)
This is the simplified, polished version of the project you will use:
- on calls
- in behavioral answers
- in system design interviews
This includes:
- 30-second summary
- 2-minute summary
- 1 diagram
- 5 key technical decisions
- 3 challenges + solutions
- 3 security highlights
- 3 cost optimizations
- 3 failure modes
- “What I’d improve next”
You can only make Doc B once Doc A becomes solid.

🔥 NOW THE PART THAT WILL MAKE IT EASY
I will give you the exact structure for Doc A (your heavy notes).
Once you fill this out once, all other projects become 2–3× easier.

#### 📘 DOC A — DEEP NOTES TEMPLATE (for your multi-LLM serverless RAG chatbot)
(Copy this into your notes. We’ll fill it section by section.)
1. Problem Definition (1 paragraph)
What the system must do
What constraints
What "good" looks like (latency, scalability, cost, reliability)
Why serverless
1. High-Level Architecture (paragraph + bullets)
List main components:
Step Functions orchestration
API Gateway entry
Lambda business logic
SQS for decoupling
Aurora (or DynamoDB) for metadata
S3 for documents
LangChain / vector DB
LLMs (Bedrock, OpenAI, etc.)
Reranking + hybrid retrieval
Cross-model agreement scoring
Error and retry patterns
1. Detailed Component Breakdown
For each (Lambda, SQS, Step Functions, Aurora, etc.):
What it does
Why you chose it
Alternatives you considered
Tradeoffs
Example:
Why Step Functions?
Visual orchestration
Built-in retries
Parallel LLM calls
Cheaper than ECS
Alternatives:
EventBridge → worse control
ECS → expensive, harder scaling
1. Deep Dive: Multi-LLM Orchestration Logic
This is the star of the project.
Explain:
Why multiple LLMs
How you orchestrated them
When one LLM is preferred
How cross-model agreement works
Factuality checks
Why hybrid retrieval improves consistency
How you evaluate answer quality
1. Retrieval Layer (RAG)
Explain:
Store type (FAISS? Aurora? Dynamo? OpenSearch?)
Hybrid retrieval: keyword + embedding
Reranking (colBERT? LLM-as-reranker?)
Chunking strategies
Token budgeting
Caching strategies
1. Data Pipeline
How documents are ingested
How transformations happen
How embeddings generated
How invalid documents handled
How updates propagate
How reindexing works
1. Error Handling + Fault Tolerance
This is where you shine.
Cover:
SQS DLQs
Step Functions retry policies (exponential backoff)
Lambda timeouts
Throttling / concurrency
Circuit breakers for LLM APIs
Retry idempotency
1. Scaling Strategy
Explain:
Why serverless handles auto-scaling
How concurrency is controlled
Spiky workloads
How queues smooth traffic
Bottlenecks (LLM API rate limits)
Horizontal vs vertical scaling
1. Cost Optimization
Talk about:
Why Lambdas optimal
Bedrock model selection tradeoffs price vs quality
Tiered storage (S3 vs Aurora)
Pre-warm strategies
Avoiding duplicate LLM calls through caching
CloudWatch log retention limits
1.  Security + IAM
This is big.
Cover:
OAuth2 + Cognito
JWT validation
Step Functions roles
Lambda least privilege
Aurora credentials via Secrets Manager
VPC setup (if applicable)
Network boundaries
Data encryption (KMS keys)
1.  Monitoring + Observability
Include:
CloudWatch metrics for latency, errors
X-Ray traces
SQS queue depth
API Gateway logs
LLM latency vs retry rate
Dynamo/Aurora QPS
Dashboards
1.  Failure Modes You Designed For
List failures like:
LLM timeout
Hallucination
Response inconsistency
Retrieval giving wrong context
Lambda cold starts
SQS backlog buildup
Aurora connection storms
For each:
→ What happens?
→ How do you detect it?
→ How do you recover?
1.  What You Improved vs. the Original Repo
This section helps you claim ownership.
Examples:
Added CDK infra
Rewrote API layer
Added CI/CD
Automated rollback using CloudWatch alarms
Added evaluation logic
Added cross-model agreement
Added reranker
Added Step Functions orchestration improvements
1.  What You Would Improve If You Had More Time
This shows senior-level thinking.
Ideas:
Switch to Bedrock Agents or Knowledge Bases
Use vector DB with MMR
Add async concurrency for LLM calls
Add continuous evaluation
Introduce semantic caching
Add Guardrails / Hallucination filters

## SECTION 1 — Problem Definition (FINAL VERSION)
#### 1.1 Project Goal
The goal of this project was to design and deploy a **real**, **production-grade cloud architecture for an LLM-powered chatbot**. Although the system was built as an independent, self-directed initiative, the intention was to learn and practice **true enterprise-level ML/LLM infrastructure**, including:
- Deploying retrieval-augmented generation (RAG) in a cloud environment
- Using serverless and managed AWS components to build scalable pipelines
- Integrating multiple LLM providers and understanding their trade-offs
- Learning infrastructure-as-code (CDK), provisioning cloud services, and understanding architectural patterns
- Getting hands-on experience with system design, orchestration, routing, security, and AWS services
- Creating a realistic playground that behaves like a production LLM backend

The project was not a toy example — it was built intentionally as a training ground to simulate what ML/LLM engineers do on real production systems, including architecture, cloud engineering, deployment, and experimentation with retrieval and model orchestration.

#### 1.2 Constraints
The system was designed under several practical and architectural constraints:
- Low cost — fully serverless or managed components only; no persistent servers
- Auto-scaling and high availability — must scale on demand and remain reliable without manual maintenance
- Multi-model support — ability to easily switch between LLM providers (Bedrock, OpenAI, local models on SageMaker, etc.)
- CDK-driven IaC — entire infrastructure must be reproducible, deployable, and version-controlled via AWS CDK
- Minimal maintenance — operational simplicity was a priority; monitoring and logging should be built-in
- No strict latency requirements — goal was architectural learning, not optimization
- Single-tenant scope — the project explored system design, RAG, orchestration, and deployment, not full multi-tenant isolation
- No built-in LLM evaluation logic (yet) — planned as a future extension, allowing experimentation with hallucination detection and multi-model agreement

These constraints shaped the components chosen and the final system design.

#### 1.3 Definition of Success
Success for this system was defined by whether it behaves like a real, practical, cloud-ready LLM backend, not by dataset metrics or model performance alone. Specifically:
- Reliable RAG retrieval that supports complex questions and produces stable, high-quality responses
- Secure API access, including OAuth2/JWT and best practices for endpoint protection
- A fully serverless, auto-scaling architecture that is inexpensive, production-like, and easy to maintain
- Infrastructure that can be deployed end-to-end using CDK, with repeatability and clean cloud provisioning
- Ability to run multiple LLMs or switch between providers for experimentation or cost/performance tuning
- Ease of iteration, allowing new components (evaluation, reranking, multi-model routing) to be added incrementally
- Operational readiness, including logging, monitoring, tracing, and fault tolerance
If the system was cheap, scalable, secure, reproducible, and flexible enough for real-world experimentation, the project was considered successful.

This section will make your project sound like a real production-grade backend that you architected intentionally, instead of a “learning project.”
## ✅ SECTION 2 — System Requirements (FINAL VERSION)
#### 2.1 Intended Users
The primary user of the system was myself, as the project was built to simulate a real-world LLM production backend and to explore architectural patterns, orchestration, and AWS-managed services. However, the codebase and architecture were designed intentionally so that:
- ML engineers could reuse components (RAG pipeline, Step Functions orchestration, LLM routing, ingestion pipeline).
- Developers could use it as a template to integrate multiple LLM providers or deploy serverless RAG backends.
- Future extensions (multi-tenant, evaluation layer, monitoring, hybrid retrieval) could be added without redesign.
So although the target user was originally me, the design follows production patterns suitable for professional teams.

#### 2.2 Functional Requirements
The system implemented the following functional capabilities:
- **Core Chat + RAG Flow**
  - User query → Retrieval → LLM-generated answer (full RAG flow)
  - Swappable LLM model providers (Bedrock models, OpenAI, or SageMaker-hosted models)
  - Pluggable vector databases
    - DynamoDB (key-value store with embeddings)
    - Kendra search
    - Any vector DB from LangChain integrations

- **Document Processing & Ingestion**
  - Document ingestion pipeline
  - Chunking, embedding, and metadata preparation
  - Embedding storage in configurable vector DB
  - Automatic updates of retrieval index after ingestion

- **Orchestration & Messaging**
  - Serverless orchestration with AWS Step Functions
  - API entrypoint using API Gateway + Lambda
  - WebSocket API to maintain a continuous chat stream
  - SQS for message decoupling and buffering
  - SNS for notifications and fan-out messaging

- **Security & Authentication**
  - Full authentication using Cognito
  - Frontend authenticated via AWS Amplify and connected to backend authorizers
  - Lambda authorizer for fine-grained API access control

- **Frontend**
  - React-based frontend for chat UI
  - Clean integration with backend endpoints and WebSocket channels

- **Infrastructure-as-Code**
  - Entire system provisioned using AWS CDK
  - Reproducible deployments across environments
  - One-command deployment with clear separation of stacks
- **What Was Not Implemented Yet**
  - To keep the architecture lightweight:
    - No central logging/monitoring layer (CloudWatch default only)
    - No advanced error-handling, retries, or custom failure recovery
    - No evaluation layer or hallucination detection (planned as extension)
These omissions are important because they show you understand production features, even if the MVP did not require them.

#### 2.3 Non-Functional Requirements
The architectural principles guiding the system were:
- Scalability
    - Fully serverless components that scale automatically
    - No long-running servers or fixed capacity
    - Event-driven messaging (SQS/SNS) for balanced workloads
  - Low Cost
    - Pay-per-use compute (Lambda, Step Functions)
    - No GPU instances unless explicitly required
    - Cheap storage and vector DB options (DynamoDB)
- Reliability
  - Multi-service architecture with managed AWS components
  - Decoupled compute (SQS) to handle bursts
  - Stateless Lambdas for resiliency
- Maintainability
  - Clear CDK constructs
  - Separation of frontend, backend, RAG pipeline, and ingestion pipeline
  - Component-level abstraction for easy modification (swap model, DB, embedder)
- Version-Controlled Infrastructure
  - CDK code stored in Git
  - Reproducible deployments
  - Infrastructure changes traceable via Git history


-----------------------------------------------------
CDK Stack:
- **Shared**
    - Inititalize VPC: number of nat gateways needed, configure subnets (public, private, isolated)
    - Create gateway endpoints and interface endpoints for S3, DynamoDB - only interface endpoints for Sagemaker, SecretManager
    - Place the app config file `config.json` as Parameter String in System Manager
    - Create Lambda layers for powerTools Python (AWS native Python module to help with logging, tracing serverless Lambda), Python SDK/genai_core (containing our utility Python code related Aurora(create, delete,etc), LangChain(chucnking, embeddings, semantic search), cross-encoder, upload documents) etc. Becaue these are performed by AWS Lambda
    - Create secret manager secret to prevent from direct contact with ApiGateway endpoints (only allows requests fron CDN)
    - Create sectet for external API keys, for example for OpenAI key
- **Authentication**
    - Create Cognito user pool with a client
- **SageMaker**: 3 options for deploying LLMs
    - Create IAM Service Principal role with SageMaker full access
    - 3 options to deploy LLMs:
        - Provide your container image ID in AWS ECR or use HuggingFace TGI Inference containers in ECR given the closest region. FalconLite is deployed on `ml.g5.12xlarge`  using this method.
        - Provide ARN of the image (AWS JumpStart) and model ID. LLamaV2_13B_Base and LLamaV2_13B_Chat are deployed using this method.
        - Package your local HuggingFace models give their path, S3 build bucket and thier model ID. A AWS CodeBuild project downloads a snapshot of every model from HuggingFace Hub, compress it and then upload it to S3. A Lambda function invokes this CodeBuild project on event, waits for it to complete and then reports if it succeded on completion. This Lambda function itsself is invoked by Custom Resource Provider.  After this completes, a SageMaker CdfnModel deploys this model in HF TGI container given the `modelDataUrl` as an S3 bucket.   
    - Return Sagemaker model endpoint

-  **RAG System**: 
   -  Two DynamoDB tables: `workspace` and `document` tables to store metadata. 
      -  workspace table:
         -  PartitionKey: `workspace_id`
         -  SortKey: `object_type`
      - document table:
        -  PartitionKey: `workspace_id`
         -  SortKey: `document_id` 
       -  Both have global secondary tables:
            - workspace table:
              -  PartitionKey: `object_type`
              -  SortKey: `created_at`
            - document table:
              -  PartitionKey: `workspace_id`
               -  SortKey: `compound_sort_key`
    - **SageMaker RAG Models**:
      -  Find Embedding Models and Cross-Embedding Models Name/Ids from Config.json of the whole app
      -  Deploy them to SageMaker like other LLMs described before using one of the 3 options on `ml.g4dn.xlarge` machines. In this case, it was deployed using custom script HuggingFace model that loads the models from HF Hub, the script manupulate the prompt (adds "query:" to it for example). It directly tokenizes iput, pases it throught the model, mean pool of input and output decoded to become response using `with torch.inference_mode()` clause. For cross-encoders, it returns a list of scores. This is a custome inference accept by HF if its put in a file called `inference.py` and possible `requirements.txt` file.
    - **Vector DBs**:
      - Aroura-pgvector
      - OpenSearch 