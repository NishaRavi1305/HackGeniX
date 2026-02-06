# End-to-End Data Flow Architecture

## High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT APPLICATION                                  │
│                    (Provides JWT tokens for authentication)                      │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              FASTAPI APPLICATION                                 │
│                          src/main.py (Entry Point)                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  Auth Middleware → JWT Validation → Role/Permission Check               │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        ▼                               ▼                               ▼
┌───────────────┐              ┌───────────────┐              ┌───────────────┐
│   /documents  │              │  /interviews  │              │   /sessions   │
│   /questions  │              │   /reports    │              │    /voice     │
└───────────────┘              └───────────────┘              └───────────────┘
        │                               │                               │
        └───────────────────────────────┼───────────────────────────────┘
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              SERVICES LAYER                                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐              │
│  │ DocumentProcessor│  │InterviewOrchest- │  │  PDFReportGen    │              │
│  │  (parsing)       │  │  rator (central) │  │   (ReportLab)    │              │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐              │
│  │ QuestionGenerator│  │ AnswerEvaluator  │  │ SemanticMatcher  │              │
│  │  (hybrid mode)   │  │  (LLM-as-Judge)  │  │(sentence-transformers)          │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘              │
│  ┌──────────────────┐  ┌──────────────────┐                                    │
│  │ QuestionBankSvc  │  │ HybridSelector   │                                    │
│  │  (JSONL loader)  │  │ (70% bank/30% LLM)│                                   │
│  └──────────────────┘  └──────────────────┘                                    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        ▼                               ▼                               ▼
┌───────────────┐              ┌───────────────┐              ┌───────────────┐
│   MongoDB     │              │   S3/MinIO    │              │ LLM Providers │
│  (Motor)      │              │  (Storage)    │              │ (vLLM/Ollama) │
└───────────────┘              └───────────────┘              └───────────────┘
```

---

## Detailed Data Flow by Phase

### Phase 1: Document Upload & Parsing

```
Client                    API                     Services                  Storage
  │                        │                          │                        │
  │─── POST /documents/resumes ────────────────────────────────────────────────▶│
  │    (PDF/DOCX file)     │                          │                    S3: resumes/
  │                        │                          │                        │
  │                        │─── DocumentProcessor ────│                        │
  │                        │    ├─ Extract text (PyMuPDF/python-docx)          │
  │                        │    ├─ NER extraction (spaCy)                      │
  │                        │    └─ LLM enrichment (skills, experience)   ─────▶│ LLM
  │                        │                          │                        │
  │                        │─────────────────────────▶│ MongoDB: resumes       │
  │◀─── ParsedResume ──────│                          │                        │
```

```
Client                    API                     Services                  Storage
  │                        │                          │                        │
  │─── POST /documents/job-descriptions ───────────────────────────────────────│
  │    (JSON or PDF)       │                          │                        │
  │                        │─── DocumentProcessor ────│                        │
  │                        │    └─ LLM parsing (requirements, skills)    ─────▶│ LLM
  │                        │                          │                        │
  │                        │─────────────────────────▶│ MongoDB: job_descriptions
  │◀─ ParsedJobDescription─│                          │                        │
```

---

### Phase 2: Resume-JD Matching

```
Client                    API                     Services                  Storage
  │                        │                          │                        │
  │─── POST /documents/match ──────────────────────────────────────────────────│
  │    {resume_id, jd_id}  │                          │                        │
  │                        │─── SemanticMatcher ──────│                        │
  │                        │    ├─ Load resume + JD from MongoDB ◀─────────────│
  │                        │    ├─ Encode with sentence-transformers           │
  │                        │    ├─ Cosine similarity (skills, experience)      │
  │                        │    └─ Generate match scores                       │
  │                        │                          │                        │
  │◀─── MatchResult ───────│  {overall: 0.78, skills: 0.82, ...}              │
```

---

### Phase 3: Interview Session Start

```
Client                    API                     Services                  Storage
  │                        │                          │                        │
  │─── POST /sessions/start ───────────────────────────────────────────────────│
  │    {resume_id, jd_id,  │                          │                        │
  │     config}            │                          │                        │
  │                        │─── InterviewOrchestrator │                        │
  │                        │    │                     │                        │
  │                        │    ├─ HybridQuestionSelector                      │
  │                        │    │   ├─ QuestionBankService                     │
  │                        │    │   │   └─ Load from data/question_bank/*.jsonl│
  │                        │    │   ├─ Select 70% from curated bank            │
  │                        │    │   └─ QuestionGenerator (LLM) ───────────────▶│ LLM
  │                        │    │       └─ Generate 30% gap-filling questions  │
  │                        │    │                     │                        │
  │                        │    ├─ Create InterviewSession                     │
  │                        │    └─────────────────────▶│ MongoDB: interview_sessions
  │                        │                          │                        │
  │◀─── InterviewSession ──│  {session_id, questions: [...], status: "in_progress"}
```

---

### Phase 4: Answer Submission (Text or Voice)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ OPTION A: Text Answer                                                            │
└─────────────────────────────────────────────────────────────────────────────────┘
Client                    API                     Services                  Storage
  │                        │                          │                        │
  │─── POST /sessions/{id}/answer ─────────────────────────────────────────────│
  │    {question_id,       │                          │                        │
  │     answer_text}       │                          │                        │
  │                        │─── AnswerEvaluator ──────│                        │
  │                        │    ├─ Build evaluation prompt                     │
  │                        │    ├─ LLM-as-Judge evaluation ───────────────────▶│ LLM
  │                        │    │   └─ Score: correctness, depth, relevance    │
  │                        │    ├─ Hallucination detection                     │
  │                        │    └─ Generate feedback                           │
  │                        │                          │                        │
  │                        │─── Update session ──────▶│ MongoDB: interview_sessions
  │◀─── AnswerRecord ──────│  {score: 8, feedback: "...", next_question: {...}}│

┌─────────────────────────────────────────────────────────────────────────────────┐
│ OPTION B: Voice Answer                                                           │
└─────────────────────────────────────────────────────────────────────────────────┘
Client                    API                     Services                  Storage
  │                        │                          │                        │
  │─── POST /voice/stt ────────────────────────────────────────────────────────│
  │    (audio file)        │                          │                    S3: audio/
  │                        │─── Whisper STT ──────────│                        │
  │                        │    └─ Transcribe audio to text                    │
  │◀─── {transcript} ──────│                          │                        │
  │                        │                          │                        │
  │─── POST /sessions/{id}/answer (with transcript) ──│ (same as Option A)    │
```

---

### Phase 5: Interview Completion & Report Generation

```
Client                    API                     Services                  Storage
  │                        │                          │                        │
  │─── POST /sessions/{id}/end ────────────────────────────────────────────────│
  │                        │                          │                        │
  │                        │─── InterviewOrchestrator │                        │
  │                        │    │                     │                        │
  │                        │    ├─ Calculate final scores                      │
  │                        │    │   ├─ Technical score (avg of tech questions) │
  │                        │    │   ├─ Behavioral score                        │
  │                        │    │   ├─ Communication score                     │
  │                        │    │   └─ Overall weighted score                  │
  │                        │    │                     │                        │
  │                        │    ├─ LLM Summary Generation ────────────────────▶│ LLM
  │                        │    │   ├─ Strengths analysis                      │
  │                        │    │   ├─ Weaknesses/gaps                         │
  │                        │    │   └─ Hiring recommendation                   │
  │                        │    │                     │                        │
  │                        │    ├─ PDFReportGenerator │                        │
  │                        │    │   └─ ReportLab PDF ─────────────────────────▶│ S3: reports/
  │                        │    │                     │                        │
  │                        │    └─ Save report ──────▶│ MongoDB: reports       │
  │                        │                          │                        │
  │◀─── FullInterviewReport│  {scores, summary, recommendation, pdf_url}      │
```

---

### Phase 6: Report Retrieval

```
Client                    API                     Services                  Storage
  │                        │                          │                        │
  │─── GET /reports/{id} ──────────────────────────────────────────────────────│
  │                        │◀─────────────────────────│ MongoDB: reports       │
  │◀─── JSON Report ───────│                          │                        │
  │                        │                          │                        │
  │─── GET /reports/{id}/pdf ──────────────────────────────────────────────────│
  │                        │◀─────────────────────────│ S3: reports/{id}.pdf   │
  │◀─── PDF Binary ────────│                          │                        │
```

---

## Data Storage Summary

| Storage | Collections/Buckets | Purpose |
|---------|---------------------|---------|
| **MongoDB** | `resumes` | Parsed resume data + metadata |
| | `job_descriptions` | Parsed JD requirements |
| | `interviews` | Interview configurations |
| | `interview_sessions` | Live session state + answers |
| | `reports` | Final interview reports (JSON) |
| | `candidates` | Candidate profiles |
| **S3/MinIO** | `resumes/` | Original resume files (PDF/DOCX) |
| | `audio/` | Voice recordings from interviews |
| | `reports/` | Generated PDF reports |

---

## LLM Integration Points

| Service | LLM Usage | Provider |
|---------|-----------|----------|
| `DocumentProcessor` | Resume/JD parsing & enrichment | Ollama/vLLM |
| `QuestionGenerator` | Gap-filling question generation | Ollama/vLLM |
| `AnswerEvaluator` | LLM-as-Judge scoring + feedback | Ollama/vLLM |
| `InterviewOrchestrator` | Report summary generation | Ollama/vLLM |

---

## Authentication Flow

```
Client JWT Token
       │
       ▼
┌──────────────────┐
│ Auth Middleware  │
│ (src/api/middleware.py)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ decode_token()   │ ─── Invalid? ──▶ 401 Unauthorized
│ (src/core/auth.py)
└────────┬─────────┘
         │ Valid
         ▼
┌──────────────────┐
│ Check Role/Perms │ ─── No access? ──▶ 403 Forbidden
│ (require_role,   │
│  require_permission)
└────────┬─────────┘
         │ Authorized
         ▼
    Route Handler
    (AuthenticatedUser injected)
```

---

## API Endpoints Summary

| Endpoint | Method | Purpose | Auth Required |
|----------|--------|---------|---------------|
| `/health` | GET | Health check | No |
| `/api/v1/documents/resumes` | POST | Upload resume | Yes |
| `/api/v1/documents/job-descriptions` | POST | Create JD | Yes |
| `/api/v1/documents/match` | POST | Match resume to JD | Yes |
| `/api/v1/interviews` | POST | Create interview config | Yes |
| `/api/v1/sessions/start` | POST | Start interview session | Yes |
| `/api/v1/sessions/{id}/answer` | POST | Submit answer | Yes |
| `/api/v1/sessions/{id}/progress` | GET | Get session progress | Yes |
| `/api/v1/sessions/{id}/end` | POST | End interview | Yes |
| `/api/v1/questions/generate` | POST | Generate questions | Yes |
| `/api/v1/questions/evaluate` | POST | Evaluate answer | Yes |
| `/api/v1/voice/stt` | POST | Speech-to-text | Yes |
| `/api/v1/voice/tts` | POST | Text-to-speech | Yes |
| `/api/v1/reports/{id}` | GET | Get JSON report | Yes |
| `/api/v1/reports/{id}/pdf` | GET | Get PDF report | Yes |

---

## Role-Based Access Control

| Role | Permissions |
|------|-------------|
| `admin` | All permissions |
| `hiring_manager` | Create interviews, manage candidates, view/export reports, manage questions |
| `interviewer` | Conduct interviews, view sessions (limited), use voice |
| `candidate` | Participate in assigned session only, use voice |
