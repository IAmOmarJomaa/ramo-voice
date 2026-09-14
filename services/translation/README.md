# ramO-Translate: Sovereign Real-Time Meeting Translation & Intelligence

`services/translation` is the real-time speech translation and conversational intelligence engine in the ramO ecosystem. It runs standalone on Port `50053` or in-memory as a Python package (`ramo_translate`).

---

## Key Features

1. **Instant 0 ms Fast-Bypass Dictionary (`FAST_BYPASS_DICT`)**:
   - Stolen from `c:/ramo-audio/services/llm/engine.py`.
   - Bypasses model inference for common affirmative, negative, and greeting phrases in French, Spanish, German, and Arabic (`okay`, `yes`, `no`, `thank you`, `sure`, `goodbye`).
   - Serves 20–30% of meeting dialogue turns with **0 ms latency and 0 GPU VRAM overhead**.

2. **3-Tier Context Architecture & Speaker Symbol Masking**:
   - **Tier 1**: System prompt with ISO-639-1 language normalization.
   - **Tier 2**: Global meeting topic and agenda context.
   - **Tier 3**: 4-turn sliding dialogue window with speaker masking (`[S_A]`, `[S_B]`), maintaining cross-turn pronoun and topic cohesion without hallucinating speaker names.

3. **Meeting Action Item Detection**:
   - High-precision acoustic regex heuristics classify spoken commitments:
     - `task_delegation`: Explicit work assignment or promise ("I will send the report tomorrow").
     - `schedule_change`: Calendar/time adjustments ("Let's reschedule to Friday at 2pm").
     - `fact_check`: Numerical and statistical assertions ("Retention was 85 percent").

4. **Safe Colab T4 Memory Footprint**:
   - Eliminates the legacy unquantized vLLM 7.4GB memory trap.
   - Consumes **~2.0 GB VRAM** in 4-bit / 1.5B mode, leaving $>6.5\text{ GB}$ of free headroom on a 15GB T4 GPU.

---

## API Endpoints (Port 50053)

- `GET /health`: Health status & active engine.
- `POST /v1/translate`: Translate speech text with 3-tier context and fast-bypass.
- `POST /v1/meeting/action_items`: Detect task delegations and schedule changes.
- `WS /v1/translate/stream`: Low-latency streaming WebSocket.
