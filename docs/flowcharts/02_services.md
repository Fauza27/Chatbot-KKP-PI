# Flowchart: `src/services/`

## `ai_services.py` — Otak Utama Chatbot

File ini adalah **orchestrator** utama: mengelola sesi user dan mengarahkan setiap pesan ke jalur yang benar.

```
chat(query, session_id)
        │
        ├─ Query kosong? ──► Return error "empty_query"
        ├─ No session_id? ──► Return error "missing_session_id"
        │
        ▼
get_or_create_memory(session_id)
  ├─ TTL cleanup: buang sesi idle > SESSION_CLEANUP_INTERVAL
  ├─ Jika sesi ada → perbarui timestamp
  └─ Jika baru → buat ConversationMemory, cek LRU cap
        │
        ▼
memory.add_user_turn(question)
        │
        ▼
_classifier.classify(question, memory)
        │
   ┌────┴──────────────────────┐
   ▼                           ▼                         ▼
CONVERSATIONAL          CLARIFICATION            NEEDS_RETRIEVAL
   │                           │                         │
invoke_conversational    invoke_clarification    _handle_retrieval()
(tanpa retrieval)        (konteks lama)               │
   │                           │               reformulate_query()
   │                           │               run_retrieval()
   │                           │               invoke_with_history()
   └───────────────────────────┴─────────────────────────┘
                               │
                               ▼
                   memory.add_assistant_turn()
                               │
                               ▼
              Return: {answer, num_docs, sources,
                       intent, confidence, reasoning}
```

### Manajemen Sesi (Session Store)

```
_session_store = { session_id: (ConversationMemory, last_access_ts) }

Setiap akses:
  ├─ _evict_idle_sessions()  → hapus yang idle > TTL
  └─ _evict_lru_if_full()    → hapus terlama jika > MAX_ACTIVE_SESSIONS
```
