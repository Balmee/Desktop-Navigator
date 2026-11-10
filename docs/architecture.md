# AURA architecture baseline

## Design principles
1. Local-first inference and storage.
2. Explicit consent for screen/audio capture.
3. Perception, reasoning, memory and actions are separate modules.
4. All model calls go through a replaceable model interface.
5. The model proposes actions; a deterministic policy layer validates them.
6. Every action is auditable and can be stopped by the user.
7. Learn first through curated memory and feedback; fine-tune only on reviewed examples.

## Runtime data flow
Desktop/audio event -> Perception -> Context snapshot -> Orchestrator -> Model router -> Reasoning/response -> Text and speech output.

When screen context is requested:
1. Capture a single screen frame with explicit consent.
2. Extract relevant text and UI metadata in later phases.
3. Send the image and user question to a local vision-language model.
4. Store a short textual summary in working context, subject to retention settings.
5. Delete the temporary screenshot.

## Model strategy
- Initial visual-language model: Qwen3-VL 8B through Ollama.
- Later: independent speech-to-text, text-to-speech and embedding models.
- PyTorch is the framework for AURA-owned training, evaluation and specialised models.
- Ollama is an inference runtime, not the learning architecture itself.

## Safety boundary
No desktop control in Phase 1. Later action requests must pass a permission policy, risk classification, user approval where required, execution and post-action verification.
