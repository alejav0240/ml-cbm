# ML worker structure

The worker is organized by execution responsibility:

```text
worker/
├── entrypoints/
│   ├── orchestrator_service.py   # HTTP boundary used by backend
│   └── analyze_video.py          # application entrypoint for one video
└── pipeline/
    ├── media.py                  # ingestion, FFmpeg and synchronization
    └── media_pipeline.py         # ordered multimedia pipeline coordinator
```

The pipeline stages remain explicit in `media_pipeline.py`:

1. extraction and normalization;
2. segmentation and stable-window indexing;
3. optional pose extraction;
4. acoustic and vocalization features;
5. movement and interaction features;
6. multimodal artifact validation;
7. inference and DEMUCA projection;
8. callback with the complete run result.

The files at the repository root (`analyze_video.py`, `orchestrator_service.py`,
`media.py`, and `media_pipeline.py`) are compatibility entrypoints. Existing
Docker commands, scripts, and tests can continue using them while new code
imports the structured modules under `worker`.
