"""Shared helpers for the localization baseline and the channel, high-norm-token, and
text-stream studies.

``config``, ``clustering``, ``io``, ``highnorm``, and ``spatial`` are pure ``numpy`` /
``scikit-learn`` and import anywhere, including the test suite, without a GPU.
``model_utils`` imports ``torch``, ``diffusers``, and ``transformers`` lazily inside
its functions, so importing the module itself stays cheap.
"""
