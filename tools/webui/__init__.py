from typing import Callable

import gradio as gr

from fish_speech.i18n import i18n
from tools.webui.inference import list_references
from tools.webui.variables import HEADER_MD, TEXTBOX_PLACEHOLDER


_VALID_THEMES = {"light", "dark"}

_NONE_CHOICE = ""  # empty string = no reference selected


def build_app(
    inference_fct: Callable, theme: str = "light", api_url: str = ""
) -> gr.Blocks:
    if theme not in _VALID_THEMES:
        theme = "light"

    def refresh_references(current: str = _NONE_CHOICE) -> gr.update:
        ids = list_references(api_url)
        choices = [_NONE_CHOICE] + ids
        # Preserve current selection if it's still in the list
        value = current if current in choices else _NONE_CHOICE
        return gr.update(choices=choices, value=value)

    with gr.Blocks(theme=gr.themes.Base()) as app:
        gr.Markdown(HEADER_MD)

        # Use light theme by default
        app.load(
            None,
            None,
            js=f"() => {{const params = new URLSearchParams(window.location.search);if (!params.has('__theme')) {{params.set('__theme', '{theme}');window.location.search = params.toString();}}}}"
        )

        # Inference
        with gr.Row():
            with gr.Column(scale=3):
                text = gr.Textbox(
                    label=i18n("Input Text"), placeholder=TEXTBOX_PLACEHOLDER, lines=10
                )

                with gr.Row():
                    with gr.Column():
                        with gr.Tab(label=i18n("Advanced Config")):
                            with gr.Row():
                                chunk_length = gr.Slider(
                                    label=i18n("Iterative Prompt Length, 0 means off"),
                                    minimum=100,
                                    maximum=400,
                                    value=300,
                                    step=8,
                                )

                                max_new_tokens = gr.Slider(
                                    label=i18n(
                                        "Maximum tokens per batch, 0 means no limit"
                                    ),
                                    minimum=0,
                                    maximum=2048,
                                    value=0,
                                    step=8,
                                )

                            with gr.Row():
                                top_p = gr.Slider(
                                    label="Top-P",
                                    minimum=0.7,
                                    maximum=0.95,
                                    value=0.8,
                                    step=0.01,
                                )

                                repetition_penalty = gr.Slider(
                                    label=i18n("Repetition Penalty"),
                                    minimum=1,
                                    maximum=1.2,
                                    value=1.1,
                                    step=0.01,
                                )

                            with gr.Row():
                                temperature = gr.Slider(
                                    label="Temperature",
                                    minimum=0.7,
                                    maximum=1.0,
                                    value=0.8,
                                    step=0.01,
                                )
                                seed = gr.Number(
                                    label="Seed",
                                    info="0 means randomized inference, otherwise deterministic",
                                    value=0,
                                )

                        with gr.Tab(label=i18n("Reference Audio")):
                            with gr.Row():
                                gr.Markdown(
                                    i18n(
                                        "5 to 10 seconds of reference audio, useful for specifying speaker."
                                    )
                                )
                            with gr.Row():
                                reference_id = gr.Dropdown(
                                    label=i18n("Reference Voice"),
                                    choices=[_NONE_CHOICE],
                                    value=_NONE_CHOICE,
                                    allow_custom_value=True,
                                    info="Select a pre-loaded voice or type a custom ID",
                                )
                                refresh_btn = gr.Button(
                                    value="\U0001f504",  # 🔄
                                    scale=0,
                                    min_width=48,
                                )

                            with gr.Row():
                                use_memory_cache = gr.Radio(
                                    label=i18n("Use Memory Cache"),
                                    choices=["on", "off"],
                                    value="on",
                                )

                            with gr.Row():
                                reference_audio = gr.Audio(
                                    label=i18n("Reference Audio"),
                                    type="filepath",
                                )
                            with gr.Row():
                                reference_text = gr.Textbox(
                                    label=i18n("Reference Text"),
                                    lines=1,
                                    placeholder="在一无所知中，梦里的一天结束了，一个新的「轮回」便会开始。",
                                    value="",
                                )

            with gr.Column(scale=3):
                with gr.Row():
                    error = gr.HTML(
                        label=i18n("Error Message"),
                        visible=True,
                    )
                with gr.Row():
                    audio = gr.Audio(
                        label=i18n("Generated Audio"),
                        type="numpy",
                        interactive=False,
                        visible=True,
                    )

                with gr.Row():
                    with gr.Column(scale=3):
                        generate = gr.Button(
                            value="\U0001f3a7 " + i18n("Generate"),
                            variant="primary",
                        )

        # Refresh reference list on button click (preserves current selection)
        refresh_btn.click(
            refresh_references,
            inputs=[reference_id],
            outputs=[reference_id],
        )

        # Auto-refresh reference list on page load
        app.load(refresh_references, inputs=[], outputs=[reference_id])

        # Submit
        generate.click(
            inference_fct,
            [
                text,
                reference_id,
                reference_audio,
                reference_text,
                max_new_tokens,
                chunk_length,
                top_p,
                repetition_penalty,
                temperature,
                seed,
                use_memory_cache,
            ],
            [audio, error],
            concurrency_limit=1,
        )

    return app
