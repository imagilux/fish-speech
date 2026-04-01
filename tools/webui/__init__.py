from typing import Callable

import gradio as gr

from fish_speech.i18n import i18n
from tools.webui.inference import list_references, save_reference
from tools.webui.variables import HEADER_MD, TEXTBOX_PLACEHOLDER


_NONE_LABEL = "None"  # visible placeholder in dropdown


def build_app(inference_fct: Callable, api_url: str = "") -> gr.Blocks:

    def refresh_references(current: str = _NONE_LABEL) -> gr.update:
        ids = list_references(api_url)
        choices = [_NONE_LABEL] + ids
        # Preserve current selection if it's still in the list
        value = current if current in choices else _NONE_LABEL
        return gr.update(choices=choices, value=value)

    with gr.Blocks() as app:
        gr.Markdown(HEADER_MD)

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
                                reference_id = gr.Dropdown(
                                    label=i18n("Reference Voice"),
                                    choices=[_NONE_LABEL],
                                    value=_NONE_LABEL,
                                    allow_custom_value=True,
                                    info="Select a saved voice or type a custom ID",
                                )
                                refresh_btn = gr.Button(
                                    value="Refresh",
                                    scale=0,
                                    min_width=80,
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
                                    placeholder="Transcription of the reference audio",
                                    value="",
                                )

                            with gr.Accordion("Save as new voice", open=False):
                                with gr.Row():
                                    save_name = gr.Textbox(
                                        label="Voice Name",
                                        placeholder="e.g. my-voice",
                                        lines=1,
                                        scale=3,
                                    )
                                    save_btn = gr.Button(
                                        value="Save",
                                        variant="secondary",
                                        scale=0,
                                        min_width=80,
                                    )
                                gr.Markdown(
                                    "*Upload reference audio and enter its transcription above, then name and save it here.*"
                                )

            with gr.Column(scale=3):
                audio = gr.Audio(
                    label=i18n("Generated Audio"),
                    type="numpy",
                    interactive=False,
                )
                generate = gr.Button(
                    value="\U0001f3a7 " + i18n("Generate"),
                    variant="primary",
                )

        # Save new voice, then auto-refresh dropdown with the new voice selected
        def do_save(name: str, audio_path: str, ref_text: str) -> gr.update:
            save_reference(name, audio_path, ref_text, api_url)
            return refresh_references(current=name.strip())

        save_btn.click(
            do_save,
            inputs=[save_name, reference_audio, reference_text],
            outputs=[reference_id],
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
            [audio],
            concurrency_limit=1,
        )

    return app
