from transformers.integrations import WandbCallback
import panel as pn
import numpy as np
import wandb
import pandas as pd
import jiwer
import tempfile
import numbers
from pathlib import Path
from io import StringIO
import holoviews as hv
hv.extension("bokeh", logo=False)


class RecordAnalyzer:
    """Class to analyze and process records from a dataset.
    """

    def __init__(self):
        self.tokenizer = None

    def record_to_html(self, sample_record) -> StringIO:
        """Convert a sample record to HTML format.

        Args:
        ----
            sample_record (dict): A sample record containing audio and spectrogram data.

        Returns:
        -------
            StringIO: HTML representation of the sample record.

        """
        audio_array = np.array(sample_record["audio"]["array"])
        audio_sr = sample_record["audio"]["sampling_rate"]
        audio_data = sample_record['audio']
        audio_duration = len(audio_data) / 16000
        audio_spectrogram = np.array(sample_record["spectrogram"])

        bounds = (0, 0, audio_duration, audio_spectrogram.max())

        waveform_int = np.int16(audio_array * 32767)

        hv_audio = pn.pane.Audio(
            waveform_int, sample_rate=audio_sr, name="Audio", throttle=500
        )

        slider = pn.widgets.FloatSlider(end=audio_duration, visible=False, step=0.001)
        line_audio = hv.VLine(0).opts(color="black")
        line_spec = hv.VLine(0).opts(color="red")

        slider.jslink(hv_audio, value="time", bidirectional=True)
        slider.jslink(line_audio, value="glyph.location")
        slider.jslink(line_spec, value="glyph.location")

        time = np.linspace(0, audio_duration, num=len(audio_array))
        line_plot_hv = (
            hv.Curve((time, audio_array), ["Time (s)", "amplitude"]).opts(
                width=500, height=150, axiswise=True
            )
            * line_audio
        )

        hv_spec_gram = (
            hv.Image(
                audio_spectrogram, bounds=(bounds), kdims=["Time (s)", "Frequency (hz)"]
            ).opts(width=500, height=150, labelled=[], axiswise=True, color_levels=512)
            * line_spec
        )

        combined = pn.Row(hv_audio, hv_spec_gram, line_plot_hv, slider)
        audio_html = StringIO()
        combined.save(audio_html)
        return audio_html

    def dataset_to_records(self, dataset) -> pd.DataFrame:
        """
        Convert a dataset to a DataFrame of records.

        Args:
            dataset (list): List of records in the dataset.

        Returns:
            pd.DataFrame: DataFrame containing the processed records.
        """
        records = []
        for item in dataset:
            record = {}
            audio_data = item['input_features']
            audio_duration = len(audio_data) / 16000
            record["audio_with_spec"] = wandb.Html(self.record_to_html(item))
            record["sentence"] = item["sentence"]
            record["length"] = audio_duration
            records.append(record)
        records = pd.DataFrame(records)
        return records

    def decode_predictions(self, predictions, tokenizer) -> list:
        """Decode model predictions into human-readable format.

        Args:
        ----
            predictions (object): Predictions generated by the model.

        Returns:
        -------
            list: Decoded predictions.

        """
        pred_ids = predictions.predictions
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        return pred_str

    def compute_measures(self, predictions, labels) -> pd.DataFrame:
        """Compute evaluation measures for model predictions.

        Args:
        ----
            predictions (list): List of model predictions.
            labels (list): List of ground truth labels.

        Returns:
        -------
            pd.DataFrame: DataFrame containing computed evaluation measures.

        """
        measures = [
            jiwer.compute_measures(ls, ps) for ps, ls in zip(predictions, labels)
        ]
        measures_df = pd.DataFrame(measures)[
            ["wer", "hits", "substitutions", "deletions", "insertions"]
        ]
        return measures_df


class WandbProgressResultsCallback(WandbCallback):
    """Callback class for logging training progress to Weights & Biases.
    """

    def __init__(self, trainer, sample_dataset, tokenizer):
        """Initialize the WandbProgressResultsCallback instance.

        Args:
        ----
            trainer: The instance of the Trainer class.
            sample_dataset: The dataset used for logging sample predictions.
            tokenizer: The tokenizer instance for decoding predictions.

        """
        super().__init__()
        self.trainer = trainer
        self.sample_dataset = sample_dataset
        self.records_analyzer = RecordAnalyzer()
        self.records_df = self.records_analyzer.dataset_to_records(sample_dataset)
        self.tokenizer = tokenizer

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Log training progress to Weights & Biases.

        Args:
        ----
            args: Arguments passed to the callback.
            state: The current state of the training.
            control: Control parameters for the callback.
            model: The trained model.
            logs: Additional logs.
            **kwargs: Additional keyword arguments.

        Returns:
        -------
            The table containing sample predictions.

        """
        super().on_log(args, state, control, model, logs)
        predictions = self.trainer.predict(self.sample_dataset)
        predictions = self.records_analyzer.decode_predictions(
            predictions, self.tokenizer
        )
        measures_df = self.records_analyzer.compute_measures(
            predictions, self.records_df["sentence"].tolist()
        )
        records_df = pd.concat([self.records_df, measures_df], axis=1)
        records_df["prediction"] = predictions
        records_df["step"] = state.global_step
        records_table = self._wandb.Table(dataframe=records_df)
        self._wandb.log({"sample_predictions": records_table})

    def on_save(self, args, state, control, model=None, tokenizer=None, **kwargs):
        """Save the trained model as an artifact in Weights & Biases.

        Args:
        ----
            args: Arguments passed to the callback.
            state: The current state of the training.
            control: Control parameters for the callback.
            model: The trained model.
            tokenizer: The tokenizer instance.
            **kwargs: Additional keyword arguments.

        """
        if self._wandb is None:
            return
        if self._log_model and self._initialized and state.is_world_process_zero:
            with tempfile.TemporaryDirectory() as temp_dir:
                self.trainer.save_model(temp_dir)
                metadata = (
                    {
                        k: v
                        for k, v in dict(self._wandb.summary).items()
                        if isinstance(v, numbers.Number) and not k.startswith("_")
                    }
                    if not args.load_best_model_at_end
                    else {
                        f"eval/{args.metric_for_best_model}": state.best_metric,
                        "train/total_floss": state.total_flos,
                    }
                )
                artifact = self._wandb.Artifact(
                    name=f"model-{self._wandb.run.id}", type="model", metadata=metadata
                )
                for f in Path(temp_dir).glob("*"):
                    if f.is_file():
                        with artifact.new_file(f.name, mode="wb") as fa:
                            fa.write(f.read_bytes())
                self._wandb.run.log_artifact(artifact)
