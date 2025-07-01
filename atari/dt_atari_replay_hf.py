import datasets
import torch
import numpy as np
import random
from dataclasses import dataclass

# Loads dataset from local files
# Refer to https://huggingface.co/datasets/edbeeching/decision_transformer_gym_replay/blob/main/decision_transformer_gym_replay.py
# And https://huggingface.co/docs/datasets/v2.11.0/en/loading#local-loading-script for loading from local script file

# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)

_DESCRIPTION = """ \
Testing an Atari DT replay script.
"""
_BASE_URL = "https://huggingface.co/datasets/moodlep/dt_atari_replay_hf/resolve/main"
_DATA_URL = f"{_BASE_URL}/trajectories.npy"

_HOMEPAGE = "https://github.com/kzl/decision-transformer/blob/e2d82e68f330c00f763507b3b01d774740bee53f/atari/readme-atari.md"

_LICENSE = "MIT"


class dt_atari_replay_hf(datasets.GeneratorBasedBuilder):
    def _info(self):
        features = datasets.Features(
            {
                "observations": datasets.Sequence(datasets.Array3D(shape=(4, 84, 84),
                                                                                     dtype='int64')),
                "actions": datasets.Sequence(datasets.Value("int32")),
                "rewards": datasets.Sequence(datasets.Value("float32")),
                "dones": datasets.Sequence(datasets.Value("bool")),
            }
        )

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            # Here we define them above because they are different between the two configurations
            features=features,
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
        )

    def _split_generators(self, dl_manager):
        # This is called before _generate_examples and passes the folder to that fnc.
        urls = _DATA_URL
        data_dir = dl_manager.download_and_extract(urls)
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir,
                    "split": "train",
                },
            )
        ]

    def _generate_examples(self, filepath, split):
        # Load from local trajectory files
        # A LINHA ABAIXO FOI CORRIGIDA
        trajectories = np.load(filepath, allow_pickle=True)

        for idx, traj in enumerate(trajectories):
            yield idx, {
                "observations": traj["observations"],
                "actions": traj["actions"],
                "rewards": np.expand_dims(traj["rewards"], axis=1),
                "dones": np.expand_dims(traj["dones"], axis=1),
            }
