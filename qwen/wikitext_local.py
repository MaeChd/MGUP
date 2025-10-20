# Part of the implementation is borrowed and modified from huggingface/wikitext dataset, publicly available at https://huggingface.co/datasets/wikitext/blob/main/wikitext.py

import os
import datasets
from glob import glob

_CITATION = """\
@misc{merity2016pointer,
      title={Pointer Sentinel Mixture Models},
      author={Stephen Merity and Caiming Xiong and James Bradbury and Richard Socher},
      year={2016},
      eprint={1609.07843},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

_DESCRIPTION = """\
 The WikiText language modeling dataset is a collection of over 100 million tokens extracted from the set of verified
 Good and Featured articles on Wikipedia. The dataset is available under the Creative Commons Attribution-ShareAlike
 License.
"""
_HOMEPAGE = "https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/"
_LICENSE = "Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)"
_DATA_URL = "https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/wikitext"

class WikitextConfig(datasets.BuilderConfig):
    """BuilderConfig for Wikitext"""

    def __init__(self, data_url, local_dir=None, **kwargs):
        """BuilderConfig for Wikitext

        Args:
          data_url: `string`, url to the dataset (word or raw level)
          local_dir: `string`, optional path to the local directory containing the dataset
          **kwargs: keyword arguments forwarded to super.
        """
        super(WikitextConfig, self).__init__(version=datasets.Version("1.0.0"), **kwargs)
        self.data_url = data_url
        self.local_dir = local_dir

class Wikitext(datasets.GeneratorBasedBuilder):
    """WikiText dataset."""

    VERSION = datasets.Version("0.1.0")
    BUILDER_CONFIGS = [
        WikitextConfig(
            name="wikitext-103-v1",
            data_url=_DATA_URL + "/" + "wikitext-103-v1.zip",
            description="Word level dataset.",
        ),
        WikitextConfig(
            name="wikitext-2-v1",
            data_url=_DATA_URL + "/" + "wikitext-2-v1.zip",
            description="Word level dataset.",
        ),
        WikitextConfig(
            name="wikitext-103-raw-v1",
            data_url=_DATA_URL + "/" + "wikitext-103-raw-v1.zip",
            description="Raw level dataset.",
        ),
        WikitextConfig(
            name="wikitext-2-raw-v1",
            data_url=_DATA_URL + "/" + "wikitext-2-raw-v1.zip",
            description="Raw level dataset.",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({"text": datasets.Value("string")}),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        if self.config.local_dir:
            # Use local directory if provided
            data_dir = os.path.join(self.config.local_dir, self.config.name)
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={"data_files": os.path.join(data_dir, "test-*.parquet"), "split": "test"},
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={"data_files": os.path.join(data_dir, "train-*.parquet"), "split": "train"},
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={"data_files": os.path.join(data_dir, "validation-*.parquet"), "split": "validation"},
                ),
            ]
        else:
            # Original download logic
            data_file = dl_manager.download_and_extract(self.config.data_url)
            if self.config.name == "wikitext-103-v1":
                data_dir = os.path.join(data_file, "wikitext-103")
                return [
                    datasets.SplitGenerator(
                        name=datasets.Split.TEST,
                        gen_kwargs={"data_files": os.path.join(data_dir, "wiki.test.tokens"), "split": "test"},
                    ),
                    datasets.SplitGenerator(
                        name=datasets.Split.TRAIN,
                        gen_kwargs={"data_files": os.path.join(data_dir, "wiki.train.tokens"), "split": "train"},
                    ),
                    datasets.SplitGenerator(
                        name=datasets.Split.VALIDATION,
                        gen_kwargs={"data_files": os.path.join(data_dir, "wiki.valid.tokens"), "split": "validation"},
                    ),
                ]
            elif self.config.name == "wikitext-103-raw-v1":
                data_dir = os.path.join(data_file, "wikitext-103-raw")
                return [
                    datasets.SplitGenerator(
                        name=datasets.Split.TEST,
                        gen_kwargs={"data_files": os.path.join(data_dir, "wiki.test.raw"), "split": "test"},
                    ),
                    datasets.SplitGenerator(
                        name=datasets.Split.TRAIN,
                        gen_kwargs={"data_files": os.path.join(data_dir, "wiki.train.raw"), "split": "train"},
                    ),
                    datasets.SplitGenerator(
                        name=datasets.Split.VALIDATION,
                        gen_kwargs={"data_files": os.path.join(data_dir, "wiki.valid.raw"), "split": "validation"},
                    ),
                ]
            elif self.config.name == "wikitext-2-raw-v1":
                data_dir = os.path.join(data_file, "wikitext-2-raw")
                return [
                    datasets.SplitGenerator(
                        name=datasets.Split.TEST,
                        gen_kwargs={"data_files": os.path.join(data_dir, "wiki.test.raw"), "split": "test"},
                    ),
                    datasets.SplitGenerator(
                        name=datasets.Split.TRAIN,
                        gen_kwargs={"data_files": os.path.join(data_dir, "wiki.train.raw"), "split": "train"},
                    ),
                    datasets.SplitGenerator(
                        name=datasets.Split.VALIDATION,
                        gen_kwargs={"data_files": os.path.join(data_dir, "wiki.valid.raw"), "split": "validation"},
                    ),
                ]
            elif self.config.name == "wikitext-2-v1":
                data_dir = os.path.join(data_file, "wikitext-2")
                return [
                    datasets.SplitGenerator(
                        name=datasets.Split.TEST,
                        gen_kwargs={"data_files": os.path.join(data_dir, "wiki.test.tokens"), "split": "test"},
                    ),
                    datasets.SplitGenerator(
                        name=datasets.Split.TRAIN,
                        gen_kwargs={"data_files": os.path.join(data_dir, "wiki.train.tokens"), "split": "train"},
                    ),
                    datasets.SplitGenerator(
                        name=datasets.Split.VALIDATION,
                        gen_kwargs={"data_files": os.path.join(data_dir, "wiki.valid.tokens"), "split": "validation"},
                    ),
                ]

    def _generate_examples(self, data_files, split):
        """Yields examples."""
        if isinstance(data_files, str):
            # If data_files is a string (possibly with wildcard), load all matching files
            file_paths = glob(data_files)
        else:
            # Assume data_files is a list of file paths
            file_paths = data_files

        for file_path in file_paths:
            if file_path.endswith(".parquet"):
                import pyarrow.parquet as pq
                table = pq.read_table(file_path)
                for idx, row in enumerate(table.to_pylist()):
                    yield f"{file_path}_{idx}", {"text": row["text"]}
            else:
                with open(file_path, encoding="utf-8") as f:
                    for idx, row in enumerate(f):
                        if row.strip():
                            yield f"{file_path}_{idx}", {"text": row}
                        else:
                            yield f"{file_path}_{idx}", {"text": ""}
