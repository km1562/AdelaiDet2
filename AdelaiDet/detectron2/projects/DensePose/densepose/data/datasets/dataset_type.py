# Copyright (c) Facebook, Inc. and its affiliates.

from enum import Enum


class DatasetType(Enum):
    """
    Dataset type, mostly used for datasets that contain datas to bootstrap models on
    """

    VIDEO_LIST = "video_list"
