#! /usr/bin/env python
# -*- coding: utf-8
# author: Thomas Wood
# email : thomas@synpon.com

import urllib3
import certifi
import json
import pickle
import networkx as nx
import numpy as np

http = urllib3.PoolManager(cert_reqs="CERT_REQUIRED", ca_certs=certifi.where())


def detect(img_url, detectron_url="0.0.0.0:8085/detectron"):
    """
    Function:
               detect( img_url )
    Arguments:
               img_url       - an image url
               detectron_url - the host:port where the detectron service
                               is listening
    Returns:
               a list of length 80 where the index of the list represents
               the object detected and the n X 5 array at that position
               represents [x_min, x_max, y_min, y_max, cls_prob] and the
               number of rows represent the number of instances of that
               object detected in that image.
    """
    res = http.request("POST", detectron_url, fields={"data": img_url})
    return pickle.loads(json.loads(res.data).get("cls_boxes"))[1:]


def load_idx2label(fname="./idx_to_label.json"):
    """
    """
    with open(fname, 'r') as f:
        d = json.load(f)

    return {int(x):d[x] for x in d}


def parse_cls_boxes(img_url, cls_boxes, idx2label, score_threshold=0.75):
    """
    """
    # Okay. What do we want to do?
    # We want to transform our numeric description into a list of objects
    # that have pertinent information about the region, basically the region
    # nodes in the graph, along with the labels, which we get from the
    # idx2label mapping.

    # Make sure our mapping and cls_boxes have the same size
    assert len(idx2label) == len(cls_boxes)

    regions = []
    # iterate over the boxes
    for k, cls_box in enumerate(cls_boxes):
        label = idx2label[k]
        n, m = cls_box.shape
        assert m == 5
        if n > 0:
            for kk in range(n):
                region = parse_region(cls_box[kk,:], img_url, label)
                score = region["score"]
                if score >= score_threshold:
                    regions.append(region)

    return regions


def rounder(x):
    return int(np.round(x))


def parse_region(region, img_url, label):
    return {
            "x_max"  : rounder(region[0]),
            "x_min"  : rounder(region[1]),
            "y_max"  : rounder(region[2]),
            "y_min"  : rounder(region[3]),
            "score"  : region[4],
            "img_url": img_url,
            "label"  : label
           }


########################################################
#                                                      #
#   THE FOLLOWING IS NOT THE RIGHT WAY TO DO IT.       #
#   I'M JUST ESTABLISHING THE STRUCTURE OF THE GRAPH.  #
#                                                      #
########################################################

