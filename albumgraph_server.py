#! /usr/bin/env python
# -*- coding: utf-8
# author: Thomas Wood
# email : thomas@synpon.com

import urllib3
import certifi
import json
import pickle
import uuid
import os
import time

import networkx as nx
import numpy as np

from flask import Flask, jsonify, request
from flask_restful import Resource, Api

http = urllib3.PoolManager(cert_reqs="CERT_REQUIRED", ca_certs=certifi.where())
FAKE_DB = "./data/albumgraph_db.graphml"

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
    if res.status == 404:
        print("404 image not found")
    json_boxes = json.loads(res.data)
    if len(json_boxes) < 1:
        return None
    cls_boxes = json_boxes.get("cls_boxes")
    if cls_boxes is None:
        return None
    else:
        return pickle.loads(cls_boxes)[1:]


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
            "label"  : label,
            "id"     : uuid.uuid4().hex # generate unique random key for region
           }


########################################################
#                                                      #
#   THE FOLLOWING IS NOT THE RIGHT WAY TO DO IT.       #
#   I'M JUST ESTABLISHING THE STRUCTURE OF THE GRAPH.  #
#                                                      #
########################################################


#######################################################
"""
album_graph - images, regions, labels
image key is img_url
region key is some shit I'm going to make up, uuid prob
label key is the label. Only 80 of these, but we're adding
them on the fly people. Look alive.

we're going to make edges between the images and the
regions they contain

we're going to make edges between the labels and the
regions they represent

That's it. Make it functional. Make it to where it will work
through a flask API.

make_empty_graph or load_graphml depending on configs

add_image(img_url) - adds an image node using the url as it's key.

add_region(img_url, region) - adds a region node using a uuid as identifier.

add_label(img_url, region, label) - add a label node using label class as id.

"""
#######################################################


def make_empty_graph():
    return nx.Graph()

def load_graph(fname=FAKE_DB):
    if not os.path.exists(fname):
        return make_empty_graph()
    else:
        return nx.read_graphml(fname)

def save_graph(G):
    nx.write_graphml(G, FAKE_DB)

def add_image_vertex(G, img_url):
    """
    """
    G.add_node(
               img_url,
               updated_on = time.strftime("%d-%b-%Y %H:%M:%S", time.gmtime()),
               category   = "image"
              )


def add_region_vertex(G, region):
    """
    """
    region_id = region["id"]
    G.add_node(
               region_id,
               updated_on = time.strftime("%d-%b-%Y %H:%M:%S", time.gmtime()),
               category   = "region"
               )


def add_label_vertex(G, label):
    """
    """
    G.add_node(
               label,
               updated_on = time.strftime("%d-%b-%Y %H:%M:%S", time.gmtime()),
               category   = "label"
               )


def add_image_region_edge(G, region):
    """
    """
    img_url = region["img_url"]
    region_id = region["id"]
    G.add_edge(
               img_url,
               region_id,
               updated_on = time.strftime("%d-%b-%Y %H:%M:%S", time.gmtime()),
               category   = "image-to-region"
               )


def add_region_label_edge(G, region):
    """
    """
    region_id = region["id"]
    label = region["label"]
    G.add_edge(
               region_id,
               label,
               updated_on = time.strftime("%d-%b-%Y %H:%M:%S", time.gmtime()),
               category   = "region-to-label"
               )

def update_graph_new_image(G, img_url, idx2label):
    cls_boxes = detect(img_url)
    if cls_boxes is None:
        print("Got nothing back from detectron")
        print(img_url)
        return None
    # Add the image vertex.
    if img_url not in G.nodes:
        add_image_vertex(G, img_url)
    else:
        print("We already have this URL in the album")
        return None

    regions = parse_cls_boxes(img_url, cls_boxes, idx2label)
    if len(regions) < 1:
        print("None of the regions were worth a damn")
        print(img_url)
        return None

    # iterate through the regions we've detected.
    for region in regions:
        # Get the region_id and label for this region.
        label = region['label']
        region_id = region["id"]

        # Add the region vertex.
        if region_id not in G.nodes:
            add_region_vertex(G, region)
        else:
            print("This region vertex already exists. Huh. Weird")
            continue

        # Add the label vertex.
        if label not in G.nodes:
            add_label_vertex(G, label)

        # Add the image-region edge.
        if (img_url, region_id) not in G.edges:
            add_image_region_edge(G, region)

        # Add the region-label edge.
        if (region_id, label) not in G.edges:
            add_region_label_edge(G, region)


app = Flask(__name__)
api = Api(app)

G = load_graph()
idx2label = load_idx2label()

class UpdateAlbumGraph(Resource):
    def post(self):
        img_url = request.form["data"]
        update_graph_new_image(G, img_url, idx2label)
    def get(self):
        return {"graph":pickle.dumps(G)}

class SaveAlbumGraph(Resource):
    def post(self):
        save_graph(G)
    def get(self):
        return {"graph": pickle.dumps(G)}

api.add_resource(UpdateAlbumGraph, "/update_albumgraph")
api.add_resource(SaveAlbumGraph, "/save_albumgraph")

if __name__ == "__main__":
    app.run(use_reloader=False, debug=True, host="0.0.0.0", port=5923)
