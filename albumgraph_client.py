#! /usr/bin/env python
# -*- coding: utf-8
# author: Thomas Wood
# email: thomas@synpon.com

import urllib3
import certifi
import json
import pickle
import time
import numpy as np

np.random.seed(273611)

http = urllib3.PoolManager(cert_reqs="CERT_REQUIRED", ca_certs=certifi.where())

albumgraph_url = "0.0.0.0:5923"
save_url = albumgraph_url + "/save_albumgraph"
update_url = albumgraph_url + "/update_albumgraph"


def update_albumgraph(img_url):
    """
    Function:
               update_albumgraph( img_url )
    Arguments:
               img_url - an image url
    Returns:
               True if status code is from albumgraph_server is 200, False
               otherwise.
    """
    r = http.request("POST", update_url, fields={"data": img_url})
    if r.status == 200:
        return True
    else:
        return False

def save_albumgraph():
    """
    Function:
               save_albumgraph()
    Arguments:
               None
    Returns:
               True if status code is from albumgraph_server is 200, False
               otherwise.
    """
    r = http.request("POST", save_url, fields={"data": ""})
    if r.status == 200:
        return True
    else:
        return False

def get_albumgraph():
    """
    Function:
               get_albumgraph()
    Arguments:
               None
    Returns:
               The NetworkX graph containing the AlbumGraph or None if something
               went wrong.
    """
    r = http.request("GET", save_url)
    if r.status == 200:
        return pickle.loads(json.loads(r.data)["graph"])
    else:
        print("Something went wrong.")
        return None

def process_album(album_urls):
    """
    Function:
               process_album( album_urls )
    Arguments:
               album_urls - A list of image URLs
    Returns:
               None
    """
    for img_url in album_urls:
        update_albumgraph(img_url)
        time.sleep(0.5)


def generate_fake_album(n_samples=100, max_id=10000):
    """
    Function:
               generate_fake_album( img_url )
    Arguments:
               n_samples - the number of samples to take
               max_id    - the maximum id that determines range we sample
    Returns:
               A list of image URLs
    """
    vg_base_url = "https://cs.stanford.edu/people/rak248/VG_100K_2/"
    album_ids = list(np.random.choice(max_id, n_samples))
    return [vg_base_url + "{}.jpg".format(x) for x in album_ids]

def main():
    """
    Function:
               main()
    Arguments:
               None
    Returns:
               None
    """
    album_urls = generate_fake_album()
    process_album(album_urls)
    save_albumgraph()
    G = get_albumgraph()
    print(len(G.nodes))


if __name__ == "__main__":
    main()
