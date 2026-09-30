"""Tensor storage snapshots preserving strides, offsets and alias relationships."""
from __future__ import annotations
import torch


def snapshot(tensors):
    stores, views, keys, types = [], {}, {}, {}
    for name, tensor in tensors.items():
        if tensor.layout != torch.strided or tensor.is_conj() or tensor.is_neg():
            raise ValueError("Capture needs ordinary strided tensors; supply a custom repro for this layout")
        storage = tensor.untyped_storage()
        identity = (str(tensor.device), storage._cdata)
        if identity in types and types[identity] != tensor.dtype:
            raise ValueError("Mixed-dtype storage aliases require a custom repro")
        types[identity] = tensor.dtype
        if identity not in keys:
            keys[identity] = len(stores)
            count = storage.nbytes() // tensor.element_size()
            stores.append(tensor.detach().as_strided((count,), (1,), 0).cpu().clone())
        views[name] = {"storage": keys[identity], "shape": list(tensor.shape),
                       "stride": list(tensor.stride()), "offset": tensor.storage_offset()}
    return {"schema_version": 1, "storages": stores, "views": views}


def restore(payload, device):
    if payload.get("schema_version") != 1:
        raise ValueError("Unknown capture schema")
    stores = [s.to(device).clone() for s in payload["storages"]]
    return {name: stores[v["storage"]].as_strided(v["shape"], v["stride"], v["offset"])
            for name, v in payload["views"].items()}
