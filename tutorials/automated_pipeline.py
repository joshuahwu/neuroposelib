from neuroposelib.features import *
from neuroposelib import DataStruct as ds
import numpy as np
from neuroposelib import read, write, utils
from neuroposelib.embed import Watershed, Embed
from pathlib import Path


# TODO: Probably be like a demo/notebook, don't maintain this
def standard_features(
    pose,
    connectivity,
    paths,
):
    ## Feature calculation
    if Path(paths["out_path"] + "postural_feats.h5").exists():
        print("Found postural_feats.h5 file - Reading ...")

        # Read postural features from h5 file
        features, labels = read.features_h5(
            path="".join([paths["out_path"], "postural_feats.h5"])
        )
    else:
        # Reshape pose to get egocentric pose features
        ego_pose, ego_pose_labels = get_ego_pose(pose, connectivity.joint_names)
        ego_pose, ego_pose_labels = standard_scale(ego_pose, ego_pose_labels)

        # Calculating joint angles
        angles, angle_labels = get_angles(pose, connectivity.angles)

        # Collect all features together
        labels = ego_pose_labels + angle_labels
        features = np.concatenate([ego_pose, angles], axis=1)

        # Save postural features to h5 file
        write.features_h5(
            features, labels, path="".join([paths["out_path"], "postural_feats.h5"])
        )
    return features, labels


def run(
    features,
    labels,
    pose,
    id,
    connectivity,
    paths,
    params,
    meta,
    meta_by_frame,
    standardize=True,
):
    if Path(paths["out_path"] + "pca_feats.h5").exists():
        pc_feats, pc_labels = read.features_h5(
            path="".join([paths["out_path"], "pca_feats.h5"])
        )
    else:
        ## PCA and Waveletting
        feat_categories = ["ego_euc", "ang"]

        pc_feats, pc_labels = pca(
            features, labels, categories=feat_categories, n_pcs=5, method="fbpca"
        )

        wlet_feats, wlet_labels = wavelet(
            pc_feats,
            pc_labels,
            id,
            sample_freq=90,
            freq=np.linspace(0.5, 25, 25),
            #   freq = np.linspace(0.5,4.5,25)**2,
            w0=5,
        )

        pc_wlet, pc_wlet_labels = pca(
            wlet_feats,
            wlet_labels,
            categories=["wlet_" + cat for cat in feat_categories],
            n_pcs=5,
            method="fbpca",
        )

        pc_feats = np.hstack((pc_feats, pc_wlet))
        pc_labels += pc_wlet_labels
        del pc_wlet, pc_wlet_labels

        write.features_h5(
            pc_feats, pc_labels, path="".join([paths["out_path"], "pca_feats.h5"])
        )

    data_obj = ds.DataStruct(
        pose=pose,
        id=id,
        id_full=id,
        meta=meta,
        meta_by_frame=meta_by_frame,
        connectivity=connectivity,
    )

    data_obj.features = pc_feats
    data_obj = data_obj[:: params["downsample"], :]

    # Embedding using t-SNE
    embedder = Embed(
        embed_method=params["single_embed"]["method"],
        perplexity=params["single_embed"]["perplexity"],
        lr=params["single_embed"]["lr"],
    )
    data_obj.embed_vals = embedder.embed(data_obj.features, save_self=True)

    # Watershed clustering
    data_obj.ws = Watershed(
        sigma=params["single_embed"]["sigma"], max_clip=1, log_out=True, pad_factor=0.05
    )
    data_obj.data.loc[:, "Cluster"] = data_obj.ws.fit_predict(data=data_obj.embed_vals)

    return data_obj
