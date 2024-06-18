# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from pinn.layers import (
    CellListNL,
    CutoffFunc,
    PolynomialBasis,
    GaussianBasis,
    AtomicOnehot,
    ANNOutput,
)

from pinn.networks.pinet import FFLayer, PILayer, IPLayer, ResUpdate
from pinn.networks.pinet2 import PIXLayer, ScaleLayer, OutLayer, DotLayer


class InvarLayer(tf.keras.layers.Layer):

    def __init__(self, pp_nodes, pi_nodes, ii_nodes, **kwargs):
        super().__init__()
        self.pi_layer = PILayer(pi_nodes, **kwargs)
        self.ii_layer = FFLayer(ii_nodes, use_bias=False, **kwargs)
        self.ip_layer = IPLayer()
        self.pp_layer = FFLayer(pp_nodes, use_bias=False, **kwargs)

    def call(self, tensors):

        ind_2, p1, basis = tensors

        i1 = self.pi_layer([ind_2, p1, basis])
        i1 = self.ii_layer(i1)
        p1 = self.ip_layer([ind_2, p1, i1])
        p1 = self.pp_layer(p1)
        return p1, i1


class EquiVarLayer(tf.keras.layers.Layer):

    def __init__(self, n_outs, weighted, **kwargs):

        super().__init__()

        self.pp_layer = FFLayer(n_outs, use_bias=False, **kwargs)
        self.pi_layer = PIXLayer(weighted=weighted, **kwargs)
        self.ip_layer = IPLayer()

        self.scale_layer = ScaleLayer()
        self.dot_layer = DotLayer(weighted=weighted)

    def call(self, tensors):

        ind_2, px, i1, diff = tensors
        
        ix = self.pi_layer([ind_2, px])
        ix = self.scale_layer([ix, i1])
        scaled_diff = self.scale_layer([diff[:, :, None], i1])
        ix = ix + scaled_diff
        px = self.ip_layer([ind_2, px, ix])
        
        dotted_px = self.dot_layer(px)
        
        return px, ix, dotted_px


class GCBlock(tf.keras.layers.Layer):
    def __init__(self, rank, weighted: bool, pp_nodes, pi_nodes, ii_nodes, **kwargs):
        super(GCBlock, self).__init__()
        self.rank = rank
        self.n_layers = int(rank // 2) + 1
        ppx_nodes = [pp_nodes[-1]]
        if rank >= 1:
            ii1_nodes = ii_nodes.copy()
            pp1_nodes = pp_nodes.copy()
            ii1_nodes[-1] *= self.n_layers
            pp1_nodes[-1] = ii_nodes[-1] * self.n_layers
            self.invar_p1_layer = InvarLayer(pp_nodes, pi_nodes, ii1_nodes, **kwargs)
            self.pp_layer = FFLayer(pp1_nodes, **kwargs)

        if rank >= 3:
            self.equivar_p3_layer = EquiVarLayer(ppx_nodes, weighted=weighted, **kwargs)

        if rank >= 5:
            self.equivar_p5_layer = EquiVarLayer(ppx_nodes, weighted=weighted, **kwargs)

        self.scale3_layer = ScaleLayer()
        self.scale5_layer = ScaleLayer()

    def call(self, tensors):

        ind_2, basis, px, diffs = tensors

        p1, i1 = self.invar_p1_layer([ind_2, px[0], basis])

        i1s = tf.split(i1, self.n_layers, axis=-1)
        px_list = [p1]
        ix_list = [i1]

        if self.rank >= 3:
            p3, i3, dotted_p3 = self.equivar_p3_layer(
                [ind_2, px[1], i1s[1], diffs[0]]
            )  # NOTE: use same i1 branch for diff_px and px, same result as separated i1
            px_list.append(dotted_p3)
            ix_list.append(i3)

        if self.rank >= 5:
            p5, i5, dotted_p5 = self.equivar_p5_layer([ind_2, px[2], i1s[2], diffs[1]])
            px_list.append(dotted_p5)
            ix_list.append(i5)

        p1t1 = self.pp_layer(
            tf.concat(
                px_list,
                axis=-1,
            )
        )

        pxt1 = tf.split(p1t1, self.n_layers, axis=-1)
        pxt1_list = [pxt1[0]]

        if self.rank >= 3:
            p3t1 = self.scale3_layer([p3, pxt1[1]])
            pxt1_list.append(p3t1)

        if self.rank >= 5:
            p5t1 = self.scale5_layer([p5, pxt1[2]])
            pxt1_list.append(p5t1)

        return pxt1_list, ix_list


class PreprocessLayer(tf.keras.layers.Layer):
    def __init__(self, rank, atom_types, rc):
        super(PreprocessLayer, self).__init__()
        self.rank = rank
        self.embed = AtomicOnehot(atom_types)
        self.nl_layer = CellListNL(rc)

    def call(self, tensors):
        tensors = tensors.copy()
        for k in ["elems", "dist"]:
            if k in tensors.keys():
                tensors[k] = tf.reshape(tensors[k], tf.shape(tensors[k])[:1])
        if "ind_2" not in tensors:
            tensors.update(self.nl_layer(tensors))
            tensors["p1"] = tf.cast(  # difference with pinet: prop->p1
                self.embed(tensors["elems"]), tensors["coord"].dtype
            )
            tensors["norm_diff"] = tensors["diff"] / tf.linalg.norm(tensors["diff"])
        
        if self.rank >= 3:
            tensors["p3"] = tf.zeros([tf.shape(tensors["ind_1"])[0], 3, 1])
        if self.rank >= 5:
            tensors["p5"] = tf.zeros([tf.shape(tensors["ind_1"])[0], 5, 1])
            diff = tensors["norm_diff"]
            x = diff[:, 0]
            y = diff[:, 1]
            z = diff[:, 2]
            x2 = x**2
            y2 = y**2
            z2 = z**2
            tensors["diff_p5"] = tf.stack(
                [
                    2 / 3 * x2 - 1 / 3 * y2 - 1 / 3 * z2,
                    2 / 3 * y2 - 1 / 3 * x2 - 1 / 3 * z2,
                    x * y,
                    x * z,
                    y * z,
                ],
                axis=1,
            )
        return tensors


class PiNet2(tf.keras.Model):
    """This class implements the Keras Model for the PiNet network."""

    def __init__(
        self,
        atom_types=[1, 6, 7, 8],
        rc=4.0,
        cutoff_type="f1",
        basis_type="polynomial",
        n_basis=4,
        gamma=3.0,
        center=None,
        pp_nodes=[16, 16],
        pi_nodes=[16, 16],
        ii_nodes=[16, 16],
        out_nodes=[16, 16],
        out_units=1,
        out_extra={},
        out_pool=False,
        act="tanh",
        depth=4,
        weighted=True,
        rank=2,
    ):
        """
        Args:
            atom_types (list): elements for the one-hot embedding
            pp_nodes (list): number of nodes for PPLayer
            pi_nodes (list): number of nodes for PILayer
            ii_nodes (list): number of nodes for IILayer
            out_nodes (list): number of nodes for OutLayer
            out_pool (str): pool atomic outputs, see ANNOutput
            depth (int): number of interaction blocks
            rc (float): cutoff radius
            basis_type (string): basis function, can be "polynomial" or "gaussian"
            n_basis (int): number of basis functions to use
            gamma (float or array): width of gaussian function for gaussian basis
            center (float or array): center of gaussian function for gaussian basis
            cutoff_type (string): cutoff function to use with the basis.
            act (string): activation function to use
            weighted (bool): whether to use weighted style
        """
        super(PiNet2, self).__init__()

        self.depth = depth
        self.rank = rank
        self.preprocess = PreprocessLayer(rank, atom_types, rc)
        self.cutoff = CutoffFunc(rc, cutoff_type)

        if basis_type == "polynomial":
            self.basis_fn = PolynomialBasis(n_basis)
        elif basis_type == "gaussian":
            self.basis_fn = GaussianBasis(center, gamma, rc, n_basis)

        if rank >= 1:
            self.res_update1 = [ResUpdate() for _ in range(depth)]
        if rank >= 3:
            self.res_update3 = [ResUpdate() for _ in range(depth)]
        if rank >= 5:
            self.res_update5 = [ResUpdate() for _ in range(depth)]
        self.gc_blocks = [
            GCBlock(rank, weighted, pp_nodes, pi_nodes, ii_nodes, activation=act)
            for _ in range(depth)
        ]
        self.out_layers = [OutLayer(out_nodes, out_units) for i in range(depth)]
        self.out_extra = out_extra
        for k, v in out_extra.items():
            setattr(self, f"{k}_out_layers", [OutLayer([], v) for i in range(depth)])
        self.ann_output = ANNOutput(out_pool)

    def call(self, tensors):
        """PiNet takes batches atomic data as input, the following keys are
        required in the input dictionary of tensors:

        - `ind_1`: [sparse indices](layers.md#sparse-indices) for the batched data, with shape `(n_atoms, 1)`;
        - `elems`: element (atomic numbers) for each atom, with shape `(n_atoms)`;
        - `coord`: coordintaes for each atom, with shape `(n_atoms, 3)`.

        Optionally, the input dataset can be processed with
        `PiNet.preprocess(tensors)`, which adds the following tensors to the
        dictionary:

        - `ind_2`: [sparse indices](layers.md#sparse-indices) for neighbour list, with shape `(n_pairs, 2)`;
        - `dist`: distances from the neighbour list, with shape `(n_pairs)`;
        - `diff`: distance vectors from the neighbour list, with shape `(n_pairs, 3)`;
        - `prop`: initial properties `(n_pairs, n_elems)`;

        Args:
            tensors (dict of tensors): input tensors

        Returns:
            output (tensor): output tensor with shape `[n_atoms, out_nodes]`
        """
        tensors = self.preprocess(tensors)

        fc = self.cutoff(tensors["dist"])
        basis = self.basis_fn(tensors["dist"], fc=fc)
        output = 0.0
        output_extra = {}
        for k in self.out_extra:
            output_extra[k] = 0.0
        px_list = [tensors["p1"]]
        diff_list = [tensors["norm_diff"]]
        if self.rank >= 3:
            px_list.append(tensors["p3"])
        if self.rank >= 5:
            px_list.append(tensors["p5"])
            diff_list.append(tensors["diff_p5"])

        for i in range(self.depth):
            px_list, ix_list = self.gc_blocks[i](
                [
                    tensors["ind_2"],
                    basis,
                    px_list,
                    diff_list
                ]
            )
            output = self.out_layers[i]([tensors["ind_1"], px_list[0], output])
            for k in self.out_extra:
                if k.startswith('p'):
                    output = getattr(self, f"{k}_out_layers")[i]([tensors["ind_1"], px_list[k], output[k]])
                else:
                    output_extra[k] = getattr(self, f"{k}_out_layers")[i]([tensors["ind_1"], px_list[0], output_extra[k]])
            if self.rank > 1:
                tensors["p1"] = self.res_update1[i]([tensors["p1"], px_list[0]])
            if self.rank > 3:
                tensors["p3"] = self.res_update3[i]([tensors["p3"], px_list[1]])
            if self.rank > 5:
                tensors["p5"] = self.res_update5[i]([tensors["p5"], px_list[2]])

        output = self.ann_output([tensors["ind_1"], output])
        if self.out_extra:
            return output, output_extra
        else:
            return output

